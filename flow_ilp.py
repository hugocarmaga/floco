import gurobipy as gp
from gurobipy import GRB
from math import log
from collections import defaultdict
import counts_to_probabs as ctp
from time import perf_counter
import sys
import numpy as np
import random
from scipy.special import logsumexp
import scipy.stats


def bounds_and_probs(length, coverage, bins, alpha, beta, epsilon, subsampling_dist, diff_cutoff):
    if bins is None:
        nbins = 0
    else:
        binsize, arr_bins = bins[0]
        nbins = int(np.floor(arr_bins.size * binsize / subsampling_dist))

    if nbins <= 1:
        return ctp.cn_probs(alpha, beta, epsilon,
            length, coverage, length, [coverage], diff_cutoff)
    else:
        sampled_bins = random.sample(arr_bins.tolist(), nbins)
        return ctp.cn_probs(alpha, beta, epsilon,
            length, coverage, binsize, sampled_bins, diff_cutoff)


def ilp(nodes, edges, coverages, alpha, beta, rlen_params, outfile, source_prob = -20, cheap_source = -2, epsilon = 0.3):
    '''Function to formulate and solve the ILP for the flow network problem.'''
    try:
        i_start = perf_counter()
        # Create a new model
        model = gp.Model("CN flow ILP")

        ### Define variables

        # Copy number (graph flow) variable
        cn = {} # node: model.addVar(vtype = GRB.INTEGER, lb = 0, name = "cn_"+node)  for node in nodes}

        # Probability of cn as piecewise linear function
        p_cn = {node: model.addVar(vtype = GRB.CONTINUOUS, lb = - GRB.INFINITY, ub = 0, name = "p_cn_"+node) for node in nodes}

        # Edge flow on the two sides of the node - or just one variable for all edges
        edge_flow = {edges[edge]: model.addVar(vtype=GRB.INTEGER, lb = 0, name = edge) for edge in edges}

        #Get different sets of nodes, with edges on both sides or just on the left/right side
        double_sides = {}
        free_left_side = {}
        free_right_side = {}
        free_both = {}

        # Set of non-empty nodes.
        nonempty_nodes = { node for node in nodes if nodes[node].clipped_len() > 0 }
        # Super-edges flow on the two sides of the node
        source_left = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "source_left_"+node) for node in nonempty_nodes}
        source_right = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "source_right_"+node) for node in nonempty_nodes}
        sink_left = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "sink_left_"+node) for node in nonempty_nodes}
        sink_right = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "sink_right_"+node) for node in nonempty_nodes}

        model.update()
        print("Using secondary branch!")

        ### Constraints

        # Some edge preprocessing to be able to get all edges per side for each node
        l_edges_in = defaultdict(list)
        r_edges_in = defaultdict(list)
        l_edges_out = defaultdict(list)
        r_edges_out = defaultdict(list)

        # Create variables needed for reverse edge flow penalty (penalize the usage of the same node in opposite directions)
        C = 10000
        x1 = defaultdict()
        x2 = defaultdict()
        pen = 0.8 * source_prob
        diff_cutoff = abs(4 * source_prob)
        subsampling_dist = max(1000, scipy.stats.skewnorm.mean(*rlen_params))
        flow_penalty = gp.LinExpr()

        for k1 in edges:
            '''Iterate over the edges and add them to the correct side of the respective nodes'''
            r_edges_out[edges[k1].node1].append(edges[k1]) if edges[k1].strand1 else l_edges_out[edges[k1].node1].append(edges[k1])
            l_edges_in[edges[k1].node2].append(edges[k1]) if edges[k1].strand2 else r_edges_in[edges[k1].node2].append(edges[k1])

            # Iterate over the edge dictionary to add x1/x2 variables and constraints
            k2 = edges[k1].echo_reverse()
            e2 = edges.get(k2)
            if e2 and k1 < k2:
                x1[k1] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x1_"+k1)
                x2[k1] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x2_"+k1)
                model.addConstr(C * x1[k1] >= edge_flow[edges[k1]], "x1_flow_"+k1)
                model.addConstr(C * x2[k1] >= edge_flow[edges[k2]], "x2_flow_"+k1)
                model.addConstr(x1[k1] + x2[k1] >= 1, "x_sum_flow_"+k1)
                flow_penalty.add(x1[k1] + x2[k1] - 1, pen)

        # Create variable for later statistics on copy number concordance with the individual node probability
        likeliest_CNs = {}

        # Iterate over all nodes to define the constraints
        for node in nodes:
            if nodes[node].clipped_len() > 0:
                # Add nodes to the respective "edge sides" dictionary
                if (l_edges_in.get(node) or l_edges_out.get(node)) and (r_edges_in.get(node) or r_edges_out.get(node)):
                    double_sides[node] = node
                else:
                    if l_edges_in.get(node) or l_edges_out.get(node):
                        free_right_side[node] = node
                    elif r_edges_in.get(node) or r_edges_out.get(node):
                        free_left_side[node] = node
                    else:
                        free_both[node] = node

                lower_bound, y = bounds_and_probs(nodes[node].clipped_len(), coverages[node], nodes[node].bins,
                    alpha, beta, epsilon, subsampling_dist, diff_cutoff)
                upper_bound = lower_bound + len(y)
                x = list(range(lower_bound, upper_bound))
                assert len(x)==len(y), "{} is not the same length as {}".format(x,y)

                likeliest_CNs[node] = lower_bound + np.argmax(y)

                cn[node] = model.addVar(vtype = GRB.INTEGER, lb = lower_bound, ub = upper_bound - 1,  name = "cn_"+node)
                model.addGenConstrPWL(cn[node], p_cn[node], x, y, "PWLConstr_"+node)

                # Flow conservation constraints
                model.addConstr(source_left[node] + sum(edge_flow[e] for e in l_edges_in[node]) == sink_right[node] + sum(edge_flow[e] for e in r_edges_out[node]), "flow_left_" +node)
                model.addConstr(source_right[node] + sum(edge_flow[e] for e in r_edges_in[node]) == sink_left[node] + sum(edge_flow[e] for e in l_edges_out[node]), "flow_right_" +node)
                model.addConstr(source_left[node] + source_right[node] + sum(edge_flow[e] for e in l_edges_in[node]) + sum(edge_flow[e] for e in r_edges_in[node]) == cn[node], "flow_in_" +node)
                model.addConstr(sink_left[node] + sink_right[node] + sum(edge_flow[e] for e in r_edges_out[node]) + sum(edge_flow[e] for e in l_edges_out[node]) == cn[node], "flow_out_" +node)

                # Constraints to add a penalty to the flow when using edges in the same node end in opposite directions
                x1[node+"_right"] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x1_right_"+node)
                x2[node+"_right"] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x2_right_"+node)
                model.addConstr(C * x1[node+"_right"] >= source_right[node], "super_right_in_"+node)
                model.addConstr(C * x2[node+"_right"] >= sink_right[node], "super_right_out_"+node)
                model.addConstr(x1[node+"_right"] + x2[node+"_right"] >= 1, "x_sum_right_"+node)
                flow_penalty.add(x1[node+"_right"] + x2[node+"_right"] - 1, pen)

                x1[node+"_left"] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x1_left_"+node)
                x2[node+"_left"] = model.addVar(vtype = GRB.INTEGER, lb = 0, ub = 1, name = "x2_left_"+node)
                model.addConstr(C * x1[node+"_left"] >= source_left[node], "super_left_in_"+node)
                model.addConstr(C * x2[node+"_left"] >= sink_left[node], "super_left_out_"+node)
                model.addConstr(x1[node+"_left"] + x2[node+"_left"] >= 1, "x_sum_left_"+node)
                flow_penalty.add(x1[node+"_left"] + x2[node+"_left"] - 1, pen)

            else:

                cn[node] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "cn_"+node)
                model.addConstr(sum(edge_flow[e] for e in l_edges_in[node]) == sum(edge_flow[e] for e in r_edges_out[node]), "flow_left_" +node)
                model.addConstr(sum(edge_flow[e] for e in r_edges_in[node]) == sum(edge_flow[e] for e in l_edges_out[node]), "flow_right_" +node)
                model.addConstr(sum(edge_flow[e] for e in l_edges_in[node]) + sum(edge_flow[e] for e in r_edges_in[node]) == cn[node], "flow_in_" +node)
                model.addConstr(sum(edge_flow[e] for e in r_edges_out[node]) + sum(edge_flow[e] for e in l_edges_out[node]) == cn[node], "flow_out_" +node)

        ### Objective function
        model.setObjective(sum(p_cn[node] for node in nodes) + source_prob * sum(source_left[node] + source_right[node] + sink_left[node] + sink_right[node] for node in double_sides) +
                           sum(cheap_source*source_left[node] + source_prob*source_right[node] + cheap_source*sink_left[node] + source_prob*sink_right[node] for node in free_left_side) +
                           sum(source_prob*source_left[node] + cheap_source*source_right[node] + source_prob*sink_left[node] + cheap_source*sink_right[node] for node in free_right_side) +
                           sum(cheap_source*source_left[node] + cheap_source*source_right[node] + cheap_source*sink_left[node] + cheap_source*sink_right[node] for node in free_both) +
                           sum(edge_flow[edges[e]] * min(0, ctp.edge_cov_pen(edges[e].sup_reads, alpha, edges[e].ovlp, rlen_params, cheap_source)) for e in edges) +
                           flow_penalty, GRB.MAXIMIZE)



        ### Optimize model
        model.update()
        model.write("model_{}-super_{}-cheap_{}.lp".format(outfile, source_prob, cheap_source))
        i_stop = perf_counter()
        print("ILP model generated and saved in {}s".format(i_stop-i_start), file=sys.stderr)

        o_start = perf_counter()
        model.optimize()
        o_stop = perf_counter()
        print("ILP optimization in {}s".format(o_stop-o_start), file=sys.stderr)

        ### Collect results
        all_results = [["Source_prob", source_prob], ["Objective_Value", model.objVal], ["Runtime",model.Runtime] ]
        copy_numbers = {node: [int(var.x), coverages.get(node)] for node, var in cn.items()}

        concordance = {}
        for node, var in cn.items():
            likeliest_CN = likeliest_CNs.get(node)
            if likeliest_CN is not None:
                concordance[node] = [coverages[node], nodes[node].clipped_len(), int(var.x),
                    likeliest_CN, int(var.x) - likeliest_CN]

        for v in model.getVars():
            all_results.append([v.varName, v.x])

        return copy_numbers, all_results, concordance


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError as e:
        print('Encountered an attribute error\n' + str(e))
        return
