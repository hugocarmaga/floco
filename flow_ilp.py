import gurobipy as gp
from gurobipy import GRB
from math import log
from collections import defaultdict
import counts_to_probabs as ctp
from time import perf_counter
import sys

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

        #Filter out nodes with no coverage
        covered_nodes = {node: nodes[node] for node in nodes if coverages.get(node)}

        #Get different sets of nodes, with edges on both sides or just on the left/right side
        double_sides = defaultdict()
        free_left_side = defaultdict()
        free_right_side = defaultdict()
        free_both = defaultdict()

        # Super-edges flow on the two sides of the node (I don't care if it's for the supersource or the supersink, only the node side matters)
        source_left = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "source_left_"+node) for node in covered_nodes}
        source_right = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "source_right_"+node) for node in covered_nodes}
        sink_left = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "sink_left_"+node) for node in covered_nodes}
        sink_right = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "sink_right_"+node) for node in covered_nodes}

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
        concordance = defaultdict(list)

        # Iterate over all nodes to define the constraints
        for node in nodes:
            cov = coverages.get(node)
            if cov:
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


                # PWL constraints for node CN probabilities
                m = nodes[node].clipped_len()

                # Compute mean and variance for the node, using its length
                mu = alpha * m
                v = max((beta * m) ** 2, mu + 1e-6)

                r = mu ** 2 / (v - mu)
                p = mu / v
                lower_bound, y = ctp.counts_to_probs(r, p, mu, cov, 3, epsilon)
                upper_bound = len(y)
                x = list(range(lower_bound, upper_bound))
                y = y[lower_bound:]
                assert len(x)==len(y), "{} is not the same length as {}".format(x,y)

                concordance[node] = [x[y.index(max(y))]]

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
                concordance[node] = [-1]

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

        for node, var in cn.items():
            if concordance[node][0] >= 0:
                concordance[node] = [coverages[node], nodes[node].clipped_len(), int(var.x), concordance[node][0], int(var.x) - concordance[node][0]]

        for v in model.getVars():
            all_results.append([v.varName, v.x])

        return copy_numbers, all_results, concordance


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError as e:
        print('Encountered an attribute error\n' + str(e))
        return
