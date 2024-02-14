import gurobipy as gp
from gurobipy import GRB
from math import log
from collections import defaultdict
import counts_to_probabs as ctp

def ilp(nodes, edges, coverages, r_bin, p_bin, bin_size, outfile, source_prob = -80, cheap_source = -2, epsilon = 0.3, WEIGHT = 1):
    '''Function to formulate and solve the ILP for the flow network problem.'''
    try:
        # Create a new model
        model = gp.Model("CN flow ILP")

        ### Define variables

        # Copy number (graph flow) variable
        cn = {} # node: model.addVar(vtype = GRB.INTEGER, lb = 0, name = "cn_"+node)  for node in nodes}

        # Probability of cn as piecewise linear function
        p_cn = {node: model.addVar(vtype = GRB.CONTINUOUS, lb = - GRB.INFINITY, ub = 0, name = "p_cn_"+node) for node in nodes}

        # Edge flow on the two sides of the node - or just one variable for all edges
        #left_edges = {edge: model.addVar(vtype=GRB.INTEGER, lb = 0, name="le_{}_{}".format(edge.node1, edge.node2)) for edge in edges}
        #right_edges = {edge: model.addVar(vtype=GRB.INTEGER, lb = 0, name="re_{}_{}".format(edge.node1, edge.node2)) for edge in edges}
        edge_flow = {edges[edge]: model.addVar(vtype=GRB.INTEGER, lb = 0, name = edge) for edge in edges}

        #Filter out nodes with no coverage
        covered_nodes = {node: nodes[node] for node in nodes if coverages.get(node)}

        #Get different sets of nodes, with edges on both sides or just on the left/right side
        double_sides = defaultdict()
        free_left_side = defaultdict()
        free_right_side = defaultdict()

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
        for e in edges:
            '''Iterate over the edges and add them to the correct side of the respective nodes'''
            r_edges_out[edges[e].node1].append(edges[e]) if edges[e].strand1 else l_edges_out[edges[e].node1].append(edges[e])
            l_edges_in[edges[e].node2].append(edges[e]) if edges[e].strand2 else r_edges_in[edges[e].node2].append(edges[e])
        
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
                    else:
                        free_left_side[node] = node

                # PWL constraints for node CN probabilities
                n = bin_size
                m = nodes[node].clipped_len()
                
                # Check if r and p values are always within boundaries
                if m > n * p_bin:
                    r = (r_bin*(1-p_bin))/(1-n/m*p_bin)
                    p = n/m * p_bin
                    mu = r * (1-p) / p
                    lower_bound, y = ctp.counts_to_probs(r, p, mu, cov, 3, epsilon)
                    upper_bound = len(y)
                    x = list(range(lower_bound, upper_bound))
                    y = y[lower_bound:]
                    assert len(x)==len(y), "{} is not the same length as {}".format(x,y)
                
                # If not, use poisson distribution instead:
                else:
                    lamb =  m / n * r_bin * (1 - p_bin) / p_bin
                    p = 10e-5
                    r = p / (1-p) * lamb
                    lower_bound, y = ctp.counts_to_probs(r, p, lamb, cov, 3, epsilon)
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
            
            else:
                concordance[node] = [-1]

                cn[node] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "cn_"+node)
                model.addConstr(sum(edge_flow[e] for e in l_edges_in[node]) == sum(edge_flow[e] for e in r_edges_out[node]), "flow_left_" +node)
                model.addConstr(sum(edge_flow[e] for e in r_edges_in[node]) == sum(edge_flow[e] for e in l_edges_out[node]), "flow_right_" +node)
                model.addConstr(sum(edge_flow[e] for e in l_edges_in[node]) + sum(edge_flow[e] for e in r_edges_in[node]) == cn[node], "flow_in_" +node)
                model.addConstr(sum(edge_flow[e] for e in r_edges_out[node]) + sum(edge_flow[e] for e in l_edges_out[node]) == cn[node], "flow_out_" +node)           

        r_edge = (r_bin*(1-p_bin))/(1-bin_size*p_bin)
        p_edge = bin_size * p_bin
        ### Objective function
        model.setObjective(sum(p_cn[node] for node in nodes) + source_prob * sum(source_left[node] + source_right[node] + sink_left[node] + sink_right[node] for node in double_sides) +
                           sum(cheap_source*source_left[node] + source_prob*source_right[node] + cheap_source*sink_left[node] + source_prob*sink_right[node] for node in free_left_side) +
                           sum(source_prob*source_left[node] + cheap_source*source_right[node] + source_prob*sink_left[node] + cheap_source*sink_right[node] for node in free_right_side) +
                           WEIGHT * sum(edge_flow[edges[e]] * ctp.edge_cov_pen(r_edge, p_edge, edges[e].sup_reads) for e in edges), GRB.MAXIMIZE)

        ### Optimize model
        print("Optimizing now!")
        model.update()
        model.write("model_{}-super_{}-cheap_{}.lp".format(outfile, source_prob, cheap_source))
        model.optimize()

        print("Optimization finished!")

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
