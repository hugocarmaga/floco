import gurobipy as gp
from gurobipy import GRB
from math import log
from collections import defaultdict
import counts_to_probabs as ctp

def ilp(nodes, edges, coverages, r_bin, p_bin, bin_size, outfile, source_prob = -80, ploidy = 2):
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
        edge_flow = {edge: model.addVar(vtype=GRB.INTEGER, lb = 0, name="e_{}_{}".format(edge.node1, edge.node2)) for edge in edges}

        #Filter out nodes with no coverage
        covered_nodes = {node: nodes[node] for node in nodes if coverages.get(node)}

        # Super-edges flow on the two sides of the node (I don't care if it's for the supersource or the supersink, only the node side matters)
        left_super = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "l_super_"+node) for node in covered_nodes}
        right_super = {node: model.addVar(vtype=GRB.INTEGER, lb = 0,  name = "r_super_"+node) for node in covered_nodes}

        model.update()
        ### Objective function
        model.setObjective(sum(p_cn[node] for node in nodes) + source_prob*sum(left_super[node] + right_super[node] for node in covered_nodes), GRB.MAXIMIZE)

        ### Constraints

        # Some edge preprocessing to be able to get all edges per side for each node
        l_edges = defaultdict(list)
        r_edges = defaultdict(list)
        for e in edges:
            '''Iterate over the edges and add them to the correct side of the respective nodes'''
            r_edges[e.node1].append(e) if e.strand1 else l_edges[e.node1].append(e)
            l_edges[e.node2].append(e) if e.strand2 else r_edges[e.node2].append(e)

        # Iterate over all nodes to define the constraints
        for node in nodes:
            cov = covered_nodes.get(node)
            if cov:
                # PWL constraints for node CN probabilities
                n = bin_size
                m = nodes[node].clipped_len()
                
                # Check if r and p values are always within boundaries
                if m > n * p_bin:
                    r = (r_bin*(1-p_bin))/(1-n/m*p_bin)
                    p = n/m * p_bin
                    mu = r * (1-p) / p
                    lower_bound, y = ctp.counts_to_probs(r, p, mu, cov, 3, ploidy)
                    upper_bound = len(y)
                    x = list(range(lower_bound, upper_bound))
                    y = y[lower_bound:]
                    assert len(x)==len(y), "{} is not the same length as {}".format(x,y)
                
                # If not, use poisson distribution instead:
                else:
                    lamb =  m / n * r_bin * (1 - p_bin) / p_bin
                    p = 10e-5
                    r = p / (1-p) * lamb
                    lower_bound, y = ctp.counts_to_probs(r, p, lamb, cov, 3, ploidy)
                    upper_bound = len(y)
                    x = list(range(lower_bound, upper_bound))
                    y = y[lower_bound:]
                    assert len(x)==len(y), "{} is not the same length as {}".format(x,y)
                     

                cn[node] = model.addVar(vtype = GRB.INTEGER, lb = lower_bound, ub = upper_bound - 1,  name = "cn_"+node)
                model.addGenConstrPWL(cn[node], p_cn[node], x, y, "PWLConstr_"+node)

                # Flow conservation constraint
                model.addConstr(left_super[node] + sum(edge_flow[e] for e in l_edges[node]) == cn[node], "flow_left_" +node)
                model.addConstr(right_super[node] + sum(edge_flow[e] for e in r_edges[node]) == cn[node], "flow_right_" +node)
            
            else:
                cn[node] = model.addVar(vtype = GRB.INTEGER, lb = 0, name = "cn_"+node)
                model.addConstr(sum(edge_flow[e] for e in l_edges[node]) == cn[node], "flow_left_" +node)
                model.addConstr(sum(edge_flow[e] for e in r_edges[node]) == cn[node], "flow_right_" +node)

        ### Optimize model
        print("Optimizing now!")
        model.update()
        model.write("model_{}-super_{}.lp".format(outfile, source_prob))
        model.optimize()

        print("Optimization finished!")

        ### Collect results
        all_results = [["Source_prob", source_prob], ["Objective_Value", model.objVal], ["Runtime",model.Runtime] ]
        copy_numbers = {node: [int(var.x), coverages.get(node)] for node, var in cn.items()}

        for v in model.getVars():
            all_results.append([v.varName, v.x])
        
        return copy_numbers, all_results
        

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError as e:
        print('Encountered an attribute error\n' + str(e))
        return
