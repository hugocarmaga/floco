import argparse
from estimation import bin_nodes, filter_bins, alpha_and_beta
from graph_processing import calculate_covs, read_graph, clip_nodes
from flow_ilp import ilp
import numpy as np
import gzip, builtins, sys

def open(filename, mode='r'):
    assert mode == 'r' or mode == 'w'
    if filename is None or filename == '-':
        return sys.stdin if mode == 'r' else sys.stdout
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode + 't')
    else:
        return builtins.open(filename, mode)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    parser.add_argument("-a", "--alignment", help="The GAF file.", required=False)
    parser.add_argument("-o", "--output", help="The name for the output csv file with the node coverages", required=True)
    parser.add_argument("-p", "--bg-ploidy", type=int, default=[1,2], nargs="+", help="Ploidy of the dataset. (default:%(default)s)")
    parser.add_argument("-S", "--expen-pen", type=float, default=-10000, help="Probability for using the super edges when there are other edges available. (default:%(default)s)")
    parser.add_argument("-s", "--cheap-pen", type=float, default=-25, help="Probability for using the super edges when there's no other edge available. (default:%(default)s)")
    parser.add_argument("-e", "--epsilon", type=float, default=0.02, help="Epsilon value for adjusting CN0 counts to probabilities (default:%(default)s)")
    parser.add_argument("-b", "--bin-size", default=100, type=int, help="Set the bin size to use for the NB parameters' estimation. (default:%(default)s)")
    parser.add_argument("-c", "--complexity", type=int, default=2, help="Model complexity (1-3): larger = slower and more accurate. (default: %(default)s)")
    parser.add_argument("-d", "--pickle", type=str, nargs=2, help="Pickle dumps with the data. Parameters dump should be given last. Dump files can be produced with '--debug'.", required=False)
    parser.add_argument("--debug", help="Produce additional files.", required=False)

    args = parser.parse_args()

    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

    return args

def write_copynums(copy_numbers, out_fname):
    with open(out_fname,"w") as out :
        out.write("Node,Length,Sum_coverage,Copy_number\n")
        for k,v in copy_numbers.items():
            out.write("{},{},{}\n".format(k, v[0], v[1]))

def write_ilpresults(all_results, out_fname):
    with open(out_fname,"w") as out :
        out.write("Name,Value\n")
        for parts in all_results:
            out.write(",".join([str(p) for p in parts])+"\n")

def write_solutionmetrics(concordance, alpha, beta, nodes, out_fname):
    discordant_nodes = sum(1 for v in concordance.values() if (len(v)==5 and v[4] != 0))
    covered_nodes = sum(1 for v in concordance.values() if v[0] >= 0)
    discordant_clipped_bp = sum(nodes[node].clipped_len() for node in concordance if concordance[node][0] >= 0 and concordance[node][4] != 0)
    full_length = sum(nodes[node].clipped_len() for node in nodes)
    with open(out_fname,"w") as out :
        out.write("##Total number of nodes with positive length: {}\n".format(covered_nodes))
        out.write("##Number of nodes with discordant copy numbers (%): {}({})\n".format(discordant_nodes, round(discordant_nodes/covered_nodes*100,2)))
        out.write("##Number of (clipped) bp with discordant copy numbers (%): {}({})\n".format(discordant_clipped_bp, round(discordant_clipped_bp/full_length*100,2)))
        out.write("##Coefficient value Alpha: {}\n".format(alpha))
        out.write("##Coefficient value Beta: {}\n".format(beta))
        out.write("#Node,Coverage,Length,Predicted_CN,Likeliest_CN,CN_difference\n")
        for node in concordance:
            out.write("{},{}\n".format(node, ",".join([str(stat) for stat in concordance[node]])))


def main():
    args = parse_arguments()

    assert args.expen_pen <= 0, "Super edge penalty cannot be positive!"
    assert args.cheap_pen <= 0, "Super edge penalty cannot be positive!"
    assert 0 <= args.epsilon <= 1, "Epsilon should be between 0 and 1!"
    assert 1 <= args.complexity <=3, "Complexity should be 1, 2 or 3!"

    import pickle
    if args.graphalignment:
        nodes, edges = read_graph(args.graph)
        clip_nodes(nodes, edges)
        nodes_to_bin = bin_nodes(nodes, args.bin_size)
        coverages, rlen_params = calculate_covs(args.alignment, nodes, edges)
        if args.params:
            alpha, beta = pickle.load(open(args.params, 'rb'))
        else:
            bins_node = filter_bins(nodes, nodes_to_bin, args.bin_size)
            alpha, beta = alpha_and_beta(bins_node, args.bin_size, args.bg_ploidy)
            with open("dump-{}.parameters.tmp.pkl".format(args.output), 'wb') as p:
                pickle.dump((alpha,beta), p)
        with open("dump-{}.tmp.pkl".format(args.output), 'wb') as f:
            pickle.dump((nodes,edges,coverages,rlen_params), f)
    elif args.pickle:
        nodes,edges,coverages,rlen_params = pickle.load(open(args.pickle[0], 'rb'))
        alpha,beta = pickle.load(open(args.pickle[1], 'rb'))

    copy_numbers, all_results, concordance = ilp(nodes, edges, coverages, alpha, beta, rlen_params, args.output, args.expen_pen, args.cheap_pen, args.epsilon, args.complexity)
    print("Writing results to output files!")
    write_copynums(copy_numbers, "copy_numbers-{}-super_{}-cheap_{}.csv".format(args.output, args.expen_pen, args.cheap_pen))
    write_ilpresults(all_results, "ilp_results-{}-super_{}-cheap_{}.csv".format(args.output, args.expen_pen, args.cheap_pen))
    write_solutionmetrics(concordance, alpha, beta, nodes, "stats_concordance-{}-super_{}-cheap_{}.csv".format(args.output, args.expen_pen, args.cheap_pen))

if __name__ == "__main__":
    main()