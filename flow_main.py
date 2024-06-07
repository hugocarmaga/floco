import argparse
from estimation import calculate_covs, read_graph, clip_nodes, bin_nodes, filter_bins, compute_bins_array, alpha_and_beta
from flow_ilp import ilp
import numpy as np
import sys

def parse_arguments():
    parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers(dest="command", required=True)

    # # Create subcommands for the different tasks: getting the node coverages, getting the negative binomial parameters, and solving the ILP
    # cov_parser = subparsers.add_parser("cov", help="Get csv file with number of aligned base pairs (coverage) per node.")
    # cov_parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    # cov_parser.add_argument("-a", "--graphalignment", help="The GAF file.", required=True)
    # cov_parser.add_argument("-c", "--outcov", help="The name for the output csv file with the node coverages", required=True)
    # cov_parser.add_argument("-p", "--ploidy", type=int, default=2, help="Ploidy of the dataset. (default:%(default)s)")
    # cov_parser.add_argument("-b", "--bin_size", nargs=3, default=[5000,50000,5000], metavar=('MIN', 'MAX', 'STEP'), type=int, help="Set the range for the bin size to use for the NB parameters' estimation. (default:%(default)s)")

    # ilp_parser = subparsers.add_parser("flow", help="Solve flow network problem to get CN information per node.")
    # ilp_parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    # ilp_parser.add_argument("-n", "--nodecovs", help="The csv file outputted by 'cov' with the individual node coverage.", required=True)
    # ilp_parser.add_argument("-o", "--outfile", help="The name for the output csv file with the copy number per node", required=True)

    # all_parser = subparsers.add_parser("all", help="Run cov and flow at once.")
    # all_parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    # all_parser.add_argument("-a", "--graphalignment", help="The GAF file.", required=True)
    # all_parser.add_argument("-c", "--outcov", help="The name for the output csv file with the node coverages", required=True)
    # all_parser.add_argument("-p", "--ploidy", type=int, default=2, help="Ploidy of the dataset. (default:%(default)s)")
    # all_parser.add_argument("-o", "--outfile", help="The name for the output csv file with the copy number per node", required=True)
    # all_parser.add_argument("-b", "--bin_size", nargs=3, default=[5000,50000,5000], metavar=('MIN', 'MAX', 'STEP'), type=int, help="Set the range for the bin size to use for the NB parameters' estimation. (default:%(default)s)")

    parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    parser.add_argument("-a", "--graphalignment", help="The GAF file.", required=False)
    parser.add_argument("-c", "--outcov", help="The name for the output csv file with the node coverages", required=True)
    parser.add_argument("-p", "--ploidy", type=int, default=2, help="Ploidy of the dataset. (default:%(default)s)")
    parser.add_argument("-S", "--super_prob", type=int, default=-10, help="Probability for using the super edges when there are other edges available. (default:%(default)s)")
    parser.add_argument("-s", "--cheap_prob", type=int, default=-2, help="Probability for using the super edges when there's no other edge available. (default:%(default)s)")
    parser.add_argument("-e", "--epsilon", type=float, default=0.3, help="Epsilon value for adjusting CN0 counts to probabilities (default:%(default)s)")
    parser.add_argument("-b", "--bin_size", default=100, type=int, help="Set the bin size to use for the NB parameters' estimation. (default:%(default)s)")
    parser.add_argument("-d", "--pickle", type=str, help="Pickle dump with the data.", required=False)

    args = parser.parse_args()

    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

    return args

    # if args.command == "cov":
    #     nodes, edges = read_graph(args.graph)
    #     clip_nodes(nodes, edges)
    #     bin_nodes(nodes, args.bin_size)
    #     node_covs(nodes, args.graphalignment, args.ploidy, args.outcov)
    # elif args.command == "flow":
    #     nodes, edges = read_graph(args.graph)
    #     clip_nodes(nodes, edges)
    #     ilp(args.graph, args.nodecovs, args.outfile, nodes, edges)
    # elif args.command == "all":
    #     nodes, edges = read_graph(args.graph)
    #     clip_nodes(nodes, edges)
    #     node_covs(nodes, args.graphalignment, args.ploidy, args.outcov)
    #     ilp(args.graph, args.nodecovs, args.outfile, nodes, edges)
    # else:
    #     raise NotImplementedError(f"Command {args.command} does not exist.",)

def write_copynums(copy_numbers, out_fname):
    with open(out_fname,"w") as out :
        out.write("Node name,Copy number,Coverage\n")
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
    import pickle
    if args.graphalignment:
        nodes, edges = read_graph(args.graph)
        clip_nodes(nodes, edges)
        nodes_to_bin = bin_nodes(nodes, args.bin_size)
        coverages, rlen_params = calculate_covs(args.graphalignment, nodes, edges)
        bins_node = filter_bins(nodes, nodes_to_bin, args.bin_size)
        bins_array = compute_bins_array(bins_node)
        alpha, beta, params = alpha_and_beta(bins_array, args.bin_size)
        with open("dump-{}.tmp.pkl".format(args.outcov), 'wb') as f:
            pickle.dump((nodes,edges,coverages,alpha,beta,params), f)
    elif args.pickle:
        nodes,edges,coverages,alpha,beta = pickle.load(open(args.pickle, 'rb'))

    copy_numbers, all_results, concordance, nb_per_size = ilp(nodes, edges, coverages, alpha, beta, rlen_params, args.outcov, args.super_prob, args.cheap_prob, args.epsilon)
    print("Writing results to output files!")
    write_copynums(copy_numbers, "copy_numbers-{}-super_{}-cheap_{}.csv".format(args.outcov, args.super_prob, args.cheap_prob))
    write_ilpresults(all_results, "ilp_results-{}-super_{}-cheap_{}.csv".format(args.outcov, args.super_prob, args.cheap_prob))
    write_solutionmetrics(concordance, alpha, beta, nodes, "stats_concordance-{}-super_{}-cheap_{}.csv".format(args.outcov, args.super_prob, args.cheap_prob))
    with open("dump-estimation_debug-{}.tmp.pkl".format(args.outcov), 'wb') as f:
        pickle.dump((nodes,edges,coverages,bins_array,alpha,beta,params,nb_per_size), f)

if __name__ == "__main__":
    main()