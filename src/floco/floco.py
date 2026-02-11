import argparse
from . import estimation
from .estimation import filter_bins, alpha_and_beta
from .graph_processing import calculate_covs, read_graph, clip_nodes, bin_nodes
from .flow_ilp import ilp
import numpy as np
import gzip, builtins, sys
import importlib.metadata

def open(filename, mode='r'):
    assert mode == 'r' or mode == 'w'
    if filename is None or filename == '-':
        return sys.stdin if mode == 'r' else sys.stdout
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode + 't')
    else:
        return builtins.open(filename, mode)

def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-g", "--graph", help="The GFA file with the graph.", required=True)
    parser.add_argument("-a", "--alignment", help="The GAF file with sequence-to-graph alignments.", required=False)
    parser.add_argument("-o", "--output", help="The name for the output csv file with the copy numbers.", required=True)
    parser.add_argument("-p", "--bg-ploidy", type=int, default=[1,2], nargs="+", help="Expected most common CN value in the graph (background ploidy of the dataset). (default:%(default)s)")
    parser.add_argument("-S", "--expen-pen", type=float, default=-10000, help="Probability for using the super edges when there are other edges available. (default:%(default)s)")
    parser.add_argument("-s", "--cheap-pen", type=float, default=-25, help="Probability for using the super edges when there is no other edge available. (default:%(default)s)")
    parser.add_argument("-e", "--epsilon", type=float, default=0.02, help="Epsilon value for adjusting CN0 counts to probabilities (default:%(default)s)")
    parser.add_argument("-b", "--bin-size", default=100, type=int, help="Set the bin size to use for the NB parameters estimation. (default:%(default)s)")
    parser.add_argument("-c", "--complexity", type=int, default=2, help="Model complexity (1-3): larger = slower and more accurate. (default: %(default)s)")
    parser.add_argument("-d", "--pickle", type=str, help="Pickle dump with the data. Dump file can be produced with '--debug'.", required=False)
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of computing threads to use by the ILP solver.", required=False)
    parser.add_argument("--debug", action='store_true' ,help="Produce additional files.", required=False)
    parser.add_argument('-h', '--help', action='help',
        help='Show this help message and exit.')
    parser.add_argument('-V', '--version', action='version', help="Show program's version number and exit.",
        version='floco {version}'.format(version=importlib.metadata.version('floco')))

    args = parser.parse_args()

    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)

    return args

def write_copynums(copy_numbers, out_fname):
    with open(out_fname,"w") as out :
        out.write("Node,Length,Sum_coverage,Copy_number\n")
        for k,v in copy_numbers.items():
            out.write("{},{}\n".format(k, ",".join([str(stat) for stat in v])))

def write_ilpresults(all_results, out_fname):
    with open(out_fname,"w") as out :
        out.write("Name,Value\n")
        for parts in all_results:
            out.write(",".join([str(p) for p in parts])+"\n")

def write_solutionmetrics(concordance, out_fname):
    with open(out_fname,"w") as out :
        out.write("#Node,Coverage,Length,Predicted_CN,Likeliest_CN\n")
        for node in concordance:
            out.write("{},{}\n".format(node, ",".join([str(stat) for stat in concordance[node]])))


def main():
    args = parse_arguments()

    try:
        import gurobipy as gp
    except ImportError:
        sys.stderr.write('Please install Gurobi (see README)\n')
        exit(1)

    assert args.expen_pen <= 0, "Super edge penalty cannot be positive!"
    assert args.cheap_pen <= 0, "Super edge penalty cannot be positive!"
    assert 0 <= args.epsilon <= 1, "Epsilon should be between 0 and 1!"
    assert 1 <= args.complexity <=3, "Complexity should be 1, 2 or 3!"

    import pickle
    if args.alignment:
        nodes, edges = read_graph(args.graph)
        clip_nodes(nodes, edges)
        nodes_to_bin = bin_nodes(nodes, args.bin_size)
        coverages, rlen_params = calculate_covs(args.alignment, nodes, edges)
        bins_node = filter_bins(nodes, nodes_to_bin, args.bin_size)
        alpha, beta = alpha_and_beta(bins_node, args.bin_size, args.bg_ploidy)
        if args.debug:
            with builtins.open("dump-{}.tmp.pkl".format(args.output), 'wb') as f:
                pickle.dump((nodes,edges,coverages,rlen_params,alpha,beta), f)
    elif args.pickle:
        nodes,edges,coverages,rlen_params,alpha,beta = pickle.load(builtins.open(args.pickle, 'rb'))

    copy_numbers, all_results, concordance = ilp(nodes, edges, coverages, alpha, beta, rlen_params, args.output, args.expen_pen, args.cheap_pen, args.epsilon, args.complexity, args.debug, args.threads)
    print("*** Writing results to output files!", file=sys.stderr)
    write_copynums(copy_numbers, args.output)
    if args.debug:
        write_ilpresults(all_results, "ilp_results-{}.csv".format(args.output.split(".csv")[0]))
        write_solutionmetrics(concordance, "stats_concordance-{}.csv".format(args.output.split(".csv")[0]))

if __name__ == "__main__":
    main()