import argparse
from estimation import calculate_covs, read_graph, clip_nodes, bin_nodes, nb_parameters
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import nbinom as nb

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    parser.add_argument("-a", "--graphalignment", help="The GAF file.", required=False)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    nodes, edges = read_graph(args.graph)
    clip_nodes(nodes, edges)
    size_list = [10000,50000,100000,200000,500000,1000000]
    for size in size_list:
        nodes_to_bin = bin_nodes(nodes, size)
        coverages = calculate_covs(args.graphalignment, nodes, edges)
        bins = list()
        for node in nodes_to_bin:
            for (bin_size, cov_bins) in nodes[node].bins:
                bins.append(cov_bins)
        bins = np.concatenate(bins)
        r, p = nb_parameters(bins, size)
        print("NB parameters for {} bin size are r: {} and p: {}".format(size,r,p))

        with open("bp_coverage_per_bin-size_{}.txt".format(size), 'w') as o:
            for bin in bins:
                o.write(str(bin)+"\n")

        counts = Counter(bins)
        labels, values = zip(*counts.items())
        indexes = np.arange(len(labels))
        plt.clf()
        plt.title("Coverage frequency for {}bp bins".format(size))
        plt.bar(indexes, values, width=10*size)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax)
        d = nb.logpmf(x, r, p)
        d -= np.max(d)
        d = np.exp(d)
        plt.plot(x, d)
        plt.savefig("distribution_size-{}.png".format(size))
        plt.clf()

if __name__ == "__main__":
    main()
