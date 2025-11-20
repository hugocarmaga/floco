from collections import defaultdict
import warnings
import numpy as np
from time import perf_counter
from scipy import stats, optimize, special
from math import sqrt
import sys

def bin_nodes(nodes, bin_size = 100):
    '''Function to select nodes to bin and iniate the bin coverages'''
    nodes_to_bin = list()
    b_start = perf_counter()
    for node in nodes:
        size = nodes[node].clipped_len()
        if size >= bin_size:
            nodes_to_bin.append(nodes[node].name)
            # For each node, create a list of tuples with (bin size, array with the size of the nr of bins)
            nodes[node].bins = list()
            nr_bins = size // bin_size
            nodes[node].bins.append((bin_size, np.zeros(nr_bins, dtype=np.uint64)))
        elif size > 0:
            nodes[node].bins = list()
            nodes[node].bins.append((size, np.zeros(1, dtype=np.uint64)))

    b_stop = perf_counter()
    print("Nodes binned in {}s".format(b_stop-b_start), file=sys.stderr)

    return nodes_to_bin

def filter_bins(nodes, nodes_to_bin, sel_size = 100):
    '''Funtion to filter bins by quality'''
    f_start = perf_counter()
    bp_cov_per_node = defaultdict()  # Dictionary to save the bins that pass the filtering criteria
    mean_per_node = defaultdict()  # Dictionary to save the average bin coverage for all the binned nodes

    # Iterate over binned_nodes
    counter=0
    for node in nodes_to_bin:
        for (bin_size, cov_bins) in nodes[node].bins:
            if bin_size == sel_size:
                # Keep only nodes with at least one bin with a coverage bigger than the bin size
                if np.any(cov_bins >= sel_size):
                    counter+=1
                    bp_cov_per_node[node] = cov_bins
                    mean_per_node[node] = np.mean(cov_bins)   # Compute the mean bin coverage per node

    # Remove top and bottom 3% of nodes, based on the mean bin coverage. Then, remove bins with coverage bigger or equal to 3 times the median bin coverage of the node.
    TOP_PERC = 3
    thresh = np.quantile(np.array(list(mean_per_node.values())), [TOP_PERC/100, (100 - TOP_PERC)/100])
    bins_node = {node: bins for node,bins in bp_cov_per_node.items() if thresh[0] <= mean_per_node[node] <= thresh[1]}

    f_stop = perf_counter()
    print("Bins filtered in {}s".format(f_stop-f_start), file=sys.stderr)

    return bins_node


def estimate_mean_std_at_ploidy(counts, bp_step, input_ploidy):
    '''Function to estimate mean and standard deviation per bin size.'''
    EPSILON = 0.01
    N_CN = max(4, input_ploidy + 3)
    print("Estimating parameters", file=sys.stderr)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        def MLE_NBinom(parameters):
            # Changing parameters
            m, v = parameters[:2]
            v = max(m * 1.00001, v)
            r = m**2 / (v - m)
            p = m / v

            cs = np.log(parameters[2:] / np.sum(parameters[2:]))
            rs = np.maximum(np.arange(N_CN), EPSILON) * r

            sum_dist = [count * special.logsumexp(cs + stats.nbinom.logpmf(round((i + 0.5)*bp_step), rs, p))
                for i, count in enumerate(counts)]
            return min(1e30, -np.sum(sum_dist))

        mode = float((np.argmax(counts) + 0.5) * bp_step)

        # Test case: CN = 1 for most of the nodes.
        m0 = mode / input_ploidy
        v0 = m0 * m0 / 25
        x0 = [m0, v0] + [0.1] * N_CN
        x0[2 + input_ploidy] = 0.9
        bounds = [(m0 / 1.5, m0 * 1.5), (v0 / 20, v0 * 20)] + [(0.0, 0.45)] * N_CN
        bounds[2 + input_ploidy] = (0.5, 1.0)

        sol = optimize.minimize(MLE_NBinom, x0, bounds=bounds, method='Nelder-Mead')
        LL = sol.fun
        m, v = sol.x[:2]
        cs = sol.x[2:]
        cs /= np.sum(cs)
        print(f"Ploidy {input_ploidy}:  L = -{LL:.0f}, m = {m:.2f}, sd = {sqrt(v):.2f}, CN fractions: {cs}", file=sys.stderr)

        return LL, m, sqrt(v)


def alpha_and_beta(bins_node, bin_size = 100, ploidies = [1,2]):
    p_start = perf_counter()

    bins = []
    for curr_bins in bins_node.values():
        bins.extend(curr_bins)
    bins = np.array(bins)

    TOP_PERC = 3
    ROUND_BINS = 5
    thresh = np.quantile(bins, [TOP_PERC/100, (100 - TOP_PERC)/100])
    bins = bins[(thresh[0] <= bins) & (bins <= thresh[1])]
    bp_step = bin_size // ROUND_BINS
    counts = np.bincount(np.round(bins / bp_step).astype(dtype=np.int64),
        minlength=int(round(thresh[1])) // bp_step + 1)
    best_ll = -np.inf
    best_a = -np.inf
    best_b = -np.inf
    for ploidy in ploidies:
        ll, a, b = estimate_mean_std_at_ploidy(counts, bp_step, ploidy)
        if ll > best_ll:
            best_ll, best_a, best_b = ll, a, b
    best_a /= bin_size
    best_b /= bin_size

    print("Alpha: {}, Beta: {}".format(best_a, best_b), file=sys.stderr)
    p_stop = perf_counter()
    print("Alpha and beta estimated in {}s".format(p_stop-p_start), file=sys.stderr)
    return best_a, best_b

