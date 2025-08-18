from scipy.stats import nbinom as nb
from scipy.stats import skewnorm as sn
from scipy.special import logsumexp
import numpy as np
from math import log
import random
from collections import deque
import itertools

def ab_to_rp(m, alpha, beta):
    mu = alpha * m
    v = max((beta * m) ** 2, mu + 1e-6)

    r = mu ** 2 / (v - mu)
    p = mu / v
    return r, p


def cn_probs(alpha, beta, epsilon,
        full_length, full_cov,
        bin_size, bin_coverages, diff_cutoff):
    """
    Calculates a range of CN probabilities based on parameters α, β & ε.
    Full length and full coverage are used to identify starting point.
    Bin size and bin coverages are used for actual log-probability calculation,
        it is possible that bin size = full length and bin coverages = [full coverage].
    Values are extended to the left and right until difference in log-probabilities do not exceed diff_cutoff.
    At least one step in both directions will be made.
    """
    MIN_EXTENSION = 1
    MAX_EXTENSION = 101

    r, p = ab_to_rp(bin_size, alpha, beta)
    probs = deque()
    max_prob = -np.inf

    pres = False
    if full_cov == 43825868 and full_length == 14178334:
        pres = True

    start_cn = int(np.round(full_cov / (full_length * alpha)))
    if pres: print("Starting CN is: {}".format(start_cn))
    for c in range(start_cn, start_cn + MAX_EXTENSION):
        c = max(c, epsilon)
        prob = np.sum(nb.logpmf(bin_coverages, r * c, p))
        if c - start_cn > MIN_EXTENSION and prob + diff_cutoff < max_prob:
            break
        max_prob = max(prob, max_prob)
        probs.append(prob)
        if pres:
            print("For c: {}".format(c))
            print("Max prob: {}".format(max_prob))
            print("Probs: {}".format(probs))

    lower_bound = start_cn
    for c in range(start_cn - 1, -1, -1):
        c = max(c, epsilon)
        prob = np.sum(nb.logpmf(bin_coverages, r * c, p))
        if start_cn - c > MIN_EXTENSION and prob + diff_cutoff < max_prob:
            break
        max_prob = max(prob, max_prob)
        probs.appendleft(prob)
        lower_bound -= 1
        if pres:
            print("For c: {}".format(c))
            print("Max prob: {}".format(max_prob))
            print("Probs: {}".format(probs))
            print("Lower bound: {}".format(lower_bound))

    probs = np.array(probs)
    probs -= logsumexp(probs)
    if pres: print("Final probs: {}".format(probs))
    return lower_bound, probs


# def counts_to_probs(i, j, r, p, d, epsilon=0.3):
#     # array of adjusted copy numbers: from i to j, but 0 is replaced with epsilon.
#     adj_c = np.maximum(np.arange(i, j), epsilon)
#     # array of r for each CN
#     rs = r * adj_c
#     # returns array of not normalized log-probabilities
#     return nb.logpmf(d, rs, p)

#     # # Create list of j+1 -Inf probabilities
#     # probs_c_given_d = [-np.inf] * j

#     # # Iterate through the initial interval to compute and save the probabilities for those CN values
#     # for c in range(i, j):
#     #     cr = max(c, epsilon) * r    # Compute CN times r, adjusting for CN=0
#     #     probs_c_given_d[c] = nb.logpmf(d, cr, p)

#     # # Turn list into numpy array to get the sum of all probabilities, and then substract the log of the sum from all the probabilities (see formula for further detail)
#     # probs_c_given_d = np.array(probs_c_given_d[i:])
#     # #s = logsumexp(probs_c_given_d)
#     # #probs_c_given_d -= s

#     # return probs_c_given_d


# def get_bounds(r, p, mu, d, n=3, epsilon=0.3):
#     # Get an initial interval of copy number values to look at. Use division of observed coverage over mean coverage to get an initial value
#     i = max(round(d/mu) - n, 0)
#     j = round(d/mu) + n

#     lower_bound = 0
#     for c in range(i-1, -1, -1):
#         cr = max(c, epsilon) * r
#         pval = 2 * min(nb.cdf(d, cr, p), nb.sf(d, cr, p))
#         if pval < 10e-3:  # If p-value is small, stop extending
#             lower_bound = c + 1
#             break

#     MAX_RIGHT_EXT = 101
#     # Extend to the right of the interval to include other values within a certain p-value
#     for c in range(j+1, j+MAX_RIGHT_EXT):
#         pval = 2 * min(nb.cdf(d, c*r, p), nb.sf(d, c*r, p))
#         if pval < 10e-3:
#             upper_bound = c
#             break

#     return lower_bound, upper_bound


def edge_cov_pen(d, alpha, ovlp, rlen_params, penalty):
    # Compute p_e0 and p_e1 for each edge, taking their "coverage" as an input
    skn = sn(*rlen_params)
    if d < np.floor(alpha * skn.sf(ovlp) / 4):
        return penalty

    else:
        return 0


def plotting(nodes, coverages, r_bin, p_bin, bin_size):
    import matplotlib.pyplot as plt
    stats = list()
    for node in coverages:
        n = bin_size
        m = nodes[node].clipped_len()
        r = (r_bin*(1-p_bin))/(1-n/m*p_bin)
        p = n/m * p_bin
        mu = r * (1-p) / p
        _, probs = counts_to_probs(r, p, mu, coverages[node])
        stats.append([m, round(mu, 1), coverages[node], probs])

    subset = random.sample(stats, 5)
    for stat in subset:
        plt.clf()
        plt.title("Node, with size {}, mean {} and coverage {}.".format(stat[0], stat[1], stat[2]))
        plt.bar(range(stat[3].size), stat[3], tick_label = range(stat[3].size))
        plt.savefig("plots/probabilities/Node-len_{}_coverage_{}-chm13-chr22.png".format(stat[0], stat[1]))
