from scipy.stats import nbinom as nb
from scipy.stats import skewnorm as sn
from scipy.special import logsumexp
import numpy as np
from math import log
import random
import matplotlib.pyplot as plt

def counts_to_probs(r, p, mu, d, n=3, epsilon=0.3):
    # Get an initial interval of copy number values to look at. Use division of observed coverage over mean coverage to get an initial value
    i = max(round(d/mu) - n, 0)
    j = round(d/mu) + n

    # Create list of j+1 -Inf probabilities
    probs_c_given_d = [-np.inf] * (j + 1)
    LOG2 = log(2)  # compute log(2) once to avoid doing it more than once

    # Iterate through the initial interval to compute and save the probabilities for those CN values
    for c in range(i, j+1):
        # Get the prior value (log(1) for anything different than the ploidy, log(2) for the ploidy value)
        #prior = 0 if c != 0 else LOG2
        cr = max(c, epsilon) * r    # Compute CN times r, adjusting for CN=0
        probs_c_given_d[c] = nb.logpmf(d, cr, p) #+ prior

    # Compute probabilities to the left of the interval to include other values within a certain p-value
    lower_bound = 0
    for c in range(i-1, -1, -1):
        cr = max(c, epsilon) * r
        pval = 2 * min(nb.cdf(d, cr, p), nb.sf(d, cr, p))
        #print(pval)
        if pval < 10e-3:  # If p-value is small, stop extending
            #print("Breaking")
            lower_bound = c + 1
            break
        else:   # Add the probability otherwise
            #prior = 0 if c != 0 else LOG2
            probs_c_given_d[c] = nb.logpmf(d, cr, p) #+ prior

    MAX_RIGHT_EXT = 101
    # Extend to the right of the interval to include other values within a certain p-value
    for c in range(j+1, j+MAX_RIGHT_EXT):
        pval = 2 * min(nb.cdf(d, c*r, p), nb.sf(d, c*r, p))
        #print(pval)
        if pval < 10e-3:
            #print("Breaking")
            break
        else:
            #prior = 0 if c != 0 else LOG2
            probs_c_given_d.append(nb.logpmf(d, c*r, p) )#+ prior)

    # Turn list into numpy array to get the sum of all probabilities, and then substract the log of the sum from all the probabilities (see formula for further detail)
    probs_c_given_d = np.array(probs_c_given_d)
    s = logsumexp(probs_c_given_d)
    probs_c_given_d -= s

    return lower_bound, list(probs_c_given_d)

def edge_cov_pen(r, p, d, alpha, ovlp, rlen_params):
    # Compute p_e0 and p_e1 for each edge, taking their "coverage" as an input
    skn = sn(*rlen_params)
    if d < np.floor(alpha * skn.sf(ovlp) / 4):
        p_e0 = nb.logsf(d, r, p)
        p_e1 = nb.logcdf(d, r, p)

        return p_e1-p_e0

    else:
        return 0


def plotting(nodes, coverages, r_bin, p_bin, bin_size):
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






