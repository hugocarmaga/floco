from scipy.stats import nbinom as nb
from scipy.special import logsumexp
from scipy.stats import skewnorm as sn
import numpy as np
from collections import deque

def ab_to_rp(m, alpha, beta, epsilon):
    mu = alpha * m
    v = max((beta * m) ** 2, mu + 1e-6)

    r = mu ** 2 / (v - mu)
    p = mu / v

    p0 = p / (r * epsilon * (1 - p))

    return r, p, p0


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

    r, p, p0 = ab_to_rp(bin_size, alpha, beta, epsilon)
    probs = deque()
    max_prob = -np.inf

    bin_coverages = np.array(bin_coverages)

    start_cn = int(np.round(full_cov / (full_length * alpha)))

    for c in range(start_cn, start_cn + MAX_EXTENSION):
        if c < 1:
            # Use exponential distribution for CN = 0
            prob = - p0 * bin_coverages.sum() + bin_coverages.size * np.log(1 - np.exp(- p0))
        else:
            # Use negative binomial otherwise
            prob = np.sum(nb.logpmf(bin_coverages, r * c, p))
        if c - start_cn > MIN_EXTENSION and prob + diff_cutoff < max_prob:
            break
        max_prob = max(prob, max_prob)
        probs.append(prob)

    lower_bound = start_cn
    for c in range(start_cn - 1, -1, -1):
        if c < 1:
            # Use exponential distribution for CN = 0
            prob = -p0 * bin_coverages.sum() + bin_coverages.size * np.log(1 - np.exp(-p0))
        else:
            # Use negative binomial otherwise
            prob = np.sum(nb.logpmf(bin_coverages, r * c, p))
        if start_cn - c > MIN_EXTENSION and prob + diff_cutoff < max_prob:
            break
        max_prob = max(prob, max_prob)
        probs.appendleft(prob)
        lower_bound -= 1

    probs = np.array(probs)
    probs -= logsumexp(probs)

    return lower_bound, probs

def edge_cov_pen(d, alpha, ovlp, rlen_params, penalty):
    # Compute p_e0 and p_e1 for each edge, taking their "coverage" as an input
    skn = sn(*rlen_params)
    if d < np.floor(alpha * skn.sf(ovlp) / 4):
        return penalty

    else:
        return 0