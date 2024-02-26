from sklearn.linear_model import LinearRegression
from collections import defaultdict
import numpy as np

def filter_bins(nodes, nodes_to_bin, sel_size):
    '''Funtion to replace the clustering step'''

    bp_cov_per_node = defaultdict()  # Dictionary to save the bins that pass the filtering criteria
    mean_per_node = defaultdict()  # Dictionary to save the average bin coverage for all the binned nodes

    # Iterate over binned_nodes
    for node in nodes_to_bin:  
        for (bin_size, cov_bins) in nodes[node].bins:
            if bin_size == sel_size:
                # Keep only nodes with 10 bins or more and with at least one bin with a coverage bigger than the bin size
                if cov_bins[cov_bins >= sel_size].size > 0 and cov_bins.size >= 10:
                    bp_cov_per_node[node] = cov_bins
                    mean_per_node[node] = np.mean(cov_bins)   # Compute the mean bin coverage per node
    
    # Remove top and bottom 3% of nodes, based on the mean bin coverage. Then, remove bins with coverage bigger or equal to 3 times the median bin coverage of the node.
    TOP_PERC = 3
    thresh = np.quantile(np.array(list(mean_per_node.values())), [TOP_PERC/100, (100 - TOP_PERC)/100])
    binned_nodes = {node: bins[bins < 3 * np.median(bins)] for node,bins in bp_cov_per_node.items() if thresh[0] <= np.mean(bins) <= thresh[1]}

    # Merge all kept bins in the same array
    filtered_bins = np.concatenate(list(binned_nodes.values()))
    
    return filtered_bins

def compute_means_and_sds(nodes, nodes_to_bin, bin_sizes):
    '''Function to get mean and standard deviation for all bins of each bin size.'''
    filtered_per_size = defaultdict()
    mean_size = defaultdict()
    sd_size = defaultdict()
    for size in bin_sizes:
        filtered_bins = filter_bins(nodes, nodes_to_bin, size)
        filtered_per_size[size] = filtered_bins
        mean_size[size] = np.mean(filtered_bins)
        sd_size[size] = np.std(filtered_bins)

    model_means = LinearRegression().fit(bin_sizes, mean_size.values())
    m1 = model_means.coef_[0]
    model_sds = LinearRegression().fit(bin_sizes, sd_size.values())
    sd1 = model_sds.coef_[0]