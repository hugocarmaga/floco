from sklearn.linear_model import LinearRegression
from collections import defaultdict, Counter
import numpy as np
from statistics import median

def clip_ovlps(nodes,edges):
    '''This function removes all of the overlapping sequencing'''
    edge_list = list(edges.values())
    for edge in edge_list:
        # First, we get the clipping values for the corresponding ends of the nodes
        clip_node1 = nodes[edge.node1].r_clipping if edge.strand1 else nodes[edge.node1].l_clipping
        clip_node2 = nodes[edge.node2].l_clipping if edge.strand2 else nodes[edge.node2].r_clipping

        if clip_node1 < edge.ovlp:
            if edge.strand1:
                nodes[edge.node1].r_clipping = edge.ovlp
            else:
                nodes[edge.node1].l_clipping = edge.ovlp
        
        if clip_node2 < edge.ovlp:
            if edge.strand2:
                nodes[edge.node2].l_clipping = edge.ovlp
            else:
                nodes[edge.node2].r_clipping = edge.ovlp


def bin_nodes100(nodes, bin_size = 100):
    '''Function to select nodes to bin and iniate the bin coverages'''
    nodes_to_bin = list()
    for node in nodes:
        if nodes[node].clipped_len() >= bin_size:
            nodes_to_bin.append(nodes[node].name)
    
    #bin_sizes = [size for size in range(bin_sizes[0], bin_sizes[1]+1, bin_sizes[2])]
    for node in nodes_to_bin:
        # For each node, create a list of tuples with (bin size, array with the size of the nr of bins)
        nodes[node].bins = list()
        nr_bins = nodes[node].clipped_len() // bin_size
        if nr_bins >= 10:
            nodes[node].bins.append((bin_size,np.zeros(nr_bins, dtype=np.uint64)))

    return nodes_to_bin

def filter_100bins(nodes, nodes_to_bin, sel_size):
    '''Funtion to replace the clustering step'''

    bp_cov_per_node = defaultdict()  # Dictionary to save the bins that pass the filtering criteria
    mean_per_node = defaultdict()  # Dictionary to save the average bin coverage for all the binned nodes

    # Iterate over binned_nodes
    for node in nodes_to_bin:  
        for (bin_size, cov_bins) in nodes[node].bins:
            if bin_size == sel_size:
                # Keep only nodes with 10 bins or more and with at least one bin with a coverage bigger than the bin size
                if cov_bins[cov_bins >= sel_size].size:
                    bp_cov_per_node[node] = cov_bins
                    mean_per_node[node] = np.mean(cov_bins)   # Compute the mean bin coverage per node
    
    # Remove top and bottom 3% of nodes, based on the mean bin coverage. Then, remove bins with coverage bigger or equal to 3 times the median bin coverage of the node.
    TOP_PERC = 3
    thresh = np.quantile(np.array(list(mean_per_node.values())), [TOP_PERC/100, (100 - TOP_PERC)/100])
    #binned_nodes = {node: bins[bins < 3 * np.median(bins)] for node,bins in bp_cov_per_node.items() if thresh[0] <= np.mean(bins) <= thresh[1]}
    bins_node = {node: bins for node,bins in bp_cov_per_node.items() if thresh[0] <= np.mean(bins) <= thresh[1]}

    # Merge all kept bins in the same array
    #filtered_bins = np.concatenate(list(binned_nodes.values()))
    
    return bins_node

""" def compute_bins_array(bins_node):
    N_POINTS = 13
    bins_array = defaultdict()
    size = 100
    bins_to_add = list(bins_node.values())
    for _ in range(N_POINTS):
        bins_array[size] = bins_to_add
        size_bins = []
        for node in bins_to_add:
            stop = len(node) if len(node) % 2 == 0 else len(node) - 1
            node_bins = []
            for bin in range(0, stop, 2):
                node_bins.append(node[bin] + node[bin+1])
            if len(node_bins) > 0:
                size_bins.append(node_bins)
        bins_to_add = size_bins
        size *= 2
    
    bins_array = {n: [[val for val in node if val < 3 * median(node)] for node in bins] for n, bins in bins_array.items()}

    return bins_array """

def compute_bins_array(bins_node):
    N_POINTS = 13
    bins_array = [[] for _ in range(N_POINTS)]
    for node_bins in bins_node.values():
        prev_start = len(bins_array[0])
        bins_array[0].extend(node_bins)

        for i in range(1, N_POINTS):
            prev_arr = bins_array[i - 1]
            curr_arr = bins_array[i]
            curr_start = len(curr_arr)

            prev_len = len(prev_arr)
            prev_stop = prev_len - (prev_len - prev_start) % 2
            if prev_start == prev_stop:
                break

            for j in range(prev_start, prev_stop, 2):
                curr_arr.append(prev_arr[j] + prev_arr[j + 1])

            bins_array[i] = curr_arr
            prev_start = curr_start

    return bins_array


def output_bins(bins_array, out):
    with open(out, 'w') as o:
        o.write("#Bin_size,Coverage\n")
        size=100
        for arr in bins_array:
            for bin in arr:
                o.write("{},{}\n".format(size,bin))
            size *= 2

def read_len_distr(read_length, out):
    distr = Counter(read_length)
    with open(out, 'w') as o:
        o.write("#Size,Freq\n")
        for k,v in distr.items():
            o.write("{},{}\n".format(k,v))

def output_edge_supp(edges, out):
    with open(out,'w') as o:
        o.write("#Len,Supp\n")
        for edge in edges:
            o.write("{},{}\n".format(edges[edge].ovlp, edges[edge].sup_reads))


""" def compute_means_and_sds(nodes, nodes_to_bin, bin_sizes):
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
    sd1 = model_sds.coef_[0] """