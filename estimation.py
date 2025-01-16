from collections import defaultdict
from dataclasses import dataclass
import re
import numpy as np
from time import perf_counter
import sys
from scipy import stats, optimize, special
from math import sqrt

@dataclass
class Edge:
    node1: str
    node2: str
    strand1: bool
    strand2: bool
    ovlp: int
    sup_reads: int = 0

    def __lt__(self,other):
        return self.ovlp < other.ovlp

    def to_tuple(self):
        return (self.node1, self.node2, self.strand1, self.strand2)

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def echo_reverse(self):
        if self.strand1 == self.strand2:
            return "e_{}_{}_{}_{}".format(self.node2, "-" if self.strand2 else "+", self.node1, "-" if self.strand1 else "+")
        else:
            return "e_{}_{}_{}_{}".format(self.node2, "+" if self.strand1 else "-", self.node1, "+" if self.strand2 else "-")

@dataclass(frozen=False)
class Node:
    name: str
    seq_len: int
    l_edges: int = 0
    r_edges: int = 0
    l_clipping: int = 0
    r_clipping: int = 0
    bins: list = None

    def __lt__(self,other):
        return self.clipped_len() < other.clipped_len()

    def clipped_len(self):
        return self.seq_len - self.l_clipping - self.r_clipping

    def right_end(self):
        return self.seq_len - self.r_clipping

    def single_node(self, start, end, strand=True):
        if strand:
            return min(self.right_end(), end) - max(self.l_clipping, start)
        else:
            return min(self.right_end(), self.seq_len - start) - max(self.l_clipping, self.seq_len - end)

    def first_node(self, pos):
        return max(self.right_end(), pos) - max(self.l_clipping, pos)

    def last_node(self, pos):
        return min(self.right_end(), pos) - min(self.l_clipping, pos)

    def middle_point(self):
        return self.clipped_len() // 2 + self.l_clipping

def write_results(nodes, coverages, read_depth, means_cov, vars_cov, means_rd, vars_rd, out_fname):
    with open("per_bin_mean_and_var_"+out_fname,"w") as out :
        out.write("Bin size,Mean_BP,Variance_BP,Mean_RD,Variance_RD\n")
        for size in means_cov:
            out.write(str(size) + "," + str(means_cov[size][0]) + "," + str(vars_cov[size]) + "," + str(means_rd[size][0]) + "," + str(vars_rd[size]) + "\n")

    with open(out_fname, "w") as out :
        out.write("#Node name,Clipped_len,BP_cov,RD_cov\n")
        for node in read_depth:
            out.write(node + "," + str(nodes[node].clipped_len()) + "," + str(coverages[node]) + "," + str(read_depth[node]) + "\n")
        out.write("#Node name,Bin_start,Bin_end,BP_cov,RD_cov\n")
        for node in nodes:
            if nodes[node].bins != None:
                for (bin_size, cov_bins, rd_bins) in nodes[node].bins:
                    start = nodes[node].l_clipping
                    end = start + bin_size
                    for i in range(cov_bins.size):
                        out.write(node + "," + str(start) + "," + str(end) + "," + str(cov_bins[i]) + "," + str(rd_bins[i]) + "\n")
                        start += bin_size
                        end += bin_size


def read_graph(graph_fname):
    '''This function takes a GFA file as an input, and returns a list of Edge objects, and a dictionary of {'node_name': Node object}'''

    edges = defaultdict()
    nodes = defaultdict()
    g_start = perf_counter()
    with open(graph_fname,"r") as graph_file:
        for line in graph_file:
            columns = line.split()

            #sequence line
            if columns[0] == "S":
                nodes[columns[1]] = Node(columns[1], len(columns[2]))  #Add Node object as a value to the dictionary of nodes, using the node name as key

            # link line
            if columns[0] == "L":
                edges["e_{}_{}_{}_{}".format(columns[1],columns[2],columns[3],columns[4])] = Edge(columns[1], columns[3], columns[2]=="+", columns[4]=="+", int(columns[5].strip("M")))  #Add Edge object to the dictionary of edges

                # Add number of edges connected to each side of each node
                if columns[2] == "+":
                    nodes[columns[1]].r_edges += 1
                elif columns[2] == "-":
                    nodes[columns[1]].l_edges += 1

                if columns[4] == "+":
                    nodes[columns[3]].l_edges += 1
                elif columns[4] == "-":
                    nodes[columns[3]].r_edges += 1
    g_stop = perf_counter()
    print("Graph read in {}s".format(g_stop-g_start), file=sys.stderr)

    return nodes, edges

def clip_nodes(nodes,edges):
    '''This function iterates over the list of edges to decide where to clip the nodes, in order to get rid of the overlaps, aiming to get the maximum amount of non-duplicated information. It takes as an input the list of edges and
    dictionary of nodes produced by read_graph(), outputting a new dictionary of nodes with the updated clipping information.'''
    c_start = perf_counter()
    # Sort edges by decreasing overlap size
    edge_list = list(edges.values())
    edge_list.sort(reverse=True)
    # Iterate over list of edges, and clip the nodes to get rid of overlaps.
    for edge in edge_list:
        node1 = nodes[edge.node1]
        node2 = nodes[edge.node2]
        # First, we get the clipping values for the corresponding ends of the nodes
        node1_clipping = node1.r_clipping if edge.strand1 else node1.l_clipping
        node2_clipping = node2.l_clipping if edge.strand2 else node2.r_clipping

        long_enough1 = node1_clipping >= edge.ovlp
        long_enough2 = node2_clipping >= edge.ovlp
        # If both clippings are bigger than the overlap, we can continue to the next edge,
        # as we're not clipping anything.
        if long_enough1 and long_enough2:
            continue

        nr_edges1 = node1.r_edges if edge.strand1 else node1.l_edges
        nr_edges2 = node2.l_edges if edge.strand2 else node2.r_edges

        # First, check if one of the nodes is longer than the clipping,
        #    then, the other node will be clipped.
        # Otherwise, clip the node with the bigger number of edges.
        # If equal,  clip the longer node.

        #              2          1      1                        1          2      2
        if (long_enough2, nr_edges1, node1.seq_len) > (long_enough1, nr_edges2, node2.seq_len):
            if edge.strand1:
                assert node1.r_clipping <= edge.ovlp
                node1.r_clipping = edge.ovlp
            else:
                assert node1.l_clipping <= edge.ovlp
                node1.l_clipping = edge.ovlp
        else:
            if edge.strand2:
                assert node2.l_clipping <= edge.ovlp
                node2.l_clipping = edge.ovlp
            else:
                assert node2.r_clipping <= edge.ovlp
                node2.r_clipping = edge.ovlp

    c_stop = perf_counter()
    print("Nodes clipped in {}s".format(c_stop-c_start), file=sys.stderr)

def calculate_covs(alignment_fname, nodes, edges):
    '''This function takes the GAF file as an input, as well as the dictionary with the Node objects, in order to count the number of aligned base pairs per node.'''

    nodes = {k: v for k, v in nodes.items() if v.clipped_len() > 0}  #Filter out nodes where the left clipping point is bigger (or the same) than the rigth clipping one

    # We create a dict to save the number of aligned bp per node, and we also initiate a counter for the total number of aligned bp in the non-clipped area of the node
    coverages = {key:0 for key in nodes}

    # Create list with read lengths
    read_length = []

    nr_align = 0
    a_start = perf_counter()
    # We read the GAF file, and count the aligned bp per each node of each alignment
    with open(alignment_fname,"r") as alignment_file:
        prev = None
        for line in alignment_file:
            nr_align += 1
            columns = line.split("\t")
            if prev == columns[0]:
                continue
            prev = columns[0]

            # Append read length
            read_length.append(int(columns[1]))

            # Start and end positions (relative to the path)
            start_pos = int(columns[7])
            end_pos = int(columns[8])

            # We create a list of lists, with the bigger one having all the nodes present in the alignment, and then, for each node, we save the node name and its strand
            aln_nodes = [[j, i == ">"] for i, j in zip(re.findall('[<>]',columns[5]),re.split('[<>]',columns[5])[1:])]

            #Add "edge coverage" to the edge dictionary
            for i in range(len(aln_nodes)-1):
                n1=aln_nodes[i][0]
                n2=aln_nodes[i+1][0]
                s1="+" if aln_nodes[i][1] else "-"
                s2="+" if aln_nodes[i+1][1] else "-"
                s11="-" if s1=="+" else "+"
                s22="+" if s2=="-" else "-"
                edge1 = "e_{}_{}_{}_{}".format(n1,s1,n2,s2)
                edge2 = "e_{}_{}_{}_{}".format(n2,s22,n1,s11)
                if edges.get(edge1):
                    edges[edge1].sup_reads += 1
                if edges.get(edge2):
                    edges[edge2].sup_reads += 1

            # Asserting that we have at least one node in the list
            if len(aln_nodes) == 0:
                continue

            # We first check if we have just one node, because if that's the case, we have to look for the start and the end of the alignment in that same node. If we have more than one node, we don't need to look for the end of the alignment in the first one
            if len(aln_nodes) == 1:
                if nodes.get(aln_nodes[0][0]) != None:
                    if aln_nodes[0][1]:    # Positive strand
                        # We want the number of base pairs within the non-clipped area of the node. So, the end position of interest will be the minimum value between the end of the alignment and the end of the non-clipped area of the node. Same reasoning for the start
                        cov = max(nodes[aln_nodes[0][0]].single_node(start_pos, end_pos), 0)
                        coverages[aln_nodes[0][0]] += cov
                        if nodes[aln_nodes[0][0]].bins != None:
                            update_bins(nodes[aln_nodes[0][0]], start_pos, end_pos)

                    else:       # Negative strand
                        cov = max(nodes[aln_nodes[0][0]].single_node(start_pos, end_pos, False), 0)
                        coverages[aln_nodes[0][0]] += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], int(columns[6]) - end_pos, int(columns[6]) - start_pos)

            # If there's more than one node, we need to find the starting point in the first node and the end point in the last one, as the middle nodes will be aligned entirely
            else:
                if nodes.get(aln_nodes[0][0]) != None:
                # For the first node, we want the starting position and then counting the difference between the end of the non-clipped area and that position.
                    if aln_nodes[0][1]:
                        # Here, the count is made by subtracting the maximum value between the right end of the non-clipped area and the start postion by the maximum between the left end of the non-clipped area and the same start position.
                        # In case the start position is the maximum value in both cases, this means that it's located right of the non-clipped area, meaning that there's no aligned bp in that same area.
                        cov = nodes[aln_nodes[0][0]].first_node(start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], start_pos, nodes[aln_nodes[0][0]].right_end())

                    else:
                        # Here, the same reasoning as above applies, just in the reverse direction, hence the use of the minimum, instead of the maximum
                        cov = nodes[aln_nodes[0][0]].last_node(nodes[aln_nodes[0][0]].seq_len - start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], nodes[aln_nodes[0][0]].l_clipping, nodes[aln_nodes[0][0]].seq_len - start_pos)

                # We iterate on all the nodes in the middle of the alignment (if they exist), those where the entire non-clipped area of the node will be counted, given that the alignment spans the entire node
                for i in range(1,len(aln_nodes)-1):
                    if nodes.get(aln_nodes[i][0]) != None:
                        cov = nodes[aln_nodes[i][0]].clipped_len()
                        coverages[aln_nodes[i][0]] += cov
                        if nodes[aln_nodes[i][0]].bins != None: update_bins(nodes[aln_nodes[i][0]], nodes[aln_nodes[i][0]].l_clipping, nodes[aln_nodes[i][0]].right_end())

                # Lastly, we check for the last node, to find the ending position of the alignment. The same reasoning as for the first node applies here.
                if nodes.get(aln_nodes[-1][0]) != None:
                    if aln_nodes[-1][1]:
                        node_end = nodes[aln_nodes[-1][0]].seq_len - (int(columns[6]) - end_pos)
                        cov = nodes[aln_nodes[-1][0]].last_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], nodes[aln_nodes[-1][0]].l_clipping, node_end)

                    else:
                        node_end = int(columns[6]) - end_pos
                        cov = nodes[aln_nodes[-1][0]].first_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], node_end, nodes[aln_nodes[-1][0]].right_end())

    a_stop = perf_counter()
    print("Read {} alignments in {}s".format(nr_align,a_stop-a_start), file=sys.stderr)

    with open('cov.csv', 'w') as f:
        f.write('node\tlen\tcov\n')
        for node in nodes.values():
            f.write(f'{node.name}\t{node.clipped_len()}\t{coverages[node.name]}\n')
            if node.bins:
                for binsize, values in node.bins:
                    for v in values:
                        f.write(f'{node.name}\t{binsize}\t{v}\n')

    ae, loce, scalee = _fit_skewnorm(read_length)

    return coverages, (ae, loce, scalee)

def _fit_skewnorm(read_lens: list):
    '''Function to fit a skew normal distribution to the read lengths distribution.'''
    from scipy.stats import skewnorm as sn

    subset_size = 100000
    arr = np.array(read_lens)
    if len(read_lens) > subset_size:
        ixs = np.random.choice(len(read_lens), subset_size, replace=False)
        arr = arr[ixs]

    return sn.fit(arr)

def bin_nodes(nodes, bin_size = 100):
    '''Function to select nodes to bin and iniate the bin coverages'''
    nodes_to_bin = list()
    b_start = perf_counter()
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

    b_stop = perf_counter()
    print("Nodes binned in {}s".format(b_stop-b_start), file=sys.stderr)

    return nodes_to_bin


def update_bins(node, start, end):
    '''Function to add coverage to the bins.'''
    # First, iterate on the list of bins, each bin_size at a time.
    for (bin_size, cov_bins) in node.bins:
        # The first covered bin is given by the start position in the clipped node per the bin_size (integer division). It will be the first bin in case the read starts before the left clipping end.
        i = max(0, (start - node.l_clipping) // bin_size)
        # The last covered bin is given as the first one. In case the read ends outside the bins, giving a j higher than the number of bins, we take the number of bins instead
        j = min(cov_bins.size, (end - node.l_clipping - 1) // bin_size + 1)
        if start <= node.l_clipping and j > cov_bins.size:
            # If the read fully overlaps the node, we add coverage to all the bins
            cov_bins[:] += bin_size
        else:
            # If not, we define the start and end positions within the clipped region, and then add the corresponding coverage to each bin
            start -= node.l_clipping
            end -= node.l_clipping
            for k in range(i, j):
                b_start = k * bin_size
                b_end = (k+1) * bin_size # - 1
                assert min(end, b_end) - max(start, b_start) >= 0, "Node {}: min({}, {}) - max({}, {}) is negative.".format(node.name, end, b_end, start, b_start)
                cov_bins[k] += min(end, b_end) - max(start, b_start)


def filter_bins(nodes, nodes_to_bin, sel_size = 100):
    '''Funtion to filter bins by quality'''
    f_start = perf_counter()
    bp_cov_per_node = defaultdict()  # Dictionary to save the bins that pass the filtering criteria
    mean_per_node = defaultdict()  # Dictionary to save the average bin coverage for all the binned nodes

    dbg = open('filtering.txt', 'w')
    dbg.write('name\tstatus\n')

    # Iterate over binned_nodes
    for node in nodes_to_bin:
        binned = False
        for (bin_size, cov_bins) in nodes[node].bins:
            if bin_size == sel_size:
                binned = True
                # Keep only nodes with 10 bins or more and with at least one bin with a coverage bigger than the bin size
                if np.any(cov_bins >= sel_size):
                    bp_cov_per_node[node] = cov_bins
                    mean_per_node[node] = np.mean(cov_bins)   # Compute the mean bin coverage per node
                else:
                    dbg.write(f'{node}\tlow_cov\n')
        if not binned:
            dbg.write(f'{node}\ttoo_short\n')

    # Remove top and bottom 3% of nodes, based on the mean bin coverage. Then, remove bins with coverage bigger or equal to 3 times the median bin coverage of the node.
    TOP_PERC = 3
    thresh = np.quantile(np.array(list(mean_per_node.values())), [TOP_PERC/100, (100 - TOP_PERC)/100])
    bins_node = {node: bins for node,bins in bp_cov_per_node.items() if thresh[0] <= mean_per_node[node] <= thresh[1]}
    for node, m in mean_per_node.items():
        if m < thresh[0]:
            dbg.write(f'{node}\tbottom_perc\n')
        elif m > thresh[1]:
            dbg.write(f'{node}\ttop_perc\n')
        else:
            dbg.write(f'{node}\tuse\n')

    f_stop = perf_counter()
    print("Bins filtered in {}s".format(f_stop-f_start), file=sys.stderr)

    dbg.close()

    return bins_node

def compute_bins_array(bins_node):
    b_start = perf_counter()
    N_POINTS = 7
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
            prev_start = curr_start

    b_stop = perf_counter()
    print("Bins array computed in {}s".format(b_stop-b_start), file=sys.stderr)

    with open('bins.csv', 'w') as f:
        f.write('level\tcov\n')
        for i, bins in enumerate(bins_array):
            for v in bins:
                f.write(f'{i}\t{v}\n')

    return bins_array

def estimate_mean_std(counts, bp_step, ROUND_BINS):
    '''Function to estimate mean and standard deviation per bin size.'''

    f = open('ab.csv', 'w')
    f.write('m\ts\tr\tp\ts\tlik\n')

    def MLE_NBinom(parameters):
        # Changing parameters
        m, sd = parameters[:2]

        sd = max(sqrt(m)*1.0001, sd)
        r = m**2 / (sd**2 - m)
        p = m / sd**2


        s = sum(parameters[2:])
        cs = np.log(parameters[2:]/s)
        rs = np.maximum(np.arange(N_CN), 0.01) * r

        sum_dist = [count * special.logsumexp(cs + stats.nbinom.logpmf((i + 1/2)*bp_step, rs, p)) for i, count in enumerate(counts)]

        LL = min(1e30, -np.sum(sum_dist))
        s_str = ','.join(map('{:.8f}'.format, parameters[2:] / s))
        f.write(f'{m:.5f}\t{sd:.5f}\t{r:.5f}\t{p:.8f}\t{s_str}\t{LL:.8f}\n')

        # Save best LL and respective params
        nonlocal params_max
        nonlocal ll_max
        if LL < ll_max:
            ll_max = LL
            params_max = tuple(parameters)

        return LL

    # Iterate over the array and get the coverage value with the highest frequency (including some padding)
    s_max = 0
    k_max = 0
    cum_sum = np.cumsum(counts)
    n = counts.size - 1
    WINDOW = ROUND_BINS // 2
    for i in range(WINDOW, counts.size - WINDOW):
        start1 = i - WINDOW - 1
        end1 = i + WINDOW
        start2 = max(2*i-WINDOW-1,i+WINDOW)
        end2 = 2*i+WINDOW

        s = cum_sum[min(end2, n)] - cum_sum[np.clip(start2, end1, n)] + cum_sum[min(end1, n)] - cum_sum[start1]
        if s > s_max:
            s_max = s
            k_max = i

    # Create a vector of coefficients for each copy number (up to copy number N_CN-1)
    N_CN = 4
    cs = []
    for j in range(N_CN):
        cs.append(sum(counts[int((j-1/2)*k_max):int((j+1/2)*k_max)+1]))

    cs = np.array(cs)
    # Normalize the vector, dividing it by the total sum of coefficients
    cs = cs / np.sum(cs)

    # Get initial values of m and sd
    m0 = (k_max + 0.5) * bp_step
    sd = m0/2
    f.write(f'{m0:.5f}\t{sd:.5f}\tNA\tNA\tNA\tNA\n')

    params_max = None
    ll_max = 1e30

    sol = optimize.minimize(MLE_NBinom, np.append([m0, sd], cs), bounds=((m0 * 0.8, m0 * 1.2), (m0 * 0.1, m0 * 10),) + tuple((c*0.5, c*1.5+1e-5) for c in cs), method='Nelder-Mead')
    f.close()

    m, sd = sol.x[:2]

    return m, sd

def alpha_and_beta(bins_array, sel_size = 100):
    '''Get alpha and beta coefficients for NB parameters estimation.'''

    p_start = perf_counter()
    # Create dictionary of bins per size
    cov = defaultdict(list)
    for arr in bins_array:
        cov[sel_size] += arr
        sel_size *= 2

    params = defaultdict()
    TOP_PERC = 3
    ROUND_BINS = 4

    # Iterate over all bin sizes
    for size in cov:
        bins = np.array(cov[size])
        #sys.stderr.write(f"Starting bin {size}: {len(bins)}\n")

        # Filter bins by quantile
        thresh = np.quantile(bins, [TOP_PERC/100, (100 - TOP_PERC)/100])
        bins = bins[(thresh[0] <= bins) & (bins <= thresh[1])]

        # Use a step to round and generate counts for rounded coverages
        bp_step = size // ROUND_BINS
        counts = np.bincount(np.round(bins / bp_step).astype(dtype=np.int64), minlength=int(round(thresh[1])) // bp_step + 1)
        sys.stderr.write(f"{len(counts)} counts\n")

        # Get mean and standard deviation from MLE
        mean, sd = estimate_mean_std(counts, bp_step, ROUND_BINS)

        # Add them to a dictionary
        params[size] = [mean, sd]

    sizes, vals = zip(*params.items())
    means, sds = zip(*vals)

    # Compute alpha and beta coefficients from linear regression
    alpha = stats.linregress(sizes,means).slope
    beta = stats.linregress(sizes,sds).slope

    print("Alpha value: {}".format(alpha), file=sys.stderr)
    print("Beta value: {}".format(beta), file=sys.stderr)

    p_stop = perf_counter()
    print("Alpha and beta estimated in {}s".format(p_stop-p_start), file=sys.stderr)

    return alpha, beta, params

