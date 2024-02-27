from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import argparse
import numpy as np
from time import perf_counter
from sklearn.cluster import KMeans
import sys
from scipy import stats, optimize, special
from math import sqrt, dist, log
import matplotlib.pyplot as plt
import counts_to_probabs as ctp

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
        # First, we get the clipping values for the corresponding ends of the nodes
        clip_node1 = nodes[edge.node1].r_clipping if edge.strand1 else nodes[edge.node1].l_clipping
        clip_node2 = nodes[edge.node2].l_clipping if edge.strand2 else nodes[edge.node2].r_clipping

        # If both clippings are bigger than the overlap, we can continue to the next edge, as we're not clipping anything. If just one of the clippings is bigger than the overlap, we just have to clip the other node
        if (clip_node1 and clip_node2) > edge.ovlp: 
            continue
        elif clip_node1 > edge.ovlp:
            if edge.strand2:
                assert nodes[edge.node2].l_clipping <= edge.ovlp
                nodes[edge.node2].l_clipping = edge.ovlp
            else:
                assert nodes[edge.node2].r_clipping <= edge.ovlp
                nodes[edge.node2].r_clipping = edge.ovlp
        elif clip_node2 > edge.ovlp:
            if edge.strand1:
                assert nodes[edge.node1].r_clipping <= edge.ovlp
                nodes[edge.node1].r_clipping = edge.ovlp
            else:
                assert nodes[edge.node1].l_clipping <= edge.ovlp
                nodes[edge.node1].l_clipping = edge.ovlp
        # If none of the previous applies, we have to decide which node to clip using further criteria.
        else:
            # The first criteria is the number of edges connected to each end (we'll clip the end with the most number of edges, as the other one is less likely to loose the information)
            nr_edges1 = nodes[edge.node1].r_edges if edge.strand1 else nodes[edge.node1].l_edges
            nr_edges2 = nodes[edge.node2].l_edges if edge.strand2 else nodes[edge.node2].r_edges

            if nr_edges1 < nr_edges2:
                if edge.strand2:
                    assert nodes[edge.node2].l_clipping <= edge.ovlp
                    nodes[edge.node2].l_clipping = edge.ovlp
                else:
                    assert nodes[edge.node2].r_clipping <= edge.ovlp
                    nodes[edge.node2].r_clipping = edge.ovlp
            elif nr_edges1 > nr_edges2:
                if edge.strand1:
                    assert nodes[edge.node1].r_clipping <= edge.ovlp
                    nodes[edge.node1].r_clipping = edge.ovlp
                else:
                    assert nodes[edge.node1].l_clipping <= edge.ovlp
                    nodes[edge.node1].l_clipping = edge.ovlp
            # If both ends have the same number of edges, we'll clip the bigger node of the two.
            else:
                if nodes[edge.node1].seq_len > nodes[edge.node2].seq_len:
                    if edge.strand1:
                        assert nodes[edge.node1].r_clipping <= edge.ovlp
                        nodes[edge.node1].r_clipping = edge.ovlp
                    else:
                        assert nodes[edge.node1].l_clipping <= edge.ovlp
                        nodes[edge.node1].l_clipping = edge.ovlp
                else:
                    if edge.strand2:
                        assert nodes[edge.node2].l_clipping <= edge.ovlp
                        nodes[edge.node2].l_clipping = edge.ovlp
                    else:
                        assert nodes[edge.node2].r_clipping <= edge.ovlp
                        nodes[edge.node2].r_clipping = edge.ovlp

    c_stop = perf_counter()
    print("Nodes clipped in {}s".format(c_stop-c_start), file=sys.stderr)

def calculate_covs(alignment_fname, nodes, edges):
    '''This function takes the GAF file as an input, as well as the dictionary with the Node objects, in order to count the number of aligned base pairs per node.'''

    nodes = {k: v for k, v in nodes.items() if v.clipped_len() > 0}  #Filter out nodes where the left clipping point is bigger (or the same) than the rigth clipping one

    # We create a dict to save the number of aligned bp per node, and we also initiate a counter for the total number of aligned bp in the non-clipped area of the node
    coverages = {key:0 for key in nodes}
    #total_bp_matches = 0
    #read_depth = {key:0 for key in nodes if nodes[key].clipped_len() < 101}

    read_length = []

    nr_align = 0
    #a_start = perf_counter()
    # We read the GAF file, and count the aligned bp per each node of each alignment 
    with open(alignment_fname,"r") as alignment_file:
        for line in alignment_file:
            nr_align += 1
            columns = line.split("\t")
            
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
                        #total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: 
                            #b_start = perf_counter()
                            update_bins(nodes[aln_nodes[0][0]], start_pos, end_pos)
                            #b_stop = perf_counter()
                            #print("Updating bins for node {}: {}s".format(aln_nodes[0][0], b_stop-b_start), file=sys.stderr)
                        '''if aln_nodes[0][0] in read_depth:
                            if start_pos <= nodes[aln_nodes[0][0]].middle_point() < end_pos:
                                read_depth[aln_nodes[0][0]] += 1'''
                    else:       # Negative strand
                        cov = max(nodes[aln_nodes[0][0]].single_node(start_pos, end_pos, False), 0)
                        coverages[aln_nodes[0][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], int(columns[6]) - end_pos, int(columns[6]) - start_pos)

                        '''if aln_nodes[0][0] in read_depth:
                            if nodes[aln_nodes[0][0]].seq_len - end_pos <= nodes[aln_nodes[0][0]].middle_point() < nodes[aln_nodes[0][0]].seq_len - start_pos:
                                read_depth[aln_nodes[0][0]] += 1'''

            # If there's more than one node, we need to find the starting point in the first node and the end point in the last one, as the middle nodes will be aligned entirely
            else:
                if nodes.get(aln_nodes[0][0]) != None:
                # For the first node, we want the starting position and then counting the difference between the end of the non-clipped area and that position.  
                    if aln_nodes[0][1]:
                        # Here, the count is made by subtracting the maximum value between the right end of the non-clipped area and the start postion by the maximum between the left end of the non-clipped area and the same start position.
                        # In case the start position is the maximum value in both cases, this means that it's located right of the non-clipped area, meaning that there's no aligned bp in that same area.
                        cov = nodes[aln_nodes[0][0]].first_node(start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], start_pos, nodes[aln_nodes[0][0]].right_end())

                        '''if aln_nodes[0][0] in read_depth:
                            if start_pos <= nodes[aln_nodes[0][0]].middle_point():
                                read_depth[aln_nodes[0][0]] += 1'''

                    else:
                        # Here, the same reasoning as above applies, just in the reverse direction, hence the use of the minimum, instead of the maximum
                        cov = nodes[aln_nodes[0][0]].last_node(nodes[aln_nodes[0][0]].seq_len - start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], nodes[aln_nodes[0][0]].l_clipping, nodes[aln_nodes[0][0]].seq_len - start_pos)

                        '''if aln_nodes[0][0] in read_depth:
                            if nodes[aln_nodes[0][0]].middle_point() < nodes[aln_nodes[0][0]].seq_len - start_pos:
                                read_depth[aln_nodes[0][0]] += 1'''

                # We iterate on all the nodes in the middle of the alignment (if they exist), those where the entire non-clipped area of the node will be counted, given that the alignment spans the entire node
                for i in range(1,len(aln_nodes)-1):
                    if nodes.get(aln_nodes[i][0]) != None:
                        cov = nodes[aln_nodes[i][0]].clipped_len()
                        coverages[aln_nodes[i][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[i][0]].bins != None: update_bins(nodes[aln_nodes[i][0]], nodes[aln_nodes[i][0]].l_clipping, nodes[aln_nodes[i][0]].right_end())

                        '''if aln_nodes[i][0] in read_depth:
                            read_depth[aln_nodes[i][0]] += 1'''
                
                # Lastly, we check for the last node, to find the ending position of the alignment. The same reasoning as for the first node applies here.
                if nodes.get(aln_nodes[-1][0]) != None:
                    if aln_nodes[-1][1]:
                        node_end = nodes[aln_nodes[-1][0]].seq_len - (int(columns[6]) - end_pos)
                        cov = nodes[aln_nodes[-1][0]].last_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], nodes[aln_nodes[-1][0]].l_clipping, node_end)

                        '''if aln_nodes[-1][0] in read_depth:
                            if nodes[aln_nodes[-1][0]].middle_point() < node_end:
                                read_depth[aln_nodes[-1][0]] += 1'''
                        
                    else:
                        node_end = int(columns[6]) - end_pos
                        cov = nodes[aln_nodes[-1][0]].first_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        #total_bp_matches += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], node_end, nodes[aln_nodes[-1][0]].right_end())

                        '''if aln_nodes[-1][0] in read_depth:
                            if node_end <= nodes[aln_nodes[-1][0]].middle_point():
                                read_depth[aln_nodes[-1][0]] += 1'''

    #a_stop = perf_counter()
    #print("Read {} alignments in {}s".format(nr_align,a_stop-a_start), file=sys.stderr)

    return coverages, read_length #, total_bp_matches, read_depth


def calculate_avg_cov(nodes, total_bp_matches, ploidy):
    total_length=0
    for node in nodes:
        total_length += nodes[node].clipped_len()
    avg_cov = total_bp_matches / (total_length*ploidy)
    return avg_cov


def bin_nodes(nodes, bin_sizes: list):
    '''Function to select nodes to bin and iniate the bin coverages'''
    #N_BIGGEST_NODES = 100  # A fixed number of nodes to bin
    '''BINNING_LEN = 3000000
    MIN_LEN = bin_sizes[0]
    sum_len = 0
    #nr_to_bin = min(len(nodes), N_BIGGEST_NODES)  # If the number of nodes is smaller than the fixed number, bin all of them
    
    # Get the nodes to bin, corresponding to the nr_to_bin biggest nodes.
    #nodes_to_bin = [k.name for k in sorted(nodes.values(), reverse=True)[:nr_to_bin]]
    sorted_nodes = [k for k in sorted(nodes.values(), reverse=True)]'''
    nodes_to_bin = list()
    for node in nodes:
        if nodes[node].clipped_len() >= min(bin_sizes):
            nodes_to_bin.append(nodes[node].name)
    
    #bin_sizes = [size for size in range(bin_sizes[0], bin_sizes[1]+1, bin_sizes[2])]
    for node in nodes_to_bin:
        # For each node, create a list of tuples with (bin size, array with the size of the nr of bins)
        nodes[node].bins = list()
        for size in bin_sizes:
            nr_bins = nodes[node].clipped_len() // size
            if nr_bins > 0:
                nodes[node].bins.append((size,np.zeros(nr_bins, dtype=np.uint64)))

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



def nb_parameters(bins, sel_size = 100):#, bins2):
    '''Function to estimate the parameters of the Negative Binomial distribution.'''

    def MLE_NBinom(parameters):
        # Changing parameters
        r, p = parameters[:2]

        # Check if mean is within 20% distance from the initial value
        m = r * (1-p) / p
        if m > k_max*1.2 or m < k_max*0.8:
            return 1e30
        
        s = sum(parameters[2:])
        cs = np.log(parameters[2:]/s)
        rs = np.maximum(np.arange(N_CN), 0.01) * r
        
        sum_dist = [count * special.logsumexp(cs + stats.nbinom.logpmf(i, rs, p)) for i, count in enumerate(counts_nozero)]

        LL = min(1e30, -np.sum(sum_dist))

        # Save best LL and respective params
        nonlocal params_max
        nonlocal ll_max
        if LL < ll_max:
            ll_max = LL
            params_max = tuple(parameters)
        
        return LL

    # Add noise to the bins, to flatten the curve, instead of having peaks
    noise_bins = bins + stats.randint.rvs(-sel_size, 2*sel_size, size=len(bins))
    # Transform the bins into frequency counts for all nodes. Turn that into an array of length (max coverage + 1)
    counts = Counter(noise_bins)
    counts_array = np.array([counts[i] for i in range(0,int(max(counts.keys()))+1)])
    counts_nozero = counts_array[counts_array > 0]

    # Iterate over the array and get the coverage value with the highest frequency (including some padding)
    s_max = 0
    k_max = 0
    WINDOW = 50
    for i in range(counts_array.size):
        s = sum(counts_array[i-WINDOW:i+WINDOW+1]) + sum(counts_array[max(2*i-WINDOW,i+WINDOW+1):2*i+WINDOW+1])
        if s > s_max:
            s_max = s
            k_max = i
    
    # Create a vector of coefficients for each copy number (up to copy number N_CN-1)
    N_CN = 7
    cs = []
    for j in range(N_CN):
        cs.append(sum(counts_array[int((j-1/2)*k_max):int((j+1/2)*k_max)+1]))
        # cs.append(0)
        # for i in range((j-1/2)*k_max, (j+1/2)*k_max+1):
        #     cs[j] += counts_array[i]
    
    cs = np.array(cs)
    # Normalize the vector, dividing it by the total sum of coefficients
    cs = cs / np.sum(cs)

    # Get initial values of r and p
    sd = k_max/4
    r_1 = k_max**2 / (sd**2 - k_max)
    p_1 = k_max/sd**2

    params_max = None
    ll_max = 1e30
    g_start = perf_counter()
    sol = optimize.minimize(MLE_NBinom, np.append([r_1, p_1], cs), bounds=((1,100),(0,1),) + ((0,1),)*N_CN, method='Nelder-Mead', options=dict(maxiter=10))

    r,p = sol.x[:2]

    g_stop = perf_counter()
    print("Parameters estimated in {}s".format(g_stop-g_start), file=sys.stderr)
    print("Optimization message: {}".format(sol.message), file=sys.stderr)
    print("Value of objective function: {}".format(sol.fun), file=sys.stderr)
    print("Value of computed LL max: {}".format(ll_max), file=sys.stderr)
    print("List of parameters for LL max: {}".format(params_max), file=sys.stderr)

    print(sol.x)

    return r,p


def node_covs(nodes, alignment, ploidy, p, r_smaller, smallest_size, outfile):
    '''Function to write the file with the node coverages.'''

    coverages, total_bp_matches = calculate_covs(alignment, nodes)
    print("Calculated", len(coverages), "coverages")

    avg_cov = calculate_avg_cov(nodes, total_bp_matches, ploidy)
    print("Average coverage:", avg_cov)

    print("Writing results into", outfile)
    write_results(coverages.items(), avg_cov, p, r_smaller, smallest_size, outfile)

def write_separate_results(coverages, nodes, nodes_to_bin, out_fname):
    with open("cov-with-size_" + out_fname,"w") as out :
        out.write("Node,Size,Coverage\n")
        for node in coverages:
            out.write(node + "," + str(nodes[node].clipped_len()) + "," + str(coverages[node]) + "\n")
    
    with open("cov-per-bin_" + out_fname,"w") as out_bin :
        out_bin.write("Bin_size,Coverage\n")
        for node in nodes_to_bin:
            for bins in nodes[node].bins:
                for each in bins[1]:
                    out_bin.write(str(bins[0]) + "," + str(each) + "\n")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", help="The GFA file.", required=True)
    parser.add_argument("-a", "--graphalignment", help="The GAF file.", required=True)
    parser.add_argument("-c", "--outcov", help="The name for the output csv file with the node coverages", required=True)
    parser.add_argument("-p", "--ploidy", type=int, default=2, help="Ploidy of the dataset. (default:%(default)s)")
    parser.add_argument("-b", "--bin_size", nargs=3, default=[5000,50000,5000], metavar=('MIN', 'MAX', 'STEP'), type=int, help="Set the range for the bin size to use for the NB parameters' estimation. (default:%(default)s)")

    args = parser.parse_args()
    np.set_printoptions(precision=6, linewidth=sys.maxsize, suppress=True, threshold=sys.maxsize)
    bins_list = [100]
    nodes, edges = read_graph(args.graph)
    clip_nodes(nodes, edges)
    import pickle
    with open("graph-{}.tmp.pkl".format(args.graph), 'wb') as f:
        pickle.dump((nodes,edges), f)
    # nodes_to_bin = bin_nodes(nodes, bins_list) #[size for size in range(args.bin_size[0], args.bin_size[1]+1, args.bin_size[2])])
    # coverages, total_bp_matches, read_depth = calculate_covs(args.graphalignment, nodes)
    # #avg_cov = calculate_avg_cov(nodes, total_bp_matches, args.ploidy)
    # #write_results(nodes, coverages, read_depth, means_cov, vars_cov, means_rd, vars_rd, args.outcov)
    # #write_separate_results(coverages, nodes, nodes_to_bin, args.outcov)
    # filtered_bins = clustering(nodes, nodes_to_bin)
    # #filtered_1k = clustering(nodes, nodes_to_bin, 1000)
    # #r, p = nb_parameters(filtered_bins)
    # #nb_parameters(filtered_bins, filtered_1k)
    # #small_nodes(nodes, coverages)
    # #ctp.plotting(nodes, coverages, r, p, 100)

if __name__ == "__main__":
    main()

