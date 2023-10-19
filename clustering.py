from collections import defaultdict
from dataclasses import dataclass
import re
import argparse
import numpy as np
from time import perf_counter
from sklearn.cluster import KMeans
import sys
from scipy import stats, optimize
from math import sqrt, dist
import matplotlib.pyplot as plt
import counts_to_probabs as ctp

@dataclass
class Edge:
    node1: str
    node2: str
    strand1: bool
    strand2: bool
    ovlp: int

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
    
    edges = list()
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
                edges.append(Edge(columns[1], columns[3], columns[2]=="+", columns[4]=="+", int(columns[5].strip("M"))))  #Append Edge object to the list of edges

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
    # Sort edges by decreasing overlap size
    edges.sort(reverse=True)

    return nodes, edges

def clip_nodes(nodes,edges):
    '''This function iterates over the list of edges to decide where to clip the nodes, in order to get rid of the overlaps, aiming to get the maximum amount of non-duplicated information. It takes as an input the list of edges and 
    dictionary of nodes produced by read_graph(), outputting a new dictionary of nodes with the updated clipping information.'''
    c_start = perf_counter()
    # Iterate over list of edges, and clip the nodes to get rid of overlaps.
    for edge in edges:
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

def calculate_covs(alignment_fname, nodes):
    '''This function takes the GAF file as an input, as well as the dictionary with the Node objects, in order to count the number of aligned base pairs per node.'''

    nodes = {k: v for k, v in nodes.items() if v.clipped_len() > 0}  #Filter out nodes where the left clipping point is bigger (or the same) than the rigth clipping one

    # We create a dict to save the number of aligned bp per node, and we also initiate a counter for the total number of aligned bp in the non-clipped area of the node
    coverages = {key:0 for key in nodes}
    total_bp_matches = 0
    read_depth = {key:0 for key in nodes if nodes[key].clipped_len() < 101}

    nr_align = 0
    #a_start = perf_counter()
    # We read the GAF file, and count the aligned bp per each node of each alignment 
    with open(alignment_fname,"r") as alignment_file:
        for line in alignment_file:
            nr_align += 1
            columns = line.split("\t")
            
            # Start and end positions (relative to the path)
            start_pos = int(columns[7])
            end_pos = int(columns[8])

            # We create a list of lists, with the bigger one having all the nodes present in the alignment, and then, for each node, we save the node name and its strand
            aln_nodes = [[j, i == ">"] for i, j in zip(re.findall('[<>]',columns[5]),re.split('[<>]',columns[5])[1:])]

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
                        total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: 
                            #b_start = perf_counter()
                            update_bins(nodes[aln_nodes[0][0]], start_pos, end_pos)
                            #b_stop = perf_counter()
                            #print("Updating bins for node {}: {}s".format(aln_nodes[0][0], b_stop-b_start), file=sys.stderr)
                        if aln_nodes[0][0] in read_depth:
                            if start_pos <= nodes[aln_nodes[0][0]].middle_point() < end_pos:
                                read_depth[aln_nodes[0][0]] += 1
                    else:       # Negative strand
                        cov = max(nodes[aln_nodes[0][0]].single_node(start_pos, end_pos, False), 0)
                        coverages[aln_nodes[0][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], int(columns[6]) - end_pos, int(columns[6]) - start_pos)

                        if aln_nodes[0][0] in read_depth:
                            if nodes[aln_nodes[0][0]].seq_len - end_pos <= nodes[aln_nodes[0][0]].middle_point() < nodes[aln_nodes[0][0]].seq_len - start_pos:
                                read_depth[aln_nodes[0][0]] += 1

            # If there's more than one node, we need to find the starting point in the first node and the end point in the last one, as the middle nodes will be aligned entirely
            else:
                if nodes.get(aln_nodes[0][0]) != None:
                # For the first node, we want the starting position and then counting the difference between the end of the non-clipped area and that position.  
                    if aln_nodes[0][1]:
                        # Here, the count is made by subtracting the maximum value between the right end of the non-clipped area and the start postion by the maximum between the left end of the non-clipped area and the same start position.
                        # In case the start position is the maximum value in both cases, this means that it's located right of the non-clipped area, meaning that there's no aligned bp in that same area.
                        cov = nodes[aln_nodes[0][0]].first_node(start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], start_pos, nodes[aln_nodes[0][0]].right_end())

                        if aln_nodes[0][0] in read_depth:
                            if start_pos <= nodes[aln_nodes[0][0]].middle_point():
                                read_depth[aln_nodes[0][0]] += 1

                    else:
                        # Here, the same reasoning as above applies, just in the reverse direction, hence the use of the minimum, instead of the maximum
                        cov = nodes[aln_nodes[0][0]].last_node(nodes[aln_nodes[0][0]].seq_len - start_pos)
                        coverages[aln_nodes[0][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[0][0]].bins != None: update_bins(nodes[aln_nodes[0][0]], nodes[aln_nodes[0][0]].l_clipping, nodes[aln_nodes[0][0]].seq_len - start_pos)

                        if aln_nodes[0][0] in read_depth:
                            if nodes[aln_nodes[0][0]].middle_point() < nodes[aln_nodes[0][0]].seq_len - start_pos:
                                read_depth[aln_nodes[0][0]] += 1

                # We iterate on all the nodes in the middle of the alignment (if they exist), those where the entire non-clipped area of the node will be counted, given that the alignment spans the entire node
                for i in range(1,len(aln_nodes)-1):
                    if nodes.get(aln_nodes[i][0]) != None:
                        cov = nodes[aln_nodes[i][0]].clipped_len()
                        coverages[aln_nodes[i][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[i][0]].bins != None: update_bins(nodes[aln_nodes[i][0]], nodes[aln_nodes[i][0]].l_clipping, nodes[aln_nodes[i][0]].right_end())

                        if aln_nodes[i][0] in read_depth:
                            read_depth[aln_nodes[i][0]] += 1
                
                # Lastly, we check for the last node, to find the ending position of the alignment. The same reasoning as for the first node applies here.
                if nodes.get(aln_nodes[-1][0]) != None:
                    if aln_nodes[-1][1]:
                        node_end = nodes[aln_nodes[-1][0]].seq_len - (int(columns[6]) - end_pos)
                        cov = nodes[aln_nodes[-1][0]].last_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], nodes[aln_nodes[-1][0]].l_clipping, node_end)

                        if aln_nodes[-1][0] in read_depth:
                            if nodes[aln_nodes[-1][0]].middle_point() < node_end:
                                read_depth[aln_nodes[-1][0]] += 1
                        
                    else:
                        node_end = int(columns[6]) - end_pos
                        cov = nodes[aln_nodes[-1][0]].first_node(node_end)
                        coverages[aln_nodes[-1][0]] += cov
                        total_bp_matches += cov
                        if nodes[aln_nodes[-1][0]].bins != None: update_bins(nodes[aln_nodes[-1][0]], node_end, nodes[aln_nodes[-1][0]].right_end())

                        if aln_nodes[-1][0] in read_depth:
                            if node_end <= nodes[aln_nodes[-1][0]].middle_point():
                                read_depth[aln_nodes[-1][0]] += 1

    #a_stop = perf_counter()
    #print("Read {} alignments in {}s".format(nr_align,a_stop-a_start), file=sys.stderr)

    return coverages, total_bp_matches, read_depth


def calculate_avg_cov(nodes, total_bp_matches, ploidy):
    total_length=0
    for node in nodes:
        total_length += nodes[node].clipped_len()
    avg_cov = total_bp_matches / (total_length*ploidy)
    return avg_cov


def bin_nodes(nodes, bin_size):
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
        if nodes[node].clipped_len() >= 100:
            #break
        #else:
            nodes_to_bin.append(nodes[node].name)
            #sum_len += node.clipped_len()
    
    #bin_sizes = [size for size in range(bin_sizes[0], bin_sizes[1]+1, bin_sizes[2])]
    for node in nodes_to_bin:
        # For each node, create a list of tuples with (bin size, array with the size of the nr of bins)
        nodes[node].bins = list()
        nr_bins = nodes[node].clipped_len() // bin_size
        if nr_bins > 0:
            nodes[node].bins.append((bin_size,np.zeros(nr_bins, dtype=np.uint64)))

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


def clustering(nodes, nodes_to_bin, sel_size = 100):
    '''Function to cluster the "best nodes" for the parameter estimation.'''

    kbp_cov_per_node = defaultdict()
    #sel_size = 1000
    g_start = perf_counter()
    for node in nodes_to_bin:  # Iterate over binned_nodes
        for (bin_size, cov_bins) in nodes[node].bins:
            if bin_size == sel_size:
                #remove zeros or not
                if cov_bins[cov_bins >= 1].size > 0:
                    kbp_cov_per_node[node] = cov_bins[cov_bins >= 1]

    mean_and_sd = []
    nodes_kbp = []
    for node in kbp_cov_per_node:
        mean_and_sd.append([np.mean(kbp_cov_per_node[node]), np.std(kbp_cov_per_node[node])])
        nodes_kbp.append(node)
    
    mean_and_sd = np.array(mean_and_sd)
    
    mean_point = np.mean(mean_and_sd, axis=0)
    dist_to_point = np.array([dist(mean_point, mean_and_sd[i]) for i in range(len(mean_and_sd))])
    max_dist = np.quantile(dist_to_point, 0.95)
    mean_and_sd_to_cluster = np.array([coord for coord, distance in zip(mean_and_sd, dist_to_point) if distance <= max_dist])

    kmeans = KMeans(n_clusters=2).fit(mean_and_sd_to_cluster)
    #kmeans = KMeans(n_clusters=3).fit(mean_and_sd)
    #agclu = AgglomerativeClustering(n_clusters=3).fit(mean_and_sd)

    #cluster_idx = np.argmax(kmeans.cluster_centers_[:,0], axis=0)
    #cluster_idx = np.argsort(kmeans.cluster_centers_[:,0], axis=0)[len(kmeans.cluster_centers_[:,0])//2]
    cluster_idx = np.argmax(kmeans.cluster_centers_[:,0], axis=0)
    cluster_center = kmeans.cluster_centers_[cluster_idx]

    mean_and_sd_cluster = [coord for coord, cluster in zip(mean_and_sd, kmeans.labels_) if cluster == cluster_idx]
    nodes_cluster = [node for node, cluster in zip(nodes_kbp, kmeans.labels_) if cluster == cluster_idx]

    dist_to_center = np.array([dist(cluster_center, mean_and_sd_cluster[i]) for i in range(len(mean_and_sd_cluster))])

    max_diff = np.quantile(dist_to_center, 0.95)
    nodes_estim = [node for node, distance in zip(nodes_cluster, dist_to_center) if distance <= max_diff]
    mean_and_sd_estim = np.array([coord for coord, distance in zip(mean_and_sd_cluster, dist_to_center) if distance <= max_diff])

    filtered_bins = np.concatenate([kbp_cov_per_node[node] for node in nodes_estim])
    print(len(filtered_bins))

    g_stop = perf_counter()
    print("Bins filtered in {}s".format(g_stop-g_start), file=sys.stderr)
    '''
    colors = [coord in mean_and_sd_estim for coord in mean_and_sd]
    plt.scatter(mean_and_sd[:,0], mean_and_sd[:,1], c=colors)
    plt.savefig("plots/Nodes_to_estimate-q0.95.png")
    '''
    # plt.clf()
    # for i in range(2):
    #     plt.scatter(mean_and_sd_to_cluster[kmeans.labels_ == i , 0], mean_and_sd_to_cluster[kmeans.labels_ == i , 1], label = i, alpha=0.05)
    # #plt.scatter(mean_and_sd[:,0], mean_and_sd[:,1], c=kmeans.labels_, label=, alpha=0.05)
    # plt.legend()
    # plt.text(10000,25000,"Center for cluster 1: "+str(kmeans.cluster_centers_[1]))
    # plt.savefig("plots/Clustering_nodes_2clusters-alpha-0.05-genome-nozeros-filtered.png")
    # plt.clf()

    #plt.scatter(mean_and_sd[:,0], mean_and_sd[:,1], c=agclu.labels_, alpha=0.05)
    #plt.savefig("plots/AgglomerativeClustering_nodes_3clusters-alpha-0.05.png")'''

    return filtered_bins



def nb_parameters(bins):#, bins2):
    '''Function to estimate the parameters of the Negative Binomial distribution.'''
    '''
    cov_per_size = defaultdict(list)
    rd_per_size = defaultdict(list)
    for node in nodes_to_bin:
        for (bin_size, cov_bins, rd_bins) in nodes[node].bins:
            cov_per_size[bin_size].extend(cov_bins)
            rd_per_size[bin_size].extend(rd_bins)

    #smallest_size = min([nodes[node].bins[i][0] for i in range(len(nodes[node].bins))])
    means_cov=defaultdict()
    vars_cov=defaultdict()
    means_rd=defaultdict()
    vars_rd=defaultdict()
    #mean_smaller = 0
    for size in cov_per_size:
        means_cov[size] = ([np.mean(cov_per_size[size])])
        vars_cov[size] = (np.var(cov_per_size[size]))
        means_rd[size] = ([np.mean(rd_per_size[size])])
        vars_rd[size] = (np.var(rd_per_size[size]))
        #if size == smallest_size: 
            #mean_smaller = np.mean(cov_per_size[size])
    '''

    def MLE_NBinom(parameters):
        r, p = parameters[0:2]
        s = sum(parameters[2:])
        c0, c1, c2, c3 = parameters[2:]/s

        sum_dist = c0 * stats.nbinom.pmf(bins, 0.01*r, p) + c1 * stats.nbinom.pmf(bins, 0.5*r, p) + c2 * stats.nbinom.pmf(bins, r, p) + c3 * stats.nbinom.pmf(bins, (3/2)*r, p)

        LL = min(1e30, -np.sum(np.log(sum_dist)))
        #print(LL)
        #print(parameters)
        return LL

    g_start = perf_counter()
    sol = optimize.minimize(MLE_NBinom, np.array([10, 0.01, 0.1, 0.1, 0.7, 0.1]), bounds=((1, 100), (0, 1), (0, 1), (0.00001, 0.99999), (0.00001, 0.99999), (0.00001, 0.99999)), method='Nelder-Mead')

    r,p,c0,c1,c2,c3 = sol.x
    n = 100
    m = 1000
    r2 = (r*(1-p))/(1-n/m*p)
    p2 = n/m * p

    g_stop = perf_counter()
    print("Parameters estimated in {}s".format(g_stop-g_start), file=sys.stderr)

    print(sol.x)
    # x_plot = np.linspace(0,50000,50001)
    # plt.hist(bins, 60, label="True values")
    # plt.plot(x_plot, stats.nbinom.pmf(x_plot, r, p)*2e7, label="NB distribution")
    # plt.legend()
    # plt.savefig("plots/Filtered_nodes_NB-100bp_to_1kb-genome-2clusters-nozeros-filtered.png")

    return r,p#,c0,c1,c2,c3

# def small_nodes(nodes, coverages, size=50):

#     small_covs = defaultdict(list)
#     for node in nodes:
#         if nodes[node].clipped_len() <= size:
#             small_covs[node] = [nodes[node].clipped_len(), coverages[node]]
    
#     covs_array = np.array(list(small_covs.values()))

#     plt.clf()
#     plt.title("Coverage per lenght of nodes under {}bp".format(size))
#     plt.scatter(covs_array[:, 0], covs_array[:, 1], alpha=0.05)
#     plt.xlabel("Node Length")
#     plt.ylabel("Coverage")
#     plt.savefig("plots/small_nodes/Cov_per_length-nodes-{}bp-alpha_0.05.png".format(size))
#     plt.clf()



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

