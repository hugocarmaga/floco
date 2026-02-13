from collections import defaultdict
from dataclasses import dataclass
import re
import numpy as np
from time import perf_counter
import gzip, builtins, sys

def open(filename, mode='r'):
    assert mode == 'r' or mode == 'w'
    if filename is None or filename == '-':
        return sys.stdin if mode == 'r' else sys.stdout
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode + 't')
    else:
        return builtins.open(filename, mode)

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

def read_graph(graph_fname):
    '''This function takes a GFA file as an input, and returns a list of Edge objects, and a dictionary of {'node_name': Node object}'''
    print("*** Starting graph preprocessing from {}".format(graph_fname), file=sys.stderr)
    edges = defaultdict()
    nodes = defaultdict()
    g_start = perf_counter()
    tmp_edges = defaultdict()
    with open(graph_fname,"r") as graph_file:
        for line in graph_file:
            columns = line.split()

            #sequence line
            if columns[0] == "S":
                # Read node length from the GFA file. If it's not present, we take it as the length of the sequence, which is the third column of the GFA file. If the length is present, we take it as the value after "LN:" in the fourth column
                node_len = int(columns[3].split(':')[2]) if len(columns) > 3 and columns[3].startswith("LN:") else len(columns[2])
                nodes[columns[1]] = Node(columns[1], node_len)  #Add Node object as a value to the dictionary of nodes, using the node name as key

            # link line
            if columns[0] == "L":
                #Check if both nodes of the edge are present in the node dictionary, otherwise, store edge in a temporary set to add it later, when we have read all the nodes. This is to avoid problems with the order of the lines in the GFA file, as we can have link lines before sequence lines.
                if nodes.get(columns[1]) == None or nodes.get(columns[3]) == None:
                    tmp_edges["e_{}_{}_{}_{}".format(columns[1],columns[2],columns[3],columns[4])] = Edge(columns[1], columns[3], columns[2]=="+", columns[4]=="+", int(columns[5].strip("M")))
                    continue
                else:
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

    # Now, we add the edges that were stored in the temporary set, as we have already read all the nodes and we know that they are present in the node dictionary.
    for edge in tmp_edges.values():
        edges["e_{}_{}_{}_{}".format(edge.node1, "+" if edge.strand1 else "-", edge.node2, "+" if edge.strand2 else "-")] = edge
        # Add number of edges connected to each side of each node
        if edge.strand1:
            nodes[edge.node1].r_edges += 1
        else:
            nodes[edge.node1].l_edges += 1

        if edge.strand2:
            nodes[edge.node2].l_edges += 1
        else:
            nodes[edge.node2].r_edges += 1

    g_stop = perf_counter()
    print("    Graph read in {}s".format(g_stop-g_start), file=sys.stderr)

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
    print("    Nodes clipped in {}s".format(c_stop-c_start), file=sys.stderr)


ONE = np.uint64(1)
MAX_U64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def filter_gaf(f):
    READ_BINS = 64

    all_names = set()
    curr_name = None
    curr_bin_size = None
    covered = None

    for line in f:
        # For faster processing, only take columns that we need.
        columns = line.split('\t', 9)
        name = columns[0]
        read_len = int(columns[1])
        start = int(columns[2])
        end = int(columns[3])

        if name != curr_name:
            if name in all_names:
                raise RuntimeError(f'Input alignments must be sorted by read name (see {name})')
            all_names.add(name)
            curr_name = name
            # Ceiling division
            curr_bin_size = (read_len + READ_BINS - 1) // READ_BINS

            start_bin = start // curr_bin_size
            end_bin = (end - 1) // curr_bin_size + 1
            nbits = end_bin - start_bin
            covered = MAX_U64 if nbits == 64 else np.uint64((ONE << np.uint64(nbits)) - 1) << np.uint64(start_bin)
            yield columns

        else:
            start_bin = start // curr_bin_size
            end_bin = (end - 1) // curr_bin_size + 1
            nbits = end_bin - start_bin
            mask = MAX_U64 if nbits == 64 else np.uint64((ONE << np.uint64(nbits)) - 1) << np.uint64(start_bin)
            if not (covered & mask):
                covered |= mask
                yield columns


def calculate_covs(alignment_fname, nodes, edges):
    '''This function takes the GAF file as an input, as well as the dictionary with the Node objects, in order to count the number of aligned base pairs per node.'''

    print("*** Reading alignments from {}".format(alignment_fname), file=sys.stderr)

    nodes = {k: v for k, v in nodes.items() if v.clipped_len() > 0}  #Filter out nodes where the left clipping point is bigger (or the same) than the rigth clipping one

    # We create a dict to save the number of aligned bp per node, and we also initiate a counter for the total number of aligned bp in the non-clipped area of the node
    coverages = {key: 0 for key in nodes}

    # Create list with read lengths
    read_length = []

    nr_align = 0
    a_start = perf_counter()
    # We read the GAF file, and count the aligned bp per each node of each alignment
    with open(alignment_fname,"r") as alignment_file:
        for columns in filter_gaf(alignment_file):
            nr_align += 1

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
    print("    Read {} alignments in {}s".format(nr_align,a_stop-a_start), file=sys.stderr)

    rlen_params = _fit_skewnorm(read_length)
    return coverages, rlen_params

def _fit_skewnorm(read_lens: list):
    '''Function to fit a skew normal distribution to the read lengths distribution.'''
    from scipy.stats import skewnorm
    from scipy.stats._warnings_errors import FitError

    subset_size = 100000
    arr = np.array(read_lens)
    if len(read_lens) > subset_size:
        ixs = np.random.choice(len(read_lens), subset_size, replace=False)
        arr = arr[ixs]

    if arr.var() <= 1e-6:
        rlen_params = (arr.mean(), np.nan, np.nan)
    else:
        try:
            rlen_params = skewnorm.fit(arr)
        except FitError:
            rlen_params = (arr.mean(), np.nan, np.nan)
    return rlen_params

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
    print("    Nodes binned in {}s".format(b_stop-b_start), file=sys.stderr)

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
