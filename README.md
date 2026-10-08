# Floco

Floco (_flow_+_copy_) is a tool to call individual node copy number (CN) on (pan)genome graphs, using sequence-to-graph alignment information.

## Table of contents

- [Installation](#installation)
  + [Requirements](#requirements)
  + [Bioconda](#bioconda)
  + [Manual](#manual)
- [Usage](#usage)
  + [Input](#input)
  + [Output](#output)
  + [Command-line options](#command-line-options)
- [Test Dataset](#test-dataset)
- [Citation](#citation)
- [Example applications](#example-applications)


## Installation

### Requirements

To run Floco, you need a [Gurobi License](https://www.gurobi.com/solutions/licensing/) for solving the ILP problem. Please install [Gurobi](https://anaconda.org/channels/Gurobi/packages/gurobi/overview), using a version compatible with your license (here, tested using version 12.0.3, older versions might be slower).

Additionally, you need:

- numpy>=2.3.2
- scikit-learn>=1.7.1
- scipy>=1.16.1

### Bioconda

To install Floco with conda, run:

```bash
conda install -c bioconda floco
```

### Manual

To manually install Floco, just run the following:

```bash
git clone https://github.com/hugocarmaga/floco.git
cd floco
python -m pip install .
# or (if you don't want to install any dependencies)
python -m pip install . --no-deps
```


## Usage

### Input

To use Floco, you need two input files:
- GFA graph: file to have the CN estimated for. It **needs** to have **sequence** and to be **sorted** (i.e., all nodes must be before any edge line)
- GAF alignments: sequence-to-graph alignments to compute coverage and subsequent CN probabilities from. We recommend using [GraphAligner](https://github.com/maickrau/GraphAligner).

### Output

By default, Floco outputs one file, required argument `-o OUTPUT`, containg node name, length, sum of bp coverage across the node and the predicted CN value.

Additionally, when using the `--debug` option, Floco will produce further additional files:
- `stats_concordance-{filename}.csv`: File containing further statistics on nodes, namely the CN value with the highest probability before being fed into the network flow.
- `ilp_results-{filename}.csv`: File containing all ILP variables results.
- `model_{filename}.lp`: File containing the ILP model definition.
- `dump-{filename}.tmp.pkl`: pickle dump with all variables and parameters before ILP. It's especially helpful when running Floco multiple times on the same pair of GFA+GAF. It can then be given as an input parameter instead of the GAF file, resulting in Floco starting directly from the ILP solving.

### Command-line options

```bash
$ floco -h
usage: floco -g <graph.gfa> (-a <alignments.gaf> | -d <pickle.pkl>) -o <output.csv> [options]

floco: Flow-based copy number estimation for genome graphs.

options:
  -g FILE, --graph FILE
                        The GFA file with the graph.
  -a FILE, --alignment FILE
                        The GAF file with sequence-to-graph alignments. Cannot be used with '--pickle'.
  -o FILE, --output FILE
                        The name for the output csv file with the copy numbers.
  -p BG_PLOIDY [BG_PLOIDY ...], --bg-ploidy BG_PLOIDY [BG_PLOIDY ...]
                        Expected most common CN value in the graph (background ploidy of the dataset). (default:[1, 2])
  -l FILE, --locityper-bg FILE
                        Locityper preprocessing data for this sample. Can be used to supplement the parameter estimation step.
  -S EXPEN_PEN, --expen-pen EXPEN_PEN
                        Probability for using the super edges when there are other edges available. (default:-10000)
  -s CHEAP_PEN, --cheap-pen CHEAP_PEN
                        Probability for using the super edges when there is no other edge available. (default:-25)
  -e EPSILON, --epsilon EPSILON
                        Epsilon value for adjusting CN0 counts to probabilities (default:0.02)
  -b BIN_SIZE, --bin-size BIN_SIZE
                        Set the bin size to use for the NB parameters estimation. (default:100)
  -c COMPLEXITY, --complexity COMPLEXITY
                        Model complexity (1-3): larger = slower and more accurate. (default: 2)
  -d FILE, --pickle FILE
                        Pickle dump with the data (cannot be used with '--alignment'). Dump file can be produced with '--debug'.
  -t THREADS, --threads THREADS
                        Number of computing threads to use by the ILP solver.
  --fix-cn FILE         Fix copy number for the given nodes (two column file with nodes and CN).
  --prior-ploidy PRIOR_PLOIDY
                        Estimate priors from GFA paths from samples with this ploidy [2]. Format: one or more numbers concatenated via comma.Path names must start with the sample name
                        followed by dot (.) or hashtag (#).
  --prior-weight PRIOR_WEIGHT
                        Give this weight to CN priors [0.02]. Use 0 to disable.
  --debug               Produce additional files.
  -h, --help            Show this help message and exit.
  -V, --version         Show program's version number and exit.
```

## Test Dataset

The test dataset can be found [here](https://zenodo.org/records/23243203). It includes a graph for chromosome 6 from HG01114 and HiFi reads aligned to it with [GraphAligner](https://github.com/maickrau/GraphAligner).

To get copy number values for the test data, simply run:
```bash
floco -g HG01114-chr6.gfa.gz -a hifi_HG01114-chr6_ga.gaf.gz -o test_dataset_copy-numbers.csv -p 1
```

The output file should look like this:
```
Node,Length,Sum_coverage,Copy_number
utig4-9,11900559,210743696,1
utig4-99,111,666,1
utig4-98,138062,5047482,2
utig4-8,2333535,39779042,1
utig4-7,25009,856214,2
utig4-734,22256,636834,1
utig4-733,23057,358577,0
utig4-731,16975505,295336678,1
utig4-730,2829470,49583270,1
```

## Citation

To cite Floco, please use:

> Magalhães, H., Weber, J., Klau, G. W., Marschall, T., & Prodanov, T. (2025)
> Sequence-to-graph alignment based copy number calling using a network flow formulation.
> bioRxiv, 2025-11.
> https://doi.org/10.1101/2025.11.21.689771

## Example applications

<p align="center"><img src=".examples/misassembly.png"/></p>

<p align="center"><img src=".examples/overview_1e.png"/></p>
