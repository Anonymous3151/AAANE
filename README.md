# AAANE: Attention-based Adversarial Autoencoder for Multi-scale Network Embedding 
Codes and Dataset for IJCAI2018 submitted paper ‘‘AAAANE: Attention-based Adversarial Autoencoder for Multi-scale Network Embedding’’. 

## Requirements

-  keras==1.2.1
-  theano==1.0
-  networkx==2.0
- numpy==1.13.3
-  scipy==0.19.1
-  gensim==3.2.0
-  scikit-learn==0.19.0




## Data
You can find the pre-processed datasets in /data folder

[Wiki][1]  : 2405 nodes, 17981 edges, 19 labels, directed:
- data/wiki/Wiki_edgelist.txt
- data/wiki/Wiki_category.txt

[Cora][2]: 2708 nodes, 5429 edges, 7 labels, directed:
- data/cora/cora_edgelist.txt
- data/cora/cora_labels.txt

[Citeseer][3]: 3312 nodes, 4732 edges, 6 labels
- data/citeseer/graph.txt
- data/citeseer/group.txt


## Usage

#### General Options

You can check out the other options available  using:

python src/main.py --help

- --input, the input file of a network;
- --graph-format, the format of input graph, adjlist or edgelist;
- --output, the output file of representation;
- --representation-size, the number of latent dimensions to learn for each node; the default is 128
- --label-file, the file of node label; ignore this option if not testing;
- --clf-ratio, the ratio of training data for node classification; the default is 0.5;
- --epochs, the training epochs; the default is 5;
- --kstep, the maximum matrix transition scale ; the default is 8;
	 
#### Example
To run “AAANE” on Cora network and evaluate the learned representations on multi-label node classification task, run the following command in the home directory of this project:

	THEANO_FLAGS="device=gpu0,floatX=float32" python src/main.py	 --label-file ../data/cora/cora_labels.txt --input ../data/cora/cora_edgelist.txt --graph-format edgelist --output vec_all.txt


#### Input
The supported input format is an edgelist or an adjlist:

	edgelist: node1 node2 <weight_float, optional>
	adjlist: node n1 n2 n3 ... nk
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* nodes. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation.


#### Evaluation

If you want to evaluate the learned node representations, you can input the node labels. It will use a portion (default: 50%) of nodes to train a classifier and test  on the rest dataset.

The supported input label format is

	node label1 label2 label3...



[1]:	https://github.com/thunlp/MMDW/tree/master/data
[2]:	https://linqs.soe.ucsc.edu/data
[3]:	http://www.cs.umd.edu/%5C~sen/lbc-proj/LBC.html
