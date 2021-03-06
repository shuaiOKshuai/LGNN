# LGNN

# Node-wise Localization of Graph Neural Networks

We present the datasets and code for our paper "[Node-wise Localization of Graph Neural Networks](https://zemin-liu.github.io/papers/LGNN-IJCAI-21.pdf)" (LGNN for short), which is published in IJCAI-2021.


## 1. Description for each file

	Folders:
		- model : the models, including LGCN (based on GCN), LGAT (based on GAT) and LGIN (based on GIN).
		- dataset : the datasets
	
	Inside files in each LGNN folder:
		- paramsConfigPython : parameters setting file
		- mainEntry.py : the main entry of the model
		- modelTraining.py : the training file 
		- gnnModel.py : the LGNN model
		- processTools.py : file that contains some tool functions
	
	LGNN_IJCAI_21_Appendix.pdf: the appendix file of the paper
	
## 2. Requirements (Environment)

	python-3.6.5
	tensorflow-1.13.1

## 3. How to run

- (1) First configure the parameters in file "paramsConfigPython";
- (2) Run "python3 mainEntry.py".
	
	Remark : Command "python3 mainEntry.py" would call the train, validation and test together, and finally output the prediction results on the test. There is no need to pretrain the model.

## 4. Datasets

The datasets could be downloaded from the link in the paper, and we also include the datasets in folder "dataset".
And you can also prepare your own datasets. The data format is as follows,
- (1) two files should be prepared, graph.node to describe the nodes and graph.edge to describe the edges;
- (2) node file ( graph.node )
	- The first row is the number of nodes + tab + the number of features
	- In the following rows, each row represents a node: the first column is the node_id, the second column is the label_id of current node, and the third to the last columns are the features of this node. All these columns should be split by tabs.
- (3) edge file ( graph.edge )
	- Each row is a directed edge, for example : 'a tab b' means a->b. We can add another line 'b tab a' to represent this is a bidirection edge. All these values should be split by tabs.

Note that, for your own dataset, when running the code the dataset would be split into train, validation and test automatically.


## 5. Cite
	
	@inproceedings{liu2021nodewise,
	  title     = {Node-wise Localization of Graph Neural Networks},
	  author    = {Liu, Zemin and Fang, Yuan and Liu, Chenghao and Hoi, Steven C.H.},
	  booktitle = {Proceedings of the Thirtieth International Joint Conference on
		       Artificial Intelligence, {IJCAI-21}},
	  publisher = {International Joint Conferences on Artificial Intelligence Organization},
	  pages     = {1520--1526},
	  year      = {2021},
	  note      = {Main Track}
	}

			
