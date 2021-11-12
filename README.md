# README #


### What is this repository for? ###

* This project implement Delta-screening method and integrate it with static Louvain algorithm. The framework can be used to detect community structures within dynamic networks. This framework is built in C++ and can be applied to other static modularity-based algorithm to track the communities in evolving networks
* The code can be run with  gcc>=5.0, c++11. The code is multi-platform and compiles on Linux, Mac OSX and Windows.

### Citation information: ###

N. Zarayeneh, A. Kalyanaraman.
A fast and efficient incremental approach toward dynamic community detection. 
Proc. 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM'19), pp. 9-16, 2019.
DOI: 10.1145/3341161.3342877

### How do I get set up? ###

* Download the respository
* cd dynamic_community_detection
*  **Preprocess data**
    1. ***Addition and deletion chunks:*** Split the change files for each time step and get two text files for addition and deletion (name them consistently: name_of_addition0.txt, name_of_deletion0.txt, name_of_addition1.txt,name_of_deletion1, ...)
    2. ***Create simple inputs:*** The input graph and changes are assumed to be simple, so you can use the to make all your inputs simple
*  **Steps to detect communities at each time step** 
    1. ***Input conversion:*** convert input graph format (a text in which each line contains a pair "source destination") to Binary format  ``` ./convert -i graph.txt -o graph.bin ``` This project can also be used to convert weighted graphs (a text in which each line contains a triple "source destination weight") using -w option: ```./convert -i graph.txt -o graph.bin -w graph.weights``` Moreover, to save space in some cases, nodes can be renumbered from 0 to nb_nodes-1 using -r option: ```./convert -i graph.txt -o graph.bin -r labelings_connection_file.txt```  
    2. ***Compute communities:*** specify the quality function and display hierarchical tree ```./louvain graph.bin -l -1 -v -q id_qual -x delta_add -a 1 -b delta_del``` To ensure a faster computation (with a loss of quality), one can use -e option to stop furthur computation to get higher modularity when the increas in modularity is below epsilon for a given iteration: ```./louvain graph.bin -l -1 -q id_qual  -x delta_add -a 1 -b delta_del -e 0.001```  If thenetwork is weighted we use -w option: ```./louvain graph.bin -l -1 -q id_qual  -x delta_add -a 1 -b delta_del -w graph.weights``` . For the initial step, the algorithm can also start with any given partition using -p option ```./louvain graph.bin -q id_qual  -x delta_add -a 1 -b delta_del -p graph.part -v ``` 
    3. ***Display information on the tree structure:*** it shows the number of hierarchical levels and nodes per level ```./hierarchy graphi.tree``` (like garph0.tree for time step 0, graph1.tree for time step 1,...) We can specify to display the nodes within each community for a given level of the tree: ```./hierarchy graphi.tree -l 2 > graph_node2comm_level2``` or display the nodes within each community for the last level of the tree: ```./hierarchy graphi.tree -m > graph_node2comm_lastlevel``` 
 
### Run the code with the provided examples ###
*  **Toy example**
    1. ```./convert -i Example.txt -o graph.bin ```
    2. ```./louvain graph.bin -l -1 -v -q id_qual -x delta_add -a 1 -b delta_del``` 
*  **Citation example** 
    1. We used the citation graph (Arxiv  HEP-TH) for this example. This dataset covers papers published between 1993 and 2002.Consequently,  we  partitioned  this  period  into  10  timesteps (one for each year).
    2. In the Preprocess.R code, set the directory path to the folder containing the cittion example that you have downloaded and run the code
    3. ```./convert -i cite.txt -o graph.bin``` 
    4. ```./louvain graph.bin -l -1 -v -q id_qual -x cite_add -a 1 -b cite_del``` 