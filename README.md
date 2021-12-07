# Streaming Community Detection in Parallel #

B.Tech. Project Semester VII

Manul Goyal (B18CSE031) and Manan Shah (B18CSE030)

Mentor: Dr. Dip Sankar Banerjee

This is the repository for the BTP "Streaming Community Detection in Parallel", which implements the parallel version of delta-screening (), written in CUDA. The delta-screening method is an algorithm for determining a subset of nodes likely to be affected by addition/deletion of edges at a given time step, and therefore, should be re-evaluated by the static community detection algorithm (Louvain). This results in potentially lesser nodes to process at each time step, leading to faster runtime. The parallelization involved:

1. Replacing the sequential Louvain algorithm with a parallel version, whose implementation is taken from (). We had to integrate this code (referred to as GPU Louvain from now on) to fit in the workflow of delta-screening, which involved making the following changes:
    - Adding a parameter for the set of nodes to be re-evaluated in the modularity optimization phase of GPU Louvain, and finding the locally gain-maximizing communities for nodes in this set only
    - Initializing the communities for each node at a given time-step based on the communities obtained in the last time-step (since delta-screening is an incremental algorithm, it starts with the current state of the communities and modifies it for the next time step, to prevent recomputations)
    - When modifying the graph at any time-step, allowing the iteration over modularity optimization phase and community aggregation phase to run at least twice, even if the first iteration does not yield positive modularity gain. This is because the first iteration can potentially agglomerate communities (which were computed in the last time-step) and there may be scope of improvement in the new contracted graph
2. Implementing parallel version of the dynamic portions of delta-screening, i.e., the computation of the set of nodes to be re-evaluated at every time step. This implementation was completely done by us, and the parallelization strategy is inspired by GPU Louvain

### Instructions for Running ###

In the following steps, we show how to run the sequential as well as parallel version of delta-screening on the ToyExample provided

1. Clone the repository and navigate to the repository directory. Then compile the code using make: \
```cd [REPO_FOLDER]``` \
```make```

Make sure you have CUDA 10.2 or later installed along with GNU C++ compiler, along with a CUDA-capable GPU.

2. Navigate to the ToyExample folder, and  create directories for cpu and gpu: \
```cd ToyExample``` \
```mkdir cpu gpu```

3. Preprocessing the graph: \
```../convert -i Example.txt -o graph.bin```

4. Go to the CPU folder: \
```cd cpu```

5. Run the CPU version of delta-screening on the Toy Example: \
```../../cpulouvain ../graph.bin -l -1 -v -b ../delta_del -a 1 -x ../delta_add```

    It will output the progress as well as the modularity and runtime at each time step.

6. Go to the gpu folder: \
```cd ../gpu```

7. Run the GPU version of delta-screening on the Toy Example: \
```../../gpulouvain_full_mem ../graph.bin -l -1 -v -b ../delta_del -a 1 -x ../delta_add```

    It will output the progress as well as the modularity and runtime at each time step.

At both steps 5 and 7, you can use the -e option to stop furthur computation to get higher modularity when the increase in modularity is below epsilon for a given iteration.



