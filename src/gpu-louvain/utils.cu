#include "utils.cuh"
#include <vector>
#include <iostream>
#include <thrust/partition.h>
#include <fstream>
#include <getopt.h>
#include <sstream>
// #include <string.h>
#include "../louvain.h"

host_structures readInputData(char *fileName) {
	std::fstream file;
	file.open(fileName);
    int V, E;
	std::string s;
	do {
		std::getline(file, s);
	} while (s[0] == '%');
	std::istringstream stream(s);
    stream >> V >> V >> E;
	printf("Vertices: %d, Edges: %d\n", V, E);
    int v1, v2;
    float w;
    host_structures hostStructures;
	hostStructures.originalV = V;
	hostStructures.V = V;
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.vertexCommunity, V * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.communityWeight, V * sizeof(float), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.originalToCommunity, V * sizeof(int), cudaHostAllocDefault));

    std::vector<std::vector<std::pair<int, float>>> neighbours(V);
    // TODO: here is assumption that graph is undirected
    int aux = E;
    for (int i = 0; i < aux; i++) {
        file >> v1 >> v2 >> w;
        // v1--;
        // v2--;
		hostStructures.communityWeight[v1] += w;
        neighbours[v1].emplace_back(v2, w);
        if (v1 != v2) {
            E++;
			hostStructures.communityWeight[v2] += w;
            neighbours[v2].emplace_back(v1, w);
			hostStructures.M += w;
        }
		hostStructures.M += w;
    }
    hostStructures.M /= 2;
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edges, E * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.weights, E * sizeof(float), cudaHostAllocDefault));
	hostStructures.E = E;
    int index = 0;
    for (int v = 0; v < V; v++) {
		hostStructures.edgesIndex[v] = index;
        for (auto & it : neighbours[v]) {
			hostStructures.edges[index] = it.first;
			hostStructures.weights[index] = it.second;
            index++;
        }
    }
	hostStructures.edgesIndex[V] = E;
    file.close();
	printf("Graph reading done\n");
    return hostStructures;
}

host_structures convertToHostStructures(Graph& gr) {
	// std::fstream file;
	// file.open(fileName);
    int V, E;
	// std::string s;
	// do {
	// 	std::getline(file, s);
	// } while (s[0] == '%');
	// std::istringstream stream(s);
    // stream >> V >> V >> E;
	V = gr.nb_nodes;
	E = gr.nb_links;
    int v1, v2;
    float w;
    host_structures hostStructures;
	hostStructures.originalV = V;
	hostStructures.V = V;
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.vertexCommunity, V * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.communityWeight, V * sizeof(float), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.originalToCommunity, V * sizeof(int), cudaHostAllocDefault));

	cout << "Point 1G" << endl;

    // std::vector<std::vector<std::pair<int, float>>> neighbours(V);

	// for(int i = 0; i < V; i++) {
	// 	pair<vector<int>::iterator, vector<long double>::iterator> p = gr.neighbors(i);//
	// 	int deg = gr.nb_neighbors(i);
	// 	for (int i=0 ; i<deg ; i++) {
	// 		int neigh  = *(p.first+i);//ith nighbor of node

	// 		// Manul: weight of edge from node to its i-th neighbor
	// 		long double neigh_w = (gr.weights.size()==0)?1.0L:*(p.second+i);//ith neighbor comm weight

	// 		hostStructures.communityWeight[i] += neigh_w;
    //     	neighbours[i].emplace_back(neigh, neigh_w);	

	// 		hostStructures.M += neigh_w;	
	// 	}
	// }

	for(int i = 0; i < V; i++) {
		hostStructures.communityWeight[i] = gr.weighted_degree(i);
	}

    hostStructures.M = (gr.total_weight) / 2;
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.edges, E * sizeof(int), cudaHostAllocDefault));
    HANDLE_ERROR(cudaHostAlloc((void**)&hostStructures.weights, E * sizeof(float), cudaHostAllocDefault));
	hostStructures.E = E;

	cout << "Point 2G" << endl;


	assert(E == gr.links.size());
	copy(gr.links.begin(), gr.links.end(), hostStructures.edges);
	if(gr.weights.empty()) {
		fill(hostStructures.weights, hostStructures.weights + hostStructures.E, 1.0f);
	} else {
		copy(gr.weights.begin(), gr.weights.end(), hostStructures.weights);
	}
	*(hostStructures.edgesIndex) = 0;
	copy(gr.degrees.begin(), gr.degrees.end(), hostStructures.edgesIndex + 1);

	cout << "Point 3G" << endl;

    // int index = 0;
    // for (int v = 0; v < V; v++) {
	// 	hostStructures.edgesIndex[v] = index;
    //     for (auto & it : neighbours[v]) {
	// 		hostStructures.edges[index] = it.first;
	// 		hostStructures.weights[index] = it.second;
    //         index++;
    //     }
    // }
	// hostStructures.edgesIndex[V] = E;
    // file.close();
    return hostStructures;
}

void init_partition(host_structures& hostStructures, int* comm_size, char *filename) {
	std::fstream file;
	file.open(filename);

	float* vertexEdgeWeights = (float*) malloc(hostStructures.V*sizeof(float));
	for(int j = 0; j < hostStructures.V; j++) {
		vertexEdgeWeights[j] = hostStructures.communityWeight[j];
		hostStructures.communityWeight[j] = 0;
	}

	cout << "Point 4G" << endl;


	for (int i = 0; i < hostStructures.V; i++)
	{
		int node, comm; file >> node >> comm;
		
		// copy(hostStructures.communityWeight, hostStructures.communityWeight + hostStructures.V, vertexEdgeWeights);
		// fill(hostStructures.communityWeight, hostStructures.communityWeight + hostStructures.V, 0);
		hostStructures.vertexCommunity[node] = comm;
		hostStructures.communityWeight[comm] += vertexEdgeWeights[node];
		hostStructures.originalToCommunity[node] = comm;
		comm_size[comm] += 1;
	}
	free(vertexEdgeWeights);
	printf("Init partition done\n");
}

void init_partition(host_structures& hostStructures, int* comm_size, std::vector<int>& n2c) {
	cout << "INIT PART starts" << endl;
	
	assert(n2c.size() == hostStructures.V);

	float* vertexEdgeWeights = (float*) malloc(hostStructures.V*sizeof(float));

	cout << "After malloc" << endl;
	
	for(int j = 0; j < hostStructures.V; j++) {
		vertexEdgeWeights[j] = hostStructures.communityWeight[j];
		hostStructures.communityWeight[j] = 0;
	}

	cout << "Point 1INIT" << endl;

	
	for (int i = 0; i < hostStructures.V; i++)
	{
		int node = i, comm = n2c[i];
		
		// copy(hostStructures.communityWeight, hostStructures.communityWeight + hostStructures.V, vertexEdgeWeights);
		// fill(hostStructures.communityWeight, hostStructures.communityWeight + hostStructures.V, 0);
		hostStructures.vertexCommunity[node] = comm;
		hostStructures.communityWeight[comm] += vertexEdgeWeights[node];
		// hostStructures.originalToCommunity[node] = comm;		// Manul: TODO: Not needed
		comm_size[comm] += 1;
	}

	free(vertexEdgeWeights);
	printf("Init partition done\n");
}

void copyStructures(host_structures& hostStructures, device_structures& deviceStructures,
					aggregation_phase_structures& aggregationPhaseStructures) {
	// copying from deviceStructures to hostStructures
	int V = hostStructures.V, E = hostStructures.E;
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communityWeight, V * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.weights, E * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edgesIndex, (V + 1) * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalToCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexEdgesSum, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.newVertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.V, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.E, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalV, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communitySize, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.partition, V * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.toOwnCommunity, V * sizeof(int)));


	thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, 1);
	thrust::sequence(thrust::device, deviceStructures.vertexCommunity, deviceStructures.vertexCommunity + V, 0);
	thrust::sequence(thrust::device, deviceStructures.newVertexCommunity, deviceStructures.newVertexCommunity + V, 0);
	thrust::sequence(thrust::device, deviceStructures.originalToCommunity, deviceStructures.originalToCommunity + V, 0);

	HANDLE_ERROR(cudaMemcpy(deviceStructures.communityWeight, hostStructures.communityWeight, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edges, hostStructures.edges, E * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.weights, hostStructures.weights, E * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edgesIndex, hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &hostStructures.V, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.E, &hostStructures.E, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.originalV, &hostStructures.originalV, sizeof(int), cudaMemcpyHostToDevice));

	// preparing aggregationPhaseStructures
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.communityDegree, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newID, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgePos, V * sizeof(int)));;
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.vertexStart, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.orderedVertices, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgeIndexToCurPos, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newEdges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newWeights, E * sizeof(float)));
}

void copyStructuresWithInitPartition(host_structures& hostStructures, device_structures& deviceStructures,
					aggregation_phase_structures& aggregationPhaseStructures, int* comm_size) {
	// copying from deviceStructures to hostStructures
	int V = hostStructures.V, E = hostStructures.E;
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communityWeight, V * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.weights, E * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edgesIndex, (V + 1) * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalToCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexEdgesSum, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.newVertexCommunity, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.V, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.E, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalV, sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communitySize, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.partition, V * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.toOwnCommunity, V * sizeof(int)));


	// thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, 1);
	// thrust::sequence(thrust::device, deviceStructures.vertexCommunity, deviceStructures.vertexCommunity + V, 0);
	// thrust::sequence(thrust::device, deviceStructures.newVertexCommunity, deviceStructures.newVertexCommunity + V, 0);
	// thrust::sequence(thrust::device, deviceStructures.originalToCommunity, deviceStructures.originalToCommunity + V, 0);

	printf("Copy with init...\n");

	HANDLE_ERROR(cudaMemcpy(deviceStructures.communitySize, comm_size, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.vertexCommunity, hostStructures.vertexCommunity, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.newVertexCommunity, hostStructures.vertexCommunity, V * sizeof(float), cudaMemcpyHostToDevice));
	// HANDLE_ERROR(cudaMemcpy(deviceStructures.originalToCommunity, hostStructures.vertexCommunity, V * sizeof(float), cudaMemcpyHostToDevice));
	thrust::sequence(thrust::device, deviceStructures.originalToCommunity, deviceStructures.originalToCommunity + V, 0);

	HANDLE_ERROR(cudaMemcpy(deviceStructures.communityWeight, hostStructures.communityWeight, V * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edges, hostStructures.edges, E * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.weights, hostStructures.weights, E * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.edgesIndex, hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &hostStructures.V, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.E, &hostStructures.E, sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(deviceStructures.originalV, &hostStructures.originalV, sizeof(int), cudaMemcpyHostToDevice));

	// preparing aggregationPhaseStructures
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.communityDegree, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newID, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgePos, V * sizeof(int)));;
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.vertexStart, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.orderedVertices, V * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgeIndexToCurPos, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newEdges, E * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newWeights, E * sizeof(float)));
}

void deleteStructures(host_structures& hostStructures, device_structures& deviceStructures,
					  aggregation_phase_structures& aggregationPhaseStructures) {
    HANDLE_ERROR(cudaFreeHost(hostStructures.vertexCommunity));
    HANDLE_ERROR(cudaFreeHost(hostStructures.communityWeight));
    HANDLE_ERROR(cudaFreeHost(hostStructures.edges));
    HANDLE_ERROR(cudaFreeHost(hostStructures.weights));
    HANDLE_ERROR(cudaFreeHost(hostStructures.edgesIndex));
    HANDLE_ERROR(cudaFreeHost(hostStructures.originalToCommunity));


	HANDLE_ERROR(cudaFree(deviceStructures.originalV));
    HANDLE_ERROR(cudaFree(deviceStructures.vertexCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.communityWeight));
	HANDLE_ERROR(cudaFree(deviceStructures.edges));
	HANDLE_ERROR(cudaFree(deviceStructures.weights));
	HANDLE_ERROR(cudaFree(deviceStructures.edgesIndex));
	HANDLE_ERROR(cudaFree(deviceStructures.originalToCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.vertexEdgesSum));
	HANDLE_ERROR(cudaFree(deviceStructures.newVertexCommunity));
	HANDLE_ERROR(cudaFree(deviceStructures.E));
	HANDLE_ERROR(cudaFree(deviceStructures.V));
	HANDLE_ERROR(cudaFree(deviceStructures.communitySize));
	HANDLE_ERROR(cudaFree(deviceStructures.partition));
    HANDLE_ERROR(cudaFree(deviceStructures.toOwnCommunity));

	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.communityDegree));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newID));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.edgePos));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.vertexStart));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.orderedVertices));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.edgeIndexToCurPos));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newEdges));
	HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newWeights));
}

int blocksNumber(int V, int threadsPerVertex) {
	return (V * threadsPerVertex + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

bool isPrime(int n) {
	for (int i = 2; i < sqrt(n) + 1; i++)
		if (n % i == 0)
			return false;
	return true;
}

int getPrime(int n) {
	do {
		n++;
	} while(!isPrime(n));
	return n;
}

void parseCommandLineArgs(int argc, char *argv[], float *minGain, bool *isVerbose, char **fileName, char **initCommFileName, char **nodeEvalSetFileName) {
	bool isF, isG;
	char opt;
	while ((opt = getopt(argc, argv, "f:g:p:e:v")) != -1) {
		switch (opt) {
			case 'g':
				isG = true;
				*minGain = strtof(optarg, NULL);
				break;
			case 'v':
				*isVerbose = true;
				break;
			case 'f':
				isF = true;
				*fileName = optarg;
				break;
			case 'p':
				*initCommFileName = optarg;
				break;
			case 'e':
				*nodeEvalSetFileName = optarg;
				break;
			default:
				printf("Usage: ./gpulouvain -f mtx-matrix-file -g min-gain [-v]\n");
				exit(1);
		}
	}
	if (!isF || !isG) {
		printf("Usage: ./gpulouvain -f mtx-matrix-file -g min-gain [-v]\n");
		exit(1);
	}
}