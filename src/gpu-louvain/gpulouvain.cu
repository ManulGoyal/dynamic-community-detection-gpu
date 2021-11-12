#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"
#include "../louvain.h"

void gpuLouvain(Louvain* c, std::vector<int>& n2c, bool initPart = false) {
	// std::cout << "hostStructures init started" << endl;
	auto hostStructures = convertToHostStructures(c->qual->g);

	cout << "convert to host structures done" << endl;
	// for(int i = 0; i < hostStructures.E; i++) printf("%d ", hostStructures.edges[i]);
	// printf("\n"); 
	// for(int i = 0; i < hostStructures.V; i++) printf("%d ", hostStructures.communityWeight[i]);
	// printf("\n"); 
	// for(int i = 0; i < hostStructures.V; i++) printf("%d ", hostStructures.edgesIndex[i]);
	// printf("\n"); 
	// for(int i = 0; i < hostStructures.E; i++) printf("%d ", hostStructures.weights[i]);
	// printf("\n"); 
	int* comm_size;

	cout << "before init partition" << endl;

	if(initPart) {
		comm_size = (int*) malloc(hostStructures.V * sizeof(int));
		init_partition(hostStructures, comm_size, n2c);
	}

	cout << "after init partition" << endl;

	std::vector<int> nodeEval;
	assert(c->qual->size == hostStructures.V);
	// if(c->qual->R.size() == qual->size) {
		
	// } else {
	// 	for(int i = 0; i < hostStructures.originalV; i++) {
	// 		nodeEval.push_back(i);
	// 	}
	// }


	// printf("Node eval set: ");
	// for (auto x : c->qual->R) printf("%d ", x);
	// printf("\n");
	
	// cout << "hostStructures init done" << endl;

    device_structures deviceStructures;
    aggregation_phase_structures aggregationPhaseStructures;

    cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	cout << "Before copy" << endl;

	if(initPart) copyStructuresWithInitPartition(hostStructures, deviceStructures, aggregationPhaseStructures, comm_size);
	else copyStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
	initM(hostStructures);

	
	if(initPart) free(comm_size);
	cout << "After copy" << endl;


	// cout << "Copy done" << endl;

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float memoryTime;
	HANDLE_ERROR(cudaEventElapsedTime(&memoryTime, start, stop));

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	///
	bool onceMore = initPart;
	for (int level = 0;; level++) {
		cout << "level " << level << endl;
		if(level == 0 && (c->qual->R.size()) != (c->qual->size)) {
			if(!optimiseModularityUsingVertexSubset((float)(c->eps_impr), deviceStructures, hostStructures, c->qual->R)) {
				if(!onceMore) break;
			}
		} else {
			if (!optimiseModularity((float)(c->eps_impr), deviceStructures, hostStructures)) {
				if(!onceMore) break;
			}
		}
		// cout << "optimize mod" << endl;
		onceMore = false;
		
		aggregateCommunities(deviceStructures, hostStructures, aggregationPhaseStructures);
		// printf("After aggregation:\n");
		// for(int i = 0; i < hostStructures.V; i++) {
		// 	printf("%d ", hostStructures.vertexCommunity[i]);
		// }
		// printf("\n");
	}
	///
	int V;
	HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
	printf("Final Modularity: %f\n", calculateModularity(V, hostStructures.M, deviceStructures));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float algorithmTime;
	HANDLE_ERROR(cudaEventElapsedTime(&algorithmTime, start, stop));
	printf("Time: %f %f\n", algorithmTime, algorithmTime + memoryTime);
	// if (isVerbose)
	// printOriginalToCommunity(deviceStructures, hostStructures);
	HANDLE_ERROR(cudaMemcpy(hostStructures.originalToCommunity, deviceStructures.originalToCommunity,
			hostStructures.originalV * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < hostStructures.originalV; i++) {
		n2c[i] = hostStructures.originalToCommunity[i];
	}

	deleteStructures(hostStructures, deviceStructures, aggregationPhaseStructures);

}

// int main(int argc, char *argv[]) {
// 	char *fileName, *initCommFileName = NULL, *nodeEvalSetFileName = NULL;
// 	float minGain;
// 	bool isVerbose;
// 	parseCommandLineArgs(argc, argv, &minGain, &isVerbose, &fileName, &initCommFileName, &nodeEvalSetFileName);
// 	if(initCommFileName) printf("%s\n", initCommFileName);

//     auto hostStructures = readInputData(fileName);
// 	int* comm_size = (int*) malloc(hostStructures.V * sizeof(int));
// 	if(initCommFileName) init_partition(hostStructures, comm_size, initCommFileName);

// 	std::vector<int> nodeEval;
// 	if(nodeEvalSetFileName) {
// 		std::ifstream fin(nodeEvalSetFileName);
// 		int nodeId;
// 		while(fin >> nodeId) {
// 			nodeEval.push_back(nodeId);
// 		}
// 	} else {
// 		for(int i = 0; i < hostStructures.originalV; i++) {
// 			nodeEval.push_back(i);
// 		}
// 	}

// 	printf("Node eval set: ");
// 	for (auto x : nodeEval) printf("%d ", x);
// 	printf("\n");
	

//     device_structures deviceStructures;
//     aggregation_phase_structures aggregationPhaseStructures;

//     cudaEvent_t start, stop;
// 	HANDLE_ERROR(cudaEventCreate(&start));
// 	HANDLE_ERROR(cudaEventCreate(&stop));
// 	HANDLE_ERROR(cudaEventRecord(start, 0));
// 	if(initCommFileName) copyStructuresWithInitPartition(hostStructures, deviceStructures, aggregationPhaseStructures, comm_size);
// 	else copyStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
// 	initM(hostStructures);
// 	HANDLE_ERROR(cudaEventRecord(stop, 0));
// 	HANDLE_ERROR(cudaEventSynchronize(stop));
// 	float memoryTime;
// 	HANDLE_ERROR(cudaEventElapsedTime(&memoryTime, start, stop));

// 	HANDLE_ERROR(cudaEventCreate(&start));
// 	HANDLE_ERROR(cudaEventCreate(&stop));
// 	HANDLE_ERROR(cudaEventRecord(start, 0));

// 	bool onceMore = (initCommFileName != NULL);
// 	for (int level = 0;; level++) {
// 		if(level == 0 && nodeEvalSetFileName) {
// 			if(!optimiseModularityUsingVertexSubset(minGain, deviceStructures, hostStructures, nodeEval)) {
// 				if(!onceMore) break;
// 			}
// 		} else {
// 			if (!optimiseModularity(minGain, deviceStructures, hostStructures)) {
// 				if(!onceMore) break;
// 			}
// 		}
// 		onceMore = false;
// 		aggregateCommunities(deviceStructures, hostStructures, aggregationPhaseStructures);
// 		printf("After aggregation:\n");
// 		for(int i = 0; i < hostStructures.V; i++) {
// 			printf("%d ", hostStructures.vertexCommunity[i]);
// 		}
// 		printf("\n");
// 	}
// 	int V;
// 	HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
// 	printf("Final Modularity: %f\n", calculateModularity(V, hostStructures.M, deviceStructures));
// 	HANDLE_ERROR(cudaEventRecord(stop, 0));
// 	HANDLE_ERROR(cudaEventSynchronize(stop));
// 	float algorithmTime;
// 	HANDLE_ERROR(cudaEventElapsedTime(&algorithmTime, start, stop));
// 	printf("Time: %f %f\n", algorithmTime, algorithmTime + memoryTime);
// 	if (isVerbose)
// 		printOriginalToCommunity(deviceStructures, hostStructures);
// 	deleteStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
// }
