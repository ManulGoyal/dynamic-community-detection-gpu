#ifndef __NODE_EVAL__CUH__
#define __NODE_EVAL__CUH__
#include "utils.cuh"
#include <vector>

// extern __constant__ float M;

// const int bucketsSize = 8;
// const int buckets[] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};
// const int primes[] = {7, 13, 29, 53, 127, 479};
// // x - number of neighbours processed concurrently, y - vertices per block
// const dim3 dims[] {
// 		{4, 32},
// 		{8, 16},
// 		{16, 8},
// 		{32, 4},
// 		{32, 4},
// 		{128, 1},
// 		{128, 1},
// };

// /**
//  * Function responsible for executing 1 phase (modularity optimisation)
//  * @param minGain          minimum gain for going to next iteration of this phase
//  * @param deviceStructures structures kept in device memory
//  * @param hostStructures   structures kept in host memory
//  * @return information whether any changes were applied
//  */
// bool optimiseModularity(float minGain, device_structures& deviceStructures, host_structures& hostStructures);

// bool optimiseModularityUsingVertexSubset(float minGain, device_structures& deviceStructures, host_structures& hostStructures, std::vector<int>& nodeEval);

// float calculateModularity(int V, float M, device_structures deviceStructures);

// void printOriginalToCommunity(device_structures& deviceStructures, host_structures& hostStructures);

// void initM(host_structures& hostStructures);

int nodeEval_add_gpu(device_structures& deviceStructures, host_structures& hostStructures, std::vector<pair<unsigned int, unsigned int>>& newEdges);

int nodeEval_del_gpu(device_structures& deviceStructures, host_structures& hostStructures, std::vector<pair<unsigned int, unsigned int>>& newEdges);

#endif /* __NODE_EVAL__CUH__ */