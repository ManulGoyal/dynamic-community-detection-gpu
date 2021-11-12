#include "community_aggregation.cuh"
#include <thrust/scan.h>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

/**
 * Computes hash (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHashAggregation(int val, int index, int prime) {
	int h1 = val % prime;
	int h2 = 1 + (val % (prime - 1));
	return (h1 + index * h2) % prime;
}

/**
 * Fills content of hashCommunity and hashWeights arrays that are later used in mergeCommunity function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight		   neighbour's weight
 * @param hashWeight	   table of sum of weights between vertices and communities
 * @param hashCommunity	   table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 * @return curPos, if this was first addition, -1 otherwise
 */
__device__ int prepareHashArraysAggregation(int community, int prime, float weight, float *hashWeight, int *hashCommunity,
		int hashTablesOffset) {
	int it = 0;
	while (true) {
		int curPos = hashTablesOffset + getHashAggregation(community, it++, prime);
		if (hashCommunity[curPos] == community) {
			atomicAdd(&hashWeight[curPos], weight);
			return -1;
		} else if (hashCommunity[curPos] == -1) {
			if (atomicCAS(&hashCommunity[curPos], -1, community) == -1) {
				atomicAdd(&hashWeight[curPos], weight);
				return curPos;
			} else if (hashCommunity[curPos] == community) {
				atomicAdd(&hashWeight[curPos], weight);
				return -1;
			}
		}
	}
}

// Manul: computes community sizes and degrees for each community, and marks these comm's
__global__ void fillArrays(int V, int *communitySize, int *communityDegree, int *newID, int *vertexCommunity, int *edgesIndex) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < V) {
		int community = vertexCommunity[vertex];
		atomicAdd(&communitySize[community], 1);
		int vertexDegree = edgesIndex[vertex + 1] - edgesIndex[vertex];
		atomicAdd(&communityDegree[community], vertexDegree);
		newID[community] = 1;
	}
}

/**
 * orderVertices is responsible for generating ordered (meaning vertices in the same community are placed
 * next to each other) vertices.
 * @param V               - number of vertices
 * @param orderedVertices - ordered vertices
 * @param vertexStart     - community -> begin index in orderedVertices array
 *                          NOTE: atomicAdd changes values in this array, that's why it has to be reset afterwards
 * @param vertexCommunity - vertex -> community
 */
__global__ void orderVertices(int V, int *orderedVertices, int *vertexStart, int *vertexCommunity) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < V) {
		int community = vertexCommunity[vertex];
		int index = atomicAdd(&vertexStart[community], 1);
		orderedVertices[index] = vertex;
	}
}

__device__ void  mergeCommunity(int V, int *communities, device_structures deviceStructures, int prime, int *edgePos,
		int *communityDegree, int *orderedVertices, int *vertexStart, int *edgeIndexToCurPos, int *newEdges,
		float *newWeights, int *hashCommunity, float *hashWeight, int *prefixSum) {
	int communitiesOwned = 0;
	int communitiesPerBlock = blockDim.y;
	int concurrentThreads = blockDim.x;
	int hashTablesOffset = threadIdx.y * prime;
	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;

	if (communityIndex < V) {
		int community = communities[communityIndex];
		if (deviceStructures.communitySize[community] > 0) {
                for (unsigned int i = threadIdx.x; i < prime; i += concurrentThreads) {
                    hashWeight[hashTablesOffset + i] = 0;
                    hashCommunity[hashTablesOffset + i] = -1;
                }
                if (concurrentThreads > WARP_SIZE)
                    prefixSum[threadIdx.x] = 0;

				if (concurrentThreads > WARP_SIZE)
					__syncthreads();

				// filling hash tables content for every vertex in community
				for (int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int neighbour = deviceStructures.edges[index];
						float weight = deviceStructures.weights[index];
						int neighbourCommunity = deviceStructures.vertexCommunity[neighbour];
						int curPos = prepareHashArraysAggregation(neighbourCommunity, prime, weight, hashWeight,
																  hashCommunity, hashTablesOffset);
						if (curPos > -1) {
                            edgeIndexToCurPos[index] = curPos;
							communitiesOwned++;
						}
					}
				}

				int communitiesOwnedPrefixSum = communitiesOwned;
				if (concurrentThreads <= WARP_SIZE) {
					for (unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) {
						int otherSum = __shfl_up_sync(FULL_MASK, communitiesOwnedPrefixSum, offset);
						if (threadIdx.x >= offset) {
							communitiesOwnedPrefixSum += otherSum;
						}
					}
					// subtraction to have exclusive sum
					communitiesOwnedPrefixSum -= communitiesOwned;
				} else {
					for (unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) {
						__syncthreads();
						prefixSum[threadIdx.x] = communitiesOwnedPrefixSum;
						__syncthreads();
						if (threadIdx.x >= offset)
							communitiesOwnedPrefixSum += prefixSum[threadIdx.x - offset];
					}
					// subtraction to have exclusive sum
					communitiesOwnedPrefixSum -= communitiesOwned;
				}


				int newEdgesIndex = edgePos[community] + communitiesOwnedPrefixSum;
				if (threadIdx.x == concurrentThreads - 1) {
					communityDegree[community] = communitiesOwnedPrefixSum + communitiesOwned;
					atomicAdd(deviceStructures.E, communityDegree[community]);
				}
				for (int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int curPos = edgeIndexToCurPos[index];
						if (curPos > -1) {
							newEdges[newEdgesIndex] = hashCommunity[curPos];
							newWeights[newEdgesIndex] = hashWeight[curPos];
							newEdgesIndex++;
						}
					}
				}
		}
	}
}

__device__ void  mergeCommunity2(int V, int *communities, device_structures deviceStructures, int prime, int *edgePos,
		int *communityDegree, int *orderedVertices, int *vertexStart, int *edgeIndexToCurPos, int *newEdges,
		float *newWeights, int *hashCommunity, float *hashWeight, int *prefixSum) {
	int communitiesOwned = 0;
	int communitiesPerBlock = blockDim.y;
	int concurrentThreads = blockDim.x;
	int hashTablesOffset = threadIdx.y * prime;
	// int hashTablesOffset = 0;

	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;

	if (communityIndex < V) {
		int community = communities[communityIndex];
		if (deviceStructures.communitySize[community] > 0) {
                for (unsigned int i = threadIdx.x; i < prime; i += concurrentThreads) {
                    hashWeight[hashTablesOffset + i] = 0;
                    hashCommunity[hashTablesOffset + i] = -1;
                }
                if (concurrentThreads > WARP_SIZE)
                    prefixSum[threadIdx.x] = 0;

				if (concurrentThreads > WARP_SIZE)
					__syncthreads();

				// filling hash tables content for every vertex in community
				for (int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int neighbour = deviceStructures.edges[index];
						float weight = deviceStructures.weights[index];
						int neighbourCommunity = deviceStructures.vertexCommunity[neighbour];
						int curPos = prepareHashArraysAggregation(neighbourCommunity, prime, weight, hashWeight,
																  hashCommunity, hashTablesOffset);
						if (curPos > -1) {
                            edgeIndexToCurPos[index] = curPos;
							communitiesOwned++;
						}
					}
				}

				int communitiesOwnedPrefixSum = communitiesOwned;
				if (concurrentThreads <= WARP_SIZE) {
					for (unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) {
						int otherSum = __shfl_up_sync(FULL_MASK, communitiesOwnedPrefixSum, offset);
						if (threadIdx.x >= offset) {
							communitiesOwnedPrefixSum += otherSum;
						}
					}
					// subtraction to have exclusive sum
					communitiesOwnedPrefixSum -= communitiesOwned;
				} else {
					for (unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) {
						__syncthreads();
						prefixSum[threadIdx.x] = communitiesOwnedPrefixSum;
						__syncthreads();
						if (threadIdx.x >= offset)
							communitiesOwnedPrefixSum += prefixSum[threadIdx.x - offset];
					}
					// subtraction to have exclusive sum
					communitiesOwnedPrefixSum -= communitiesOwned;
				}


				int newEdgesIndex = edgePos[community] + communitiesOwnedPrefixSum;
				if (threadIdx.x == concurrentThreads - 1) {
					communityDegree[community] = communitiesOwnedPrefixSum + communitiesOwned;
					atomicAdd(deviceStructures.E, communityDegree[community]);
				}
				for (int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) {
					int vertex = orderedVertices[vertexStart[community] + vertexIndex];
					int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
					int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

					for (int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) {
						int index = vertexBaseIndex + neighbourIndex;
						int curPos = edgeIndexToCurPos[index];
						if (curPos > -1) {
							newEdges[newEdgesIndex] = hashCommunity[curPos];
							newWeights[newEdgesIndex] = hashWeight[curPos];
							newEdgesIndex++;
						}
					}
				}
		}
	}
}

__global__ void mergeCommunityShared(int V, int *communities, device_structures deviceStructures, int prime, int *edgePos,
										  int *communityDegree, int *orderedVertices, int *vertexStart, int *edgeIndexToCurPos, int *newEdges,
										  float *newWeights) {
	int communitiesPerBlock = blockDim.y;
	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;
	if (communityIndex < V) {
		extern __shared__ int s[];
		int *hashCommunity = s;
		auto *hashWeight = (float *) &hashCommunity[communitiesPerBlock * prime];
		auto *prefixSum = (int *) &hashWeight[communitiesPerBlock * prime];
		mergeCommunity(V, communities, deviceStructures, prime, edgePos, communityDegree, orderedVertices, vertexStart,
				edgeIndexToCurPos, newEdges, newWeights, hashCommunity, hashWeight, prefixSum);
	}
}

__global__ void mergeCommunityGlobal(int V, int *communities, device_structures deviceStructures, int prime, int *edgePos,
									 int *communityDegree, int *orderedVertices, int *vertexStart, int *edgeIndexToCurPos, int *newEdges,
									 float *newWeights, int *hashCommunity, float *hashWeight) {
	int communitiesPerBlock = blockDim.y;
	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;
	if (communityIndex < V) {
		extern __shared__ int s[];
		auto *prefixSum = s;
		hashCommunity = &hashCommunity[blockIdx.x * prime];
		hashWeight = &hashWeight[blockIdx.x * prime];
		mergeCommunity(V, communities, deviceStructures, prime, edgePos, communityDegree, orderedVertices, vertexStart,
					   edgeIndexToCurPos, newEdges, newWeights, hashCommunity, hashWeight, prefixSum);
	}
}

__global__ void mergeCommunityGlobal2(int V, int *communities, device_structures deviceStructures, int *primes, int *edgePos,
									 int *communityDegree, int *orderedVertices, int *vertexStart, int *edgeIndexToCurPos, int *newEdges,
									 float *newWeights, int *hashCommunity, float *hashWeight) {
	int communitiesPerBlock = blockDim.y;
	int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;
	if (communityIndex < V) {
		extern __shared__ int s[];
		auto *prefixSum = s;
		hashCommunity = &hashCommunity[primes[blockIdx.x]];
		hashWeight = &hashWeight[primes[blockIdx.x]];
		mergeCommunity2(V, communities, deviceStructures, primes[blockIdx.x+1]-primes[blockIdx.x], edgePos, communityDegree, orderedVertices, vertexStart,
					   edgeIndexToCurPos, newEdges, newWeights, hashCommunity, hashWeight, prefixSum);
	}
}


__global__ void compressEdges(int V, device_structures deviceStructures, int *communityDegree, int *newEdges,
		float *newWeights, int *newID, int *edgePos, int *vertexStart) {
	int communitiesPerBlock = blockDim.y;
	int concurrentThreads = blockDim.x;
	int community = blockIdx.x * communitiesPerBlock + threadIdx.y;
	if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		deviceStructures.edgesIndex[*deviceStructures.V] = *deviceStructures.E;
	}
	if (community < V && deviceStructures.communitySize[community] > 0) {
		int neighboursBaseIndex = edgePos[community];
		int communityNewID = newID[community];
		if (threadIdx.x == 0) {
			deviceStructures.vertexCommunity[communityNewID] = communityNewID;
			deviceStructures.newVertexCommunity[communityNewID] = communityNewID;
			deviceStructures.edgesIndex[communityNewID] = vertexStart[community];
		}
		for (int neighbourIndex = threadIdx.x; neighbourIndex < communityDegree[community]; neighbourIndex += concurrentThreads) {
			int newIndex = neighbourIndex + neighboursBaseIndex;
			int oldIndex = vertexStart[community] + neighbourIndex;
			deviceStructures.edges[oldIndex] = newID[newEdges[newIndex]];
			deviceStructures.weights[oldIndex] = newWeights[newIndex];
			atomicAdd(&deviceStructures.communityWeight[communityNewID], newWeights[newIndex]);
		}
	}
}

__global__ void updateOriginalToCommunity(device_structures deviceStructures, int *newID) {
	int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (vertex < *deviceStructures.originalV) {
		int community = deviceStructures.originalToCommunity[vertex];
		deviceStructures.originalToCommunity[vertex] = newID[community];
	}
}

struct IsInBucketAggregation
{
	IsInBucketAggregation(int llowerBound, int uupperBound, int *ccomunityDegree) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		communityDegree = ccomunityDegree;
	}

	int lowerBound, upperBound;
	int *communityDegree;
	__host__ __device__
	bool operator()(const int &v) const
	{
		int edgesNumber = communityDegree[v];
		
		return edgesNumber > lowerBound && edgesNumber <= upperBound;
	}
};

struct IsInBucketAggregation2
{
	IsInBucketAggregation2(int llowerBound, int uupperBound) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
	}

	int lowerBound, upperBound;
	__host__ __device__
	bool operator()(const int &v) const
	{
		
		return v > lowerBound && v <= upperBound;
	}
};

void aggregateCommunities(device_structures &deviceStructures, host_structures &hostStructures,
						  aggregation_phase_structures& aggregationPhaseStructures) {
    int V = hostStructures.V, E = hostStructures.E;
	int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int *communityDegree = aggregationPhaseStructures.communityDegree,
		*newID = aggregationPhaseStructures.newID, *edgePos = aggregationPhaseStructures.edgePos,
		*vertexStart = aggregationPhaseStructures.vertexStart,
		*orderedVertices = aggregationPhaseStructures.orderedVertices,
		*edgeIndexToCurPos = aggregationPhaseStructures.edgeIndexToCurPos,
		*newEdges = aggregationPhaseStructures.newEdges;
	float *newWeights = aggregationPhaseStructures.newWeights;

	cout << "Point 1A" << endl;

	// cout << "agg1" << endl;
	// cout << V << endl;
	int *vertices = (int*) malloc(V*sizeof(int));
	for (int i = 0; i < V; i++)
		vertices[i] = i;
	int *deviceVertices;

	// cout << "agg1.0.0.1" << endl;
	HANDLE_ERROR(cudaMalloc((void**)&deviceVertices, V * sizeof(int)));

	// cout << V << " agg1.0.1" << endl;

	HANDLE_ERROR(cudaMemcpy(deviceVertices, vertices, V * sizeof(int), cudaMemcpyHostToDevice));

	free(vertices);

	cout << "Point 2A" << endl;

	// cout << "agg1.1" << endl;

	thrust::fill(thrust::device, newID, newID + V, 0);
	thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, 0);
	thrust::fill(thrust::device, communityDegree, communityDegree + V, 0);
	fillArrays<<<blocks, THREADS_PER_BLOCK>>>(V, deviceStructures.communitySize, communityDegree, newID,
			deviceStructures.vertexCommunity, deviceStructures.edgesIndex);
	
	// cout << "agg1.2" << endl;

	// Manul: computes sum of newID, which is equal to total number of comm's
	int newV = thrust::reduce(thrust::device, newID, newID + V);
	thrust::exclusive_scan(thrust::device, newID, newID + V , newID);
	thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, edgePos);
	thrust::exclusive_scan(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, vertexStart);

	cout << "Point 3A" << endl;
	// cout << "agg2" << endl;

	orderVertices<<<blocks, THREADS_PER_BLOCK>>>(V, orderedVertices, vertexStart,
			deviceStructures.vertexCommunity);
//	 resetting vertexStart state to one before orderVertices call
	thrust::exclusive_scan(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, vertexStart);
    thrust::fill(thrust::device, edgeIndexToCurPos, edgeIndexToCurPos + E, -1);

	int bucketsSize = 4;
	int buckets[] = {0, 127, 479, INT_MAX};
	int primes[] = {191, 719};
	dim3 dims[] {
			{32, 4},
			{128, 1},
			{128, 1},
	};
	thrust::fill(thrust::device, deviceStructures.E, deviceStructures.E + 1, 0);

	cout << "Point 4A" << endl;

	for (int bucketNum = 0; bucketNum < bucketsSize - 2; bucketNum++) {
			dim3 blockDimension = dims[bucketNum];
			int prime = primes[bucketNum];
			auto predicate = IsInBucketAggregation(buckets[bucketNum], buckets[bucketNum + 1], communityDegree);
			int *deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
			int partitionSize = thrust::distance(deviceVertices, deviceVerticesEnd);
			if (partitionSize > 0) {
				unsigned int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int));
				if (blockDimension.x > WARP_SIZE)
					sharedMemSize += blockDimension.x * sizeof(int);
				unsigned int blocksDegrees = (partitionSize + blockDimension.y - 1) / blockDimension.y;
				mergeCommunityShared<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, deviceVertices, deviceStructures, prime, edgePos,
						communityDegree, orderedVertices, vertexStart, edgeIndexToCurPos, newEdges, newWeights);
			}
	}

	cout << "Point 5A" << endl;

	// cout << "agg3" << endl;

	dim3 blockDimension;
	// last bucket case
	int bucketNum = bucketsSize - 2;
	blockDimension = dims[bucketNum];
	int commDegree = newV;
	// int prime = getPrime(commDegree * 1.5);		// Manul: commented
	auto predicate = IsInBucketAggregation(buckets[bucketNum], buckets[bucketNum + 1], communityDegree);
	int *deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
	int partitionSize = thrust::distance(deviceVertices, deviceVerticesEnd);

	cout << "Point 6A" << endl;
	
	// int maxAllocatableCount = 1000000;
	// int partitionSizeSafe = (partitionSize > 0) ? partitionSize : 1;
	// int prime = getPrime(min((double) maxAllocatableCount / partitionSizeSafe, commDegree * 1.5));

	
	// int* temp = (int*) malloc(partitionSize*sizeof(int));
	// int* cdeg = (int*) malloc(V*sizeof(int));

	// int *nodedeg = (int *) malloc(partitionSize*sizeof(int));

	// cout << "deg start" << endl;
	// HANDLE_ERROR(cudaMemcpy(temp, deviceVertices, partitionSize*sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(cdeg, communityDegree, V*sizeof(int), cudaMemcpyDeviceToHost));
	// cout << "deg done" << endl;

	// long long mnn = LLONG_MAX, mxx = LLONG_MIN;
	// for(int i = 0; i < partitionSize; i++) {
	// 	// cout <<" sfsdf " << endl;
	// 	// cout << i << " " << temp[i] << endl;
	// 	nodedeg[i] = cdeg[temp[i]];
	// 	mnn = min(mnn, (long long) cdeg[temp[i]]);
	// 	mxx = max(mxx, (long long) cdeg[temp[i]]);
	// }

	// sort(nodedeg, nodedeg + partitionSize);
	// ofstream fout("temp.txt");
	// for(int i = 0; i < partitionSize; i++) {
	// 	fout << nodedeg[i] << endl;
	// 	cout << nodedeg[i] << endl;
	// }
	// fout.close();
	
	// cout << mnn << " " << mxx << endl;

	if (partitionSize > 0) {

		int* communityDegree2;
		HANDLE_ERROR(cudaMalloc((void**)&communityDegree2, V*sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(communityDegree2, communityDegree, V*sizeof(int), cudaMemcpyDeviceToDevice));

		auto predicate2 = IsInBucketAggregation2(buckets[bucketNum], buckets[bucketNum + 1]);
		int *deviceVerticesEnd2 = thrust::partition(thrust::device, communityDegree2, communityDegree2 + V, predicate2);
		int partitionSize2 = thrust::distance(communityDegree2, deviceVerticesEnd2);

		cout << "Point 7A" << endl;

		// int maxDegree = thrust::reduce(thrust::device, communityDegree2, communityDegree2 + partitionSize2);

		// int prime = getPrime(maxDegree * 1.5);
		int* commDeg2 = (int*) malloc(partitionSize2*sizeof(int));
		HANDLE_ERROR(cudaMemcpy(commDeg2, communityDegree2, partitionSize2*sizeof(int), cudaMemcpyDeviceToHost));
		int* primes2 = (int*)malloc((partitionSize2+1)*sizeof(int));
		for(int i = 0; i < partitionSize2; i++) primes2[i] = getPrime(commDeg2[i] * 1.5);
		primes2[partitionSize2] = 0;

		cout << "Point 8A" << endl;

		int* primesD;
		HANDLE_ERROR(cudaMalloc((void**)&primesD, (partitionSize2+1)*sizeof(int)));

		HANDLE_ERROR(cudaMemcpy(primesD, primes2, (partitionSize2+1)*sizeof(int), cudaMemcpyHostToDevice));

		thrust::exclusive_scan(thrust::device, primesD, primesD+partitionSize2+1, primesD);
		
		int prime;
		HANDLE_ERROR(cudaMemcpy(&prime, primesD + partitionSize2, sizeof(int), cudaMemcpyDeviceToHost));

		cout << "Total " << prime << endl;

		int *hashCommunity;
		float *hashWeight;
		// cout << prime << " " << partitionSize << " " << (long long)prime*partitionSize << endl;
		HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, prime * sizeof(int)));		// Manul: change
		HANDLE_ERROR(cudaMalloc((void**)&hashWeight, prime * sizeof(float)));	// Manul: change
		// HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, (long long) prime * partitionSize * sizeof(int)));		// Manul: change
		// HANDLE_ERROR(cudaMalloc((void**)&hashWeight, (long long) prime * partitionSize * sizeof(float)));	// Manul: change
		unsigned int sharedMemSize = THREADS_PER_BLOCK * sizeof(int);
		unsigned int blocksDegrees = (partitionSize + blockDimension.y - 1) / blockDimension.y;
		// mergeCommunityGlobal<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, deviceVertices, deviceStructures, prime, edgePos,
		// 																	   communityDegree, orderedVertices, vertexStart, edgeIndexToCurPos, newEdges, newWeights,
		// 																	   hashCommunity, hashWeight);
		mergeCommunityGlobal2<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, deviceVertices, deviceStructures, primesD, edgePos,
																			   communityDegree, orderedVertices, vertexStart, edgeIndexToCurPos, newEdges, newWeights,
																			   hashCommunity, hashWeight);
		HANDLE_ERROR(cudaFree(hashCommunity));
		HANDLE_ERROR(cudaFree(hashWeight));

		HANDLE_ERROR(cudaFree(communityDegree2));
		HANDLE_ERROR(cudaFree(primesD));
		free(primes2);
		free(commDeg2);

	}

	// cout << "agg4" << endl;

	HANDLE_ERROR(cudaMemcpy(&hostStructures.E, deviceStructures.E, sizeof(int), cudaMemcpyDeviceToHost));
	hostStructures.V = newV;
	HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &newV, sizeof(int), cudaMemcpyHostToDevice));
	thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + hostStructures.V, 1);
	int blocksNum = (V * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	blockDimension = {WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE};

	thrust::fill(thrust::device, deviceStructures.communityWeight, deviceStructures.communityWeight + hostStructures.V, (float) 0);
	// vertexStart will contain starting indexes in compressed list
	thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, vertexStart);
	compressEdges<<<blocksNum, blockDimension>>>(V, deviceStructures, communityDegree, newEdges, newWeights, newID, edgePos, vertexStart);
	HANDLE_ERROR(cudaFree(deviceVertices));
	updateOriginalToCommunity<<<(hostStructures.originalV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(deviceStructures, newID);

	// cout << "agg5" << endl;

	// printf("newID\n");
	// int *communityDegreeH = (int*) malloc(V*sizeof(int));
	// int *newIDH = (int*) malloc(V*sizeof(int));
	// int *edgePosH = (int*) malloc(V*sizeof(int));
	// int *vertexStartH = (int*) malloc(V*sizeof(int));
	// int *orderedVerticesH = (int*) malloc(V*sizeof(int));
	// int *edgeIndexToCurPosH = (int*) malloc(E*sizeof(int));
	// int *newEdgesH = (int*) malloc(E*sizeof(int));
	// float *newWeightsH = (float*) malloc(E*sizeof(float));
	
	// HANDLE_ERROR(cudaMemcpy(newIDH, aggregationPhaseStructures.newID, V * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(communityDegreeH, aggregationPhaseStructures.communityDegree, V * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(edgePosH, aggregationPhaseStructures.edgePos, V * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(vertexStartH, aggregationPhaseStructures.vertexStart, V * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(orderedVerticesH, aggregationPhaseStructures.orderedVertices, V * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(edgeIndexToCurPosH, aggregationPhaseStructures.edgeIndexToCurPos, E * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(newEdgesH, aggregationPhaseStructures.newEdges, E * sizeof(int), cudaMemcpyDeviceToHost));
	// HANDLE_ERROR(cudaMemcpy(newWeightsH, aggregationPhaseStructures.newWeights, E * sizeof(float), cudaMemcpyDeviceToHost));

	// printf("newIDs:\n");
	// for (int i = 0; i < V; i++) printf("%d ", newIDH[i]);
	// printf("\n");
	// printf("comm Degrees:\n");
	// for (int i = 0; i < V; i++) printf("%d ", communityDegreeH[i]);
	// printf("\n");
	// printf("edge pos:\n");
	// for (int i = 0; i < V; i++) printf("%d ", edgePosH[i]);
	// printf("\n");
	// printf("vertex start:\n");
	// for (int i = 0; i < V; i++) printf("%d ", vertexStartH[i]);
	// printf("\n");
	// printf("ordered vertices:\n");
	// for (int i = 0; i < V; i++) printf("%d ", orderedVerticesH[i]);
	// printf("\n");
	// printf("edge index to cur pos:\n");
	// for (int i = 0; i < E; i++) printf("%d ", edgeIndexToCurPosH[i]);
	// printf("\n");
	// printf("new edges:\n");
	// for (int i = 0; i < E; i++) printf("%d ", newEdgesH[i]);
	// printf("\n");
	// printf("new weights:\n");
	// for (int i = 0; i < E; i++) printf("%f ", newWeightsH[i]);
	// printf("\n");
	
}
