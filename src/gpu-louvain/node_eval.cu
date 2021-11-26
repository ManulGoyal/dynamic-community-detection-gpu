#include "node_eval.cuh"
#include "modularity_optimisation.cuh"
#include <thrust/partition.h>
#include <vector>

__constant__ float MM;

struct nodeIndex {
	int node, index, prime;
	int *hashCommunity;

	__device__ __host__
	nodeIndex() {}

	__device__ __host__
	nodeIndex(int node, int index) : node(node), index(index) {} 
};

/**
 * Computes hashing (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHashNE(int val, int index, int prime) {
	int h1 = val % prime;
	int h2 = 1 + (val % (prime - 1));
	return (h1 + index * h2) % prime;
}


/**
 * Fills content of hashCommunity and hash_weights arrays that are later used in computeGain function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight		   neighbour's weight
 * @param hashWeight	   table of sum of weights between vertices and communities
 * @param hashCommunity	   table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 */
__device__ int prepareHashArraysNE(int community, int prime, float weight, float *hashWeight, int *hashCommunity,
								  int hashTablesOffset) {
	int it = 0, curPos;
	do {
		curPos = hashTablesOffset + getHashNE(community, it++, prime);
		if (hashCommunity[curPos] == community)
			atomicAdd(&hashWeight[curPos], weight);
		else if (hashCommunity[curPos] == -1) {
			if (atomicCAS(&hashCommunity[curPos], -1, community) == -1)
				atomicAdd(&hashWeight[curPos], weight);
			else if (hashCommunity[curPos] == community)
				atomicAdd(&hashWeight[curPos], weight);
		}
	} while (hashCommunity[curPos] != community);
	return curPos;
}

/**
 * Computes gain that would be obtained if we would move vertex to community.
 * @param vertex      	   vertex number
 * @param prime            prime number used for hashing (and size of vertex's area in hash arrays)
 * @param community 	   neighbour's community
 * @param currentCommunity current community of vertex
 * @param communityWeight  community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexEdgesSum   vertex -> sum of edges adjacent to vertex
 * @param hashCommunity    table informing which community's info is stored in given index
 * @param hashWeight       table of sum of weights between vertices and communities
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices
 * @return gain that would be obtained by moving vertex to community
 */
__device__ float computeGainNE(int vertex, int community, int currentCommunity, float *communityWeight,
							 float *vertexEdgesSum, float vertexToCommunity) {
	
	// Manul: vertexToCommunity contains e_{vertex -> community}
	float communitySum = communityWeight[community];
	float currentCommunitySum = communityWeight[currentCommunity] - vertexEdgesSum[vertex];
	float gain = vertexToCommunity / MM + vertexEdgesSum[vertex] * (currentCommunitySum - communitySum) / (2 * MM * MM);
	
	// remove
	// printf("computeGainNE -> vertexToCommunity = %f, vertexEdgesSum[vertex] = %f, currentCommunitySum - communitySum = %f, MM = %f\n", vertexToCommunity, vertexEdgesSum[vertex], (currentCommunitySum - communitySum), MM);

	return gain;
}

__device__ bool findInHashTable(int* hashCommunity, int community, int prime, int hashTablesOffset) {
	int it = 0, curPos;
	while(true) {
		curPos = hashTablesOffset + getHashNE(community, it++, prime);
		if (hashCommunity[curPos] == community)
			return true;
		if (hashCommunity[curPos] == -1)
			return false;
	}
	return false;
}

/**
 * Finds new vertex -> community assignment (stored in newVertexCommunity) that maximise gains for each vertex.
 * @param V                number of vertices
 * @param vertices		   source vertices of newly added edges
 * @param prime            prime number used for hashing
 * @param deviceStructures structures kept in device memory
 */
__device__ void computeBestComm(int V, nodeIndex *vertices, int prime, device_structures deviceStructures, int *hashCommunity, float *hashWeight, float *vertexToCurrentCommunity, float *bestGains, int *bestCommunities, int *nodeEval, int *commEval) {   //

			// Manul: vertexToCurrentCommunity[i] stores the value of ei -> C(i)\{i} after
			// Manul: this function finishes execution

	// remove
	// if(threadIdx.x == 0) {
	// 	printf(" - In computeBestComm kernel\n");
	// }


	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		int *vertexCommunity = deviceStructures.vertexCommunity, *edgesIndex = deviceStructures.edgesIndex,
		*edges = deviceStructures.edges, *communitySize = deviceStructures.communitySize;
		float *weights = deviceStructures.weights, *communityWeight = deviceStructures.communityWeight,
		*vertexEdgesSum = deviceStructures.vertexEdgesSum;

		int concurrentNeighbours = blockDim.x;	// Manul: threads per vertex
		int hashTablesOffset = threadIdx.y * prime;

        if (threadIdx.x == 0) {
		    vertexToCurrentCommunity[threadIdx.y] = 0;
		}

		// Manul: this loop just fills all entries of hashWeight with 0 and hashCommunity with -1
		// Manul: suppose blockDim.x = concurrentNeighbours = 4. Then thread with x index
		// Manul: equal to 0 will process the 0th, 4th, 8th, ... neighbours of this vertex (threadIdx.y)
		for (unsigned int i = threadIdx.x; i < prime; i += concurrentNeighbours) {
			hashWeight[hashTablesOffset + i] = 0;
			hashCommunity[hashTablesOffset + i] = -1;
		}

		if (concurrentNeighbours > WARP_SIZE)
			__syncthreads();

		int vertex = vertices[vertexIndex].node;
		int currentCommunity = vertexCommunity[vertex];
		int bestCommunity = currentCommunity;
		float bestGain = 0;

		/*** commented ***/
		// if(threadIdx.x == 0) {
		// 	nodeEval[vertex] = 1;
		// }
		/*** commented ***/

		// putting data in hash table
		// Manul: the thread with x = 0 will process the 0th, (0+blockDim.x)th, ... neighbours of vertex
		int neighbourIndex = threadIdx.x + edgesIndex[vertex];
		int upperBound = edgesIndex[vertex + 1];
		int curPos;

		while (neighbourIndex < upperBound) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];

			if(findInHashTable(vertices[vertexIndex].hashCommunity, community, vertices[vertexIndex].prime, 0)) {

				// remove
				// printf("Node %d Neighbour %d comm in hash table\n", vertex, neighbour);

				// Manul: this is the weight of the edge (vertex, neighbour)
				float weight = weights[neighbourIndex];	
				// this lets us achieve ei -> C(i)\{i} instead of ei -> C(i)
				if (neighbour != vertex) {
					// Manul: finds the appropriate position in the hash tables for 'community',
					// Manul: which is returned as curPos, and also sets hashCommunity[curPos] = community
					// Manul: and adds 'weight' to hashWeight[curPos]
					curPos = prepareHashArraysNE(community, prime, weight, hashWeight, hashCommunity, hashTablesOffset);
					if (community == currentCommunity)
						atomicAdd(&vertexToCurrentCommunity[threadIdx.y], weight);
				}
				if ((community < currentCommunity || communitySize[community] > 1 || communitySize[currentCommunity] > 1) &&
					community != currentCommunity) {

					// Manul: Note: Although here there is no guarantee that hashWeight[curPos] contains
					// Manul: the fully computed value of e_{vertex -> community}, but as 
					// Manul: threads update hashWeight[curPos], it can only increase, and
					// Manul: therefore gain can also only increase (since gain increases with
					// Manul: e_{vertex -> community}). Hence, the thread that last processes a 
					// Manul: neighbour of community = 'community' will see the actual value of
					// Manul: e_{vertex -> community} in hashWeight[curPos], and thus compute the
					// Manul: actual gain obtained from moving 'vertex' to 'community', thus,
					// Manul: this last thread will have bestGain >= the actual gain
					float gain = computeGainNE(vertex, community, currentCommunity, communityWeight, vertexEdgesSum, hashWeight[curPos]);
					
					// remove
					// printf("Node %d Neighbour %d Computed gain = %f\n", vertex, neighbour, gain);

					if (gain > bestGain || (gain == bestGain && community < bestCommunity)) {
						bestGain = gain;
						bestCommunity = community;
					}
				}
			}
			
			neighbourIndex += concurrentNeighbours;
		}

		// Manul: now bestGain contains the best gain that can be obtained by moving 'vertex'
		// Manul: from its original community to community of one of its (threadIdx.x + k * concurrentNeighbours)-th neighbours, where k = 0,1,2,...
		// Manul: bestCommunity contains the community ID of the community that yields the
		// Manul: bestGain from the aforementioned communities

		
		if (concurrentNeighbours <= WARP_SIZE) {
			// Manul: the below loop simply max-aggregates the bestGain's of all threads
			// Manul: in the same warp (which process the same vertex) and puts the max
			// Manul: value in the 0-th, blockDim.x-th, 2*blockDim.x-th... threads of the warp
			for (unsigned int offset = concurrentNeighbours / 2; offset > 0; offset /= 2) {
				float otherGain = __shfl_down_sync(FULL_MASK, bestGain, offset);
				int otherCommunity = __shfl_down_sync(FULL_MASK, bestCommunity, offset);
				if (otherGain > bestGain || (otherGain == bestGain && otherCommunity < bestCommunity)) {
					bestGain = otherGain;
					bestCommunity = otherCommunity;
				}
			}
			// Manul: now, if this thread has x = 0, then bestGain = max(bestGain's of all threads with same y) = max(bestGain's of all threads processing the same vertex)
			// Manul: and, bestCommunity = the neighbouring community corresponding to this bestGain 
		} else {
			// Manul: here, the blockDim = (128, 1), so, the whole block is assigned to a single vertex
			// Manul: here, concurrentNeighbours = 128 and this thread is responsible for processing
			// Manul: the (threadIdx.x + k * 128)-th neighbours of this vertex (k = 0,1,2,...)
			// Manul: bestGains array in shared memory has size = 128 * sizeof(float)
            bestGains[threadIdx.x] = bestGain;
            bestCommunities[threadIdx.x] = bestCommunity;

			// Manul: this loop max-aggregates the bestGain values of each of the 128 threads
			// Manul: in this block into the thread with x index of 0
			for (unsigned int offset = concurrentNeighbours / 2; offset > 0; offset /= 2) {
				__syncthreads();
				if (threadIdx.x < offset) {
					float otherGain = bestGains[threadIdx.x + offset];
					int otherCommunity = bestCommunities[threadIdx.x + offset];
					if (otherGain > bestGains[threadIdx.x] ||
					   (otherGain == bestGains[threadIdx.x] && otherCommunity < bestCommunities[threadIdx.x])) {
						bestGains[threadIdx.x] = otherGain;
						bestCommunities[threadIdx.x] = otherCommunity;
					}
				}
			}
            bestGain = bestGains[threadIdx.x];
            bestCommunity = bestCommunities[threadIdx.x];
		}

		// remove
		// if(threadIdx.x == 0) {
		// 	printf("Node %d, bestGain %f\n", vertex, bestGain);
		// }

		// Manul: the bestGain in thread with x index of 0 is the required best gain that can
		// Manul: be obtained by moving 'vertex' into one of its neighbouring communities
		// Manul: bestGain - vertexToCurrentCommunity[threadIdx.y] / M will give the actual value 
		// Manul: of delta_Q_{vertex -> bestCommunity}
		if (threadIdx.x == 0 && bestGain - vertexToCurrentCommunity[threadIdx.y] / MM > 0) {
			// newVertexCommunity[vertex] = bestCommunity;
			// atomicExch(&commEval[bestCommunity], 1);	// Manul: TODO: Is this overkill?
			commEval[bestCommunity] = 1;
			nodeEval[vertex] = 1;

			// remove
			// printf(" -- Marking comm\n");
		} 
		// else {
		// 	// Manul: Issue: Won't this possibly overwrite the change made by the above if statement?
		// 	newVertexCommunity[vertex] = currentCommunity;
		// }
	}
}

__global__ void computeBestCommShared(int V, nodeIndex *vertices, int prime, device_structures deviceStructures, int *nodeEval, int *commEval) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		extern __shared__ int s[];
		// Manul: hashCommunity is int array of size verticesPerBlock * prime
		int *hashCommunity = s;
		// Manul: hashWeight is float array of size verticesPerBlock * prime
		auto *hashWeight = (float *) &hashCommunity[verticesPerBlock * prime];
		// Manul: vertexToCurrentCommunity is float array of size verticesPerBlock
		// Manul: vertexToCurrentCommunity[i] is used to store e_{i -> C(i)\{i}}
		auto *vertexToCurrentCommunity = (float *) &hashWeight[verticesPerBlock * prime];

		// Manul: the next two arrays are only used by blocks corresponding to buckets 
		// Manul: with blockDim.x (i.e., threads per vertex) > 32, because all threads belonging
		// Manul: to same vertex do not belong to the same warp in this case
		// Manul: so, each thread assigned to the same vertex has to store its intermediate
		// Manul: results into shared memory arrays, which are defined below

		// Manul: bestGains is float array of size THREADS_PER_BLOCK (128)
		float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
		// Manul: bestCommunities is int array of size THREADS_PER_BLOCK (128)
		int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
		computeBestComm(V, vertices, prime, deviceStructures, hashCommunity, hashWeight, vertexToCurrentCommunity,
				bestGains, bestCommunities, nodeEval, commEval);
	}
}

// __global__ void computeMoveGlobal(int V, int *vertices, int prime, device_structures deviceStructures, int *hashCommunity, float *hashWeight) {
// 	int verticesPerBlock = blockDim.y;
// 	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
// 	if (vertexIndex < V) {
// 		extern __shared__ int s[];
// 		auto *vertexToCurrentCommunity = (float *) s;
// 		float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
// 		int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
// 		hashCommunity = hashCommunity + blockIdx.x * prime;
// 		hashWeight = hashWeight + blockIdx.x * prime;
// 		computeMove(V, vertices, prime, deviceStructures, hashCommunity, hashWeight, vertexToCurrentCommunity,
// 					bestGains, bestCommunities);
// 	}
// }

__global__ void computeBestCommGlobal(int V, nodeIndex *vertices, int *primes, device_structures deviceStructures, int *hashCommunity, float *hashWeight, int *nodeEval, int *commEval) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		extern __shared__ int s[];
		auto *vertexToCurrentCommunity = (float *) s;
		float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
		int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
		hashCommunity = hashCommunity + primes[blockIdx.x];
		hashWeight = hashWeight + primes[blockIdx.x];
		computeBestComm(V, vertices, primes[blockIdx.x+1] - primes[blockIdx.x], deviceStructures, hashCommunity, hashWeight, vertexToCurrentCommunity,
					bestGains, bestCommunities, nodeEval, commEval);
	}
}

__device__ void insertCommunityInHashTable(int* hashCommunity, int community, int prime, int hashTablesOffset) {
	int it = 0, curPos;
	do {
		curPos = hashTablesOffset + getHashNE(community, it++, prime);
		if (hashCommunity[curPos] == -1) {
			atomicCAS(&hashCommunity[curPos], -1, community);
				
			// else if (hashCommunity[curPos] == community)
			// 	;
		}
	} while (hashCommunity[curPos] != community);
}

__device__ void computeCommunitiesSV(int V, nodeIndex *vertices, int *edges, int *edgesIndex, int prime, device_structures deviceStructures, int *hashCommunity) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		int* vertexCommunity = deviceStructures.vertexCommunity;
		// extern __shared__ int s[];
		// auto *vertexToCurrentCommunity = (float *) s;
		// float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
		// int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
		nodeIndex vertex = vertices[vertexIndex];
		int currentCommunity = vertexCommunity[vertex.node];
		// int prime = primes[vertex.index + 1] - primes[vertex.index];
		int concurrentNeighbours = blockDim.x;
		int hashTablesOffset = threadIdx.y * prime;

		if(threadIdx.x == 0) {
			vertices[vertexIndex].prime = prime;
			vertices[vertexIndex].hashCommunity = hashCommunity + hashTablesOffset;
		}

		// hashWeight = hashWeight + primes[blockIdx.x];
		// computeMove(V, vertices, primes[blockIdx.x+1] - primes[blockIdx.x], deviceStructures, hashCommunity, hashWeight, vertexToCurrentCommunity,
		// 			bestGains, bestCommunities);
		for (unsigned int i = threadIdx.x; i < prime; i += concurrentNeighbours) {
			hashCommunity[hashTablesOffset + i] = -1;
		}

		// Manul: TODO: check
		if (concurrentNeighbours > WARP_SIZE)
			__syncthreads();
		
		insertCommunityInHashTable(hashCommunity, currentCommunity, prime, hashTablesOffset);

		int neighbourIndex = threadIdx.x + edgesIndex[vertex.index];
		int upperBound = edgesIndex[vertex.index + 1];

		while (neighbourIndex < upperBound) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];

			if(currentCommunity != community) {
				insertCommunityInHashTable(hashCommunity, community, prime, hashTablesOffset);
			}
			neighbourIndex += concurrentNeighbours;
		}
	}
}

__global__ void computeCommunitiesSVLastBucket(int V, nodeIndex *vertices, int *edges, int *edgesIndex, int *primes, device_structures deviceStructures, int *hashCommunity) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		hashCommunity = hashCommunity + primes[blockIdx.x];
		// hashWeight = hashWeight + primes[blockIdx.x];
		computeCommunitiesSV(V, vertices, edges, edgesIndex, primes[blockIdx.x+1] - primes[blockIdx.x], deviceStructures, hashCommunity);
	}
}

__global__ void computeCommunitiesSVGeneral(int V, nodeIndex *vertices, int *edges, int *edgesIndex, int prime, device_structures deviceStructures, int *hashCommunity) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		// extern __shared__ int s[];
		// Manul: hashCommunity is int array of size verticesPerBlock * prime
		// int *hashCommunity = s;
		// Manul: hashWeight is float array of size verticesPerBlock * prime
		// auto *hashWeight = (float *) &hashCommunity[verticesPerBlock * prime];
		// Manul: vertexToCurrentCommunity is float array of size verticesPerBlock
		// Manul: vertexToCurrentCommunity[i] is used to store e_{i -> C(i)\{i}}
		// auto *vertexToCurrentCommunity = (float *) &hashWeight[verticesPerBlock * prime];

		// Manul: the next two arrays are only used by blocks corresponding to buckets 
		// Manul: with blockDim.x (i.e., threads per vertex) > 32, because all threads belonging
		// Manul: to same vertex do not belong to the same warp in this case
		// Manul: so, each thread assigned to the same vertex has to store its intermediate
		// Manul: results into shared memory arrays, which are defined below

		// Manul: bestGains is float array of size THREADS_PER_BLOCK (128)
		// float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
		// Manul: bestCommunities is int array of size THREADS_PER_BLOCK (128)
		// int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
		hashCommunity += blockIdx.x * verticesPerBlock * prime;
		computeCommunitiesSV(V, vertices, edges, edgesIndex, prime, deviceStructures, hashCommunity);
	}
}


__global__ void computeFinalNodeEval(int V, nodeIndex *vertices, device_structures deviceStructures, int *finalNodeEval) {   //

	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		int *edgesIndex = deviceStructures.edgesIndex, *edges = deviceStructures.edges;

		int concurrentNeighbours = blockDim.x;	// Manul: threads per vertex
		// int hashTablesOffset = threadIdx.y * prime;

        // if (threadIdx.x == 0) {
		//     vertexToCurrentCommunity[threadIdx.y] = 0;
		// }

		// Manul: this loop just fills all entries of hashWeight with 0 and hashCommunity with -1
		// Manul: suppose blockDim.x = concurrentNeighbours = 4. Then thread with x index
		// Manul: equal to 0 will process the 0th, 4th, 8th, ... neighbours of this vertex (threadIdx.y)
		// for (unsigned int i = threadIdx.x; i < prime; i += concurrentNeighbours) {
		// 	hashWeight[hashTablesOffset + i] = 0;
		// 	hashCommunity[hashTablesOffset + i] = -1;
		// }

		// Manul: TODO: Check if should be commented
		// if (concurrentNeighbours > WARP_SIZE)
		// 	__syncthreads();

		int vertex = vertices[vertexIndex].node;
		// int currentCommunity = vertexCommunity[vertex];
		// int bestCommunity = currentCommunity;
		// float bestGain = 0;

		/*** commented ***/
		if(threadIdx.x == 0) {
			finalNodeEval[vertex] = 1;
		}
		/*** commented ***/

		// putting data in hash table
		// Manul: the thread with x = 0 will process the 0th, (0+blockDim.x)th, ... neighbours of vertex
		int neighbourIndex = threadIdx.x + edgesIndex[vertex];
		int upperBound = edgesIndex[vertex + 1];

		while (neighbourIndex < upperBound) {
			int neighbour = edges[neighbourIndex];
			// int community = vertexCommunity[neighbour];
			
			finalNodeEval[neighbour] = 1;
			neighbourIndex += concurrentNeighbours;
		}
		
	}
}


struct isInBucket
{
	isInBucket(int llowerBound, int uupperBound, int *eedgesIndex) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		edgesIndex = eedgesIndex;
	}

	int lowerBound, upperBound;
	int *edgesIndex;
	__host__ __device__
	bool operator()(const nodeIndex &v) const
	{
		int edgesNumber = edgesIndex[v.node + 1] - edgesIndex[v.node];
		return edgesNumber > lowerBound && edgesNumber <= upperBound;
	}
};

struct isInBucketSV
{
	isInBucketSV(int llowerBound, int uupperBound, int *eedgesIndex) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		edgesIndex = eedgesIndex;
	}

	int lowerBound, upperBound;
	int *edgesIndex;
	__host__ __device__
	bool operator()(const nodeIndex &v) const
	{
		int edgesNumber = edgesIndex[v.index + 1] - edgesIndex[v.index];
		return edgesNumber > lowerBound && edgesNumber <= upperBound;
	}
};

struct isInBucketEval
{
	isInBucketEval(int llowerBound, int uupperBound, int *eedgesIndex, int *nnodeEval) {
		lowerBound = llowerBound;
		upperBound = uupperBound;
		edgesIndex = eedgesIndex;
		nodeEval = nnodeEval;
	}

	int lowerBound, upperBound;
	int *edgesIndex, *nodeEval;
	__host__ __device__
	bool operator()(const nodeIndex &v) const
	{
		int edgesNumber = edgesIndex[v.node + 1] - edgesIndex[v.node];
		return nodeEval[v.node] && (edgesNumber > lowerBound && edgesNumber <= upperBound);
	}
};

__global__ void computeNodeEval(int V, int *finalNodeEval, int *commEval, int *R, int *R_size, device_structures deviceStructures) {
	int vertex = blockDim.x * blockIdx.x + threadIdx.x;

	if(vertex < V) {
		if(finalNodeEval[vertex] == 0) {
			int community = deviceStructures.vertexCommunity[vertex];
			if(commEval[community] == 1) {
				finalNodeEval[vertex] = 1;
			}
		}
		if(finalNodeEval[vertex] == 1) {
			R[atomicAdd(R_size, 1)] = vertex;
		}
	}
}

int* computeFinalNodeEval_gpu(device_structures& deviceStructures, host_structures& hostStructures, vector<nodeIndex>& sourceVerticesNI, int* nodeEval, nodeIndex *partition) {
	int svCount = sourceVerticesNI.size();
	
	int lastBucketNum = bucketsSize - 2;
    dim3 lastBlockDimension = dims[lastBucketNum];

	cout << "Before partition" << endl;

	auto predicate = isInBucketEval(buckets[lastBucketNum], buckets[lastBucketNum + 1], hostStructures.edgesIndex, nodeEval);		// Manul: TODO: check if it works

	/***/
	// nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);				//
	/***/
    
	cout << "After partition" << endl;

	/***/
	// int verticesInLastBucket = thrust::distance(partition, deviceVerticesEnd);
	/***/

    cout << "Point 1M" << endl;

	/***/
    // int* primes_d;		// Manul change
	// int *hashCommunity;
	// float *hashWeight;
	/***/

	// assert(hashOffsetsSV.size() == svCount+1);

	// HANDLE_ERROR(cudaMalloc((void**)&primes_d, (svCount + 1) * sizeof(int)));		// free it
	// HANDLE_ERROR(cudaMemcpy(primes_d, &hashOffsetsSV[0], (svCount + 1) * sizeof(int), cudaMemcpyHostToDevice));

	/***/
    // if (verticesInLastBucket > 0) {

	// 	/* Manul change start */
	// 	nodeIndex* partition_h = (nodeIndex*) malloc(verticesInLastBucket * sizeof(nodeIndex));
	// 	int* primes_h = (int*) malloc((verticesInLastBucket + 1) * sizeof(int));
	// 	HANDLE_ERROR(cudaMemcpy(partition_h, partition, verticesInLastBucket * sizeof(nodeIndex), cudaMemcpyDeviceToHost));
	// 	primes_h[0] = 0;
	// 	for(int vi = 0; vi < verticesInLastBucket; vi++) {
	// 		int v = partition_h[vi].node;
	// 		// int v_ind = partition_h[vi].index;
	// 		int degv = hostStructures.edgesIndex[v+1] - hostStructures.edgesIndex[v];
	// 		primes_h[vi + 1] = primes_h[vi] + getPrime(degv * 1.5);
	// 	}
	// 	HANDLE_ERROR(cudaMalloc((void**)&primes_d, (verticesInLastBucket + 1) * sizeof(int)));		// free it done
	// 	HANDLE_ERROR(cudaMemcpy(primes_d, primes_h, (verticesInLastBucket + 1) * sizeof(int), cudaMemcpyHostToDevice));
	// 	/* Manul change end */


	// 	HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, primes_h[verticesInLastBucket] * sizeof(int)));		// Manul change free it done
    //     HANDLE_ERROR(cudaMalloc((void**)&hashWeight, primes_h[verticesInLastBucket] * sizeof(float)));		// Manul change free it done

	// 	free(partition_h);
	// 	free(primes_h);
    // }
	/***/

	int *finalNodeEval;
	HANDLE_ERROR(cudaMalloc((void**)&finalNodeEval, hostStructures.V * sizeof(int))); // free it done
	HANDLE_ERROR(cudaMemset(finalNodeEval, 0, hostStructures.V * sizeof(int)));

	cout << "Point 2M" << endl;

    for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
        dim3 blockDimension = dims[bucketNum];
        // int prime = primes[bucketNum];
        auto predicate = isInBucketEval(buckets[bucketNum], buckets[bucketNum + 1], hostStructures.edgesIndex, nodeEval);
        nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);			//
        int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
        if (verticesInBucket > 0) {
            // int sharedMemSize =
            //         blockDimension.y * prime * (sizeof(float) + sizeof(int)) + blockDimension.y * sizeof(float);
            // if (blockDimension.x > WARP_SIZE)
            //     sharedMemSize += THREADS_PER_BLOCK * (sizeof(int) + sizeof(float));
            int blocksNum = (verticesInBucket + blockDimension.y - 1) / blockDimension.y;

			// HANDLE_ERROR(cudaMalloc((void**)&hashCommunity[bucketNum], prime * verticesInBucket * sizeof(int)));		// Manul change free it

			// computeCommunitiesSVGeneral<<<blocksNum, blockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, prime, deviceStructures, hashCommunity[bucketNum]);

            // computeBestCommShared<<<blocksNum, blockDimension, sharedMemSize>>>(verticesInBucket, partition, prime, deviceStructures, nodeEval, commEval);

			computeFinalNodeEval<<<blocksNum, blockDimension>>>(verticesInBucket, partition, deviceStructures, finalNodeEval);

			// // updating vertex -> community assignment
            // updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, partition,
            //                                                                     deviceStructures);
            // // updating community weight
            // thrust::fill(thrust::device, deviceStructures.communityWeight,
            //                 deviceStructures.communityWeight + hostStructures.V, (float) 0);
            // computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);
        }
    }

    cout << "Point 3M" << endl;


    // last bucket case
    nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);		//
    int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
    if (verticesInBucket > 0) {
        unsigned int blocksNum = (verticesInBucket + lastBlockDimension.y - 1) / lastBlockDimension.y;
        // int sharedMemSize = THREADS_PER_BLOCK * (sizeof(int) + sizeof(float)) + lastBlockDimension.y * sizeof(float);
        // computeMoveGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
                // verticesInBucket, partition, lastBucketPrime,deviceStructures, hashCommunity, hashWeight);	// Manul change
		// computeCommunitiesSVLastBucket<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, primes_d, deviceStructures, hashCommunity[lastBucketNum]);
        // computeBestCommGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
        //         verticesInBucket, partition, primes_d, deviceStructures, hashCommunity, hashWeight, nodeEval, commEval);	// Manul change
		computeFinalNodeEval<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, deviceStructures, finalNodeEval);

		/***/
		// HANDLE_ERROR(cudaFree(hashCommunity));
        // HANDLE_ERROR(cudaFree(hashWeight));
		// HANDLE_ERROR(cudaFree(primes_d));
		/***/
		// HANDLE_ERROR(cudaFree(partition));
    }

	// Manul: partition unfreed
	// Manul: finalNodeEval unfreed

	return finalNodeEval;
}

nodeIndex* computeCommunities_gpu(device_structures& deviceStructures, vector<pair<unsigned int, unsigned int>>& newEdges, vector<int>& newEdgesIndex, vector<int>& edgesSV, vector<nodeIndex>& sourceVerticesNI, int *hashCommunitySV[]) {
	int svCount = sourceVerticesNI.size();
	
	int *newEdgesIndex_d, *edgesSV_d;
	HANDLE_ERROR(cudaMalloc((void**)&newEdgesIndex_d, newEdgesIndex.size() * sizeof(int)));	// free it Done
	HANDLE_ERROR(cudaMemcpy(newEdgesIndex_d, &newEdgesIndex[0], newEdgesIndex.size() * sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&edgesSV_d, edgesSV.size() * sizeof(int)));	// free it Done
	HANDLE_ERROR(cudaMemcpy(edgesSV_d, &edgesSV[0], edgesSV.size() * sizeof(int), cudaMemcpyHostToDevice));


    int lastBucketNum = bucketsSize - 2;
    dim3 lastBlockDimension = dims[lastBucketNum];
    auto predicate = isInBucketSV(buckets[lastBucketNum], buckets[lastBucketNum + 1], newEdgesIndex_d);		// Manul: TODO: check if it works

    nodeIndex *partition;
	HANDLE_ERROR(cudaMalloc((void**)&partition, svCount*sizeof(nodeIndex)));		// free it done
	// thrust::sequence(thrust::device, partition, partition + V, 0);	// ?
	HANDLE_ERROR(cudaMemcpy(partition, &sourceVerticesNI[0], svCount * sizeof(nodeIndex), cudaMemcpyHostToDevice));		

	cout << "comcom gpu part" << endl;
    
    nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);				//
    int verticesInLastBucket = thrust::distance(partition, deviceVerticesEnd);

	cout << "comcom gpu part done" << endl;
    
    cout << "Point 1M" << endl;

    int* primesSV_d;		// Manul change

	// assert(hashOffsetsSV.size() == svCount+1);

	// HANDLE_ERROR(cudaMalloc((void**)&primesSV_d, (svCount + 1) * sizeof(int)));		// free it
	// HANDLE_ERROR(cudaMemcpy(primesSV_d, &hashOffsetsSV[0], (svCount + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (verticesInLastBucket > 0) {

		/* Manul change start */
		nodeIndex* partition_h = (nodeIndex*) malloc(verticesInLastBucket * sizeof(nodeIndex));
		int* primes_h = (int*) malloc((verticesInLastBucket + 1) * sizeof(int));
		HANDLE_ERROR(cudaMemcpy(partition_h, partition, verticesInLastBucket * sizeof(nodeIndex), cudaMemcpyDeviceToHost));
		primes_h[0] = 0;
		for(int vi = 0; vi < verticesInLastBucket; vi++) {
			// int v = partition_h[vi].node;
			int v_ind = partition_h[vi].index;
			int degv = newEdgesIndex[v_ind+1] - newEdgesIndex[v_ind];
			primes_h[vi + 1] = primes_h[vi] + getPrime(degv * 1.5);
		}
		HANDLE_ERROR(cudaMalloc((void**)&primesSV_d, (verticesInLastBucket + 1) * sizeof(int)));		// free it Done
		HANDLE_ERROR(cudaMemcpy(primesSV_d, primes_h, (verticesInLastBucket + 1) * sizeof(int), cudaMemcpyHostToDevice));
		/* Manul change end */


		HANDLE_ERROR(cudaMalloc((void**)&hashCommunitySV[bucketsSize-2], primes_h[verticesInLastBucket] * sizeof(int)));		// Manul change free it done
        // HANDLE_ERROR(cudaMalloc((void**)&hashWeight, primes_h[verticesInLastBucket] * sizeof(float)));		// Manul change

		free(partition_h);
		free(primes_h);
    }
	

	cout << "Point 2M" << endl;

    for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
        dim3 blockDimension = dims[bucketNum];
        int prime = primes[bucketNum];
        auto predicate = isInBucketSV(buckets[bucketNum], buckets[bucketNum + 1], newEdgesIndex_d);
        deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);			//
        int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
        if (verticesInBucket > 0) {
			cout << " - bucketNum " << bucketNum << " " << verticesInBucket << endl;
            // int sharedMemSize =
            //         blockDimension.y * prime * (sizeof(float) + sizeof(int)) + blockDimension.y * sizeof(float);
            // if (blockDimension.x > WARP_SIZE)
            //     sharedMemSize += THREADS_PER_BLOCK * (sizeof(int) + sizeof(float));
            int blocksNum = (verticesInBucket + blockDimension.y - 1) / blockDimension.y;

			HANDLE_ERROR(cudaMalloc((void**)&hashCommunitySV[bucketNum], prime * verticesInBucket * sizeof(int)));		// Manul change free it done

			computeCommunitiesSVGeneral<<<blocksNum, blockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, prime, deviceStructures, hashCommunitySV[bucketNum]);


            // computeMoveShared<<<blocksNum, blockDimension, sharedMemSize>>>(verticesInBucket, partition, prime,
            //                                                                     deviceStructures);
            
			
			// // updating vertex -> community assignment
            // updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, partition,
            //                                                                     deviceStructures);
            // // updating community weight
            // thrust::fill(thrust::device, deviceStructures.communityWeight,
            //                 deviceStructures.communityWeight + hostStructures.V, (float) 0);
            // computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);
        }
    }

    cout << "Point 3M" << endl;


    // last bucket case
    deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);		//
    int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
    if (verticesInBucket > 0) {
        unsigned int blocksNum = (verticesInBucket + lastBlockDimension.y - 1) / lastBlockDimension.y;
        // int sharedMemSize = THREADS_PER_BLOCK * (sizeof(int) + sizeof(float)) + lastBlockDimension.y * sizeof(float);
        // computeMoveGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
                // verticesInBucket, partition, lastBucketPrime,deviceStructures, hashCommunity, hashWeight);	// Manul change
		computeCommunitiesSVLastBucket<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, primesSV_d, deviceStructures, hashCommunitySV[lastBucketNum]);
        // computeMoveGlobal2<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
        //         verticesInBucket, partition, primes_d, deviceStructures, hashCommunity, hashWeight);	// Manul change

		HANDLE_ERROR(cudaFree(primesSV_d));
    }

	HANDLE_ERROR(cudaFree(newEdgesIndex_d));
	HANDLE_ERROR(cudaFree(edgesSV_d));

	// Manul: hashCommunitySV[..] are still unfreed
	// Manul: partition is still unfreed 

	return partition;
}


/* 
 * @param newEdges      List of newly added edges, sorted by source vertices
 */
int nodeEval_add_gpu(device_structures& deviceStructures, host_structures& hostStructures, std::vector<pair<unsigned int, unsigned int>>& newEdges) {
	cout << "nodeEval_add_gpu starts" << endl;

	// remove
	printf("Init MM in nodeEval_add_gpu : %f\n", hostStructures.M);
	
	HANDLE_ERROR(cudaMemcpyToSymbol(MM, &hostStructures.M, sizeof(float)));

	// printf("Before modularity optimization:\n");
	// int* devVertexCommunity = (int*) malloc(sizeof(int));
	// for(int i = 0; i < hostStructures.V; i++) {
	// 	// printf("%d ", hostStructures.vertexCommunity[i]);
	// 	HANDLE_ERROR(cudaMemcpy(devVertexCommunity, (deviceStructures.vertexCommunity+i), sizeof(float), cudaMemcpyDeviceToHost));
	// 	printf("%d ", *devVertexCommunity);
	// }
	// printf("\n");
	
    vector<int> sourceVertices, newEdgesIndex, edgesSV;
	vector<nodeIndex> sourceVerticesNI;
    int prevVertex = -1;
	int count = 0;

    for (int i = 0; i < newEdges.size(); i++)
    {
        int vertex = newEdges[i].first;
        if(vertex != prevVertex) {
			sourceVertices.push_back(vertex);
            sourceVerticesNI.emplace_back(vertex, count++);
            
            prevVertex = vertex;
			// if(i > 0) {
			// 	int primeSV = getPrime((i - newEdgesIndex.back()) * 1.5);
			// 	primesSV.push_back(primeSV);
			// 	hashOffsetsSV.push_back(hashOffsetsSV.back() + primeSV);
			// }
			newEdgesIndex.push_back(i);
        }
		edgesSV.push_back(newEdges[i].second);
    }

	cout << "NEG 1" << endl;

	// int primeSV = getPrime((newEdges.size() - newEdgesIndex.back()) * 1.5);
	// primesSV.push_back(primeSV);
	// hashOffsetsSV.push_back(hashOffsetsSV.back() + primeSV);

    int svCount = sourceVerticesNI.size();
    newEdgesIndex.push_back(newEdges.size());

	int *hashCommunitySV[bucketsSize - 1];
	memset(hashCommunitySV, 0, sizeof(int*) * (bucketsSize-1));

	nodeIndex* partition = computeCommunities_gpu(deviceStructures, newEdges, newEdgesIndex, edgesSV, sourceVerticesNI, hashCommunitySV);

	int *nodeEval, *commEval;

	HANDLE_ERROR(cudaMalloc((void**)&nodeEval, hostStructures.V * sizeof(int))); // free it done
	HANDLE_ERROR(cudaMalloc((void**)&commEval, hostStructures.V * sizeof(int))); // free it done

	// Manul: TODO: These memsets are needed, right?
	HANDLE_ERROR(cudaMemset(nodeEval, 0, hostStructures.V * sizeof(int)));
	HANDLE_ERROR(cudaMemset(commEval, 0, hostStructures.V * sizeof(int)));

	cout << "NEG 3" << endl;


	int lastBucketNum = bucketsSize - 2;
    dim3 lastBlockDimension = dims[lastBucketNum];

	cout << "Before partition" << endl;

	auto predicate = isInBucket(buckets[lastBucketNum], buckets[lastBucketNum + 1], hostStructures.edgesIndex);		// Manul: TODO: check if it works


	nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);				//
    
	cout << "After partition" << endl;

	int verticesInLastBucket = thrust::distance(partition, deviceVerticesEnd);
    
    cout << "Point 1M" << endl;

    int* primes_d;		// Manul change
	int *hashCommunity;
	float *hashWeight;

	// assert(hashOffsetsSV.size() == svCount+1);

	// HANDLE_ERROR(cudaMalloc((void**)&primes_d, (svCount + 1) * sizeof(int)));		// free it
	// HANDLE_ERROR(cudaMemcpy(primes_d, &hashOffsetsSV[0], (svCount + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (verticesInLastBucket > 0) {

		/* Manul change start */
		nodeIndex* partition_h = (nodeIndex*) malloc(verticesInLastBucket * sizeof(nodeIndex));
		int* primes_h = (int*) malloc((verticesInLastBucket + 1) * sizeof(int));
		HANDLE_ERROR(cudaMemcpy(partition_h, partition, verticesInLastBucket * sizeof(nodeIndex), cudaMemcpyDeviceToHost));
		primes_h[0] = 0;
		for(int vi = 0; vi < verticesInLastBucket; vi++) {
			int v = partition_h[vi].node;
			// int v_ind = partition_h[vi].index;
			int degv = hostStructures.edgesIndex[v+1] - hostStructures.edgesIndex[v];
			primes_h[vi + 1] = primes_h[vi] + getPrime(degv * 1.5);
		}
		HANDLE_ERROR(cudaMalloc((void**)&primes_d, (verticesInLastBucket + 1) * sizeof(int)));		// free it done
		HANDLE_ERROR(cudaMemcpy(primes_d, primes_h, (verticesInLastBucket + 1) * sizeof(int), cudaMemcpyHostToDevice));
		/* Manul change end */


		HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, primes_h[verticesInLastBucket] * sizeof(int)));		// Manul change free it done
        HANDLE_ERROR(cudaMalloc((void**)&hashWeight, primes_h[verticesInLastBucket] * sizeof(float)));		// Manul change free it done

		free(partition_h);
		free(primes_h);
    }
	

	cout << "Point 2M" << endl;

    for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
        dim3 blockDimension = dims[bucketNum];
        int prime = primes[bucketNum];
        auto predicate = isInBucket(buckets[bucketNum], buckets[bucketNum + 1], hostStructures.edgesIndex);
        deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);			//
        int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
        if (verticesInBucket > 0) {
            int sharedMemSize =
                    blockDimension.y * prime * (sizeof(float) + sizeof(int)) + blockDimension.y * sizeof(float);
            if (blockDimension.x > WARP_SIZE)
                sharedMemSize += THREADS_PER_BLOCK * (sizeof(int) + sizeof(float));
            int blocksNum = (verticesInBucket + blockDimension.y - 1) / blockDimension.y;

			// HANDLE_ERROR(cudaMalloc((void**)&hashCommunity[bucketNum], prime * verticesInBucket * sizeof(int)));		// Manul change free it

			// computeCommunitiesSVGeneral<<<blocksNum, blockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, prime, deviceStructures, hashCommunity[bucketNum]);


            computeBestCommShared<<<blocksNum, blockDimension, sharedMemSize>>>(verticesInBucket, partition, prime, deviceStructures, nodeEval, commEval);

			
			// // updating vertex -> community assignment
            // updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, partition,
            //                                                                     deviceStructures);
            // // updating community weight
            // thrust::fill(thrust::device, deviceStructures.communityWeight,
            //                 deviceStructures.communityWeight + hostStructures.V, (float) 0);
            // computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);
        }
    }

    cout << "Point 3M" << endl;


    // last bucket case
    deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);		//
    int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
    if (verticesInBucket > 0) {
        unsigned int blocksNum = (verticesInBucket + lastBlockDimension.y - 1) / lastBlockDimension.y;
        int sharedMemSize = THREADS_PER_BLOCK * (sizeof(int) + sizeof(float)) + lastBlockDimension.y * sizeof(float);
        // computeMoveGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
                // verticesInBucket, partition, lastBucketPrime,deviceStructures, hashCommunity, hashWeight);	// Manul change
		// computeCommunitiesSVLastBucket<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, primes_d, deviceStructures, hashCommunity[lastBucketNum]);
        computeBestCommGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
                verticesInBucket, partition, primes_d, deviceStructures, hashCommunity, hashWeight, nodeEval, commEval);	// Manul change

		HANDLE_ERROR(cudaFree(hashCommunity));
        HANDLE_ERROR(cudaFree(hashWeight));
		HANDLE_ERROR(cudaFree(primes_d));
		// HANDLE_ERROR(cudaFree(partition));
    }

	for(int i = 0; i < bucketsSize - 1; i++) {
		if(hashCommunitySV[i]) {
			HANDLE_ERROR(cudaFree(hashCommunitySV[i]));
		}
	}

	int* finalNodeEval = computeFinalNodeEval_gpu(deviceStructures, hostStructures, sourceVerticesNI, nodeEval, partition);

	HANDLE_ERROR(cudaFree(partition));
	HANDLE_ERROR(cudaFree(nodeEval));


	int* R_array = deviceStructures.partition;
	// HANDLE_ERROR(cudaMalloc((void**)&R_array, hostStructures.V * sizeof(int))); // free it
	
	int* R_size;
	HANDLE_ERROR(cudaMalloc((void**)&R_size, sizeof(int))); // free it done
	HANDLE_ERROR(cudaMemset(R_size, 0, sizeof(int)));

	const int V_PER_BLOCK = 512;
	int blocksNum = (hostStructures.V + V_PER_BLOCK - 1) / V_PER_BLOCK;
	computeNodeEval<<<blocksNum, V_PER_BLOCK>>>(hostStructures.V, finalNodeEval, commEval, R_array, R_size, deviceStructures);

	HANDLE_ERROR(cudaFree(commEval));

	// int *nodeEval_h = (int*) malloc(hostStructures.V * sizeof(int));	// free it done
	// HANDLE_ERROR(cudaMemcpy(nodeEval_h, finalNodeEval, hostStructures.V * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(finalNodeEval));

	// R.clear();
	// for(int i = 0; i < hostStructures.V; i++) {
	// 	if(nodeEval_h[i] == 1) R.push_back(i); 
	// }

	// free(nodeEval_h);

	int R_size_h;
	HANDLE_ERROR(cudaMemcpy(&R_size_h, R_size, sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(R_size));

	// R_array unfreed

	// assert(R_size_h == R.size());

	// remove
	// int* R_gpu = (int*) malloc(R_size_h * sizeof(int));	// free it done

	// HANDLE_ERROR(cudaMemcpy(R_gpu, R_array, R_size_h * sizeof(int), cudaMemcpyDeviceToHost));

	// sort(R_gpu, R_gpu + R_size_h);
	// sort(R.begin(), R.end());

	// for(int i = 0; i < R_size_h; i++) { cout << R[i] << " " << R_gpu[i]; assert(R[i] == R_gpu[i]); }

	// free(R_gpu);

	// printf("After modularity optimization:\n");
	// for(int i = 0; i < hostStructures.V; i++) {
	// 	printf("%d ", hostStructures.vertexCommunity[i]);
	// }
	// printf("\n");

	cout << "Point 6M" << endl;

	return R_size_h;

}

__global__ void computeCommunitiesDelSV(int V, nodeIndex *vertices, int *edges, int *edgesIndex, int *commEval, int *nodeEval, device_structures deviceStructures) {
	int verticesPerBlock = blockDim.y;
	int vertexIndex = blockIdx.x * verticesPerBlock + threadIdx.y;
	if (vertexIndex < V) {
		int* vertexCommunity = deviceStructures.vertexCommunity;
		// extern __shared__ int s[];
		// auto *vertexToCurrentCommunity = (float *) s;
		// float *bestGains = &vertexToCurrentCommunity[verticesPerBlock];
		// int *bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];
		nodeIndex vertex = vertices[vertexIndex];
		int currentCommunity = vertexCommunity[vertex.node];
		// int prime = primes[vertex.index + 1] - primes[vertex.index];
		int concurrentNeighbours = blockDim.x;
		// int hashTablesOffset = threadIdx.y * prime;

		// if(threadIdx.x == 0) {
		// 	vertices[vertexIndex].prime = prime;
		// 	vertices[vertexIndex].hashCommunity = hashCommunity + hashTablesOffset;
		// }

		// hashWeight = hashWeight + primes[blockIdx.x];
		// computeMove(V, vertices, primes[blockIdx.x+1] - primes[blockIdx.x], deviceStructures, hashCommunity, hashWeight, vertexToCurrentCommunity,
		// 			bestGains, bestCommunities);
		// for (unsigned int i = threadIdx.x; i < prime; i += concurrentNeighbours) {
		// 	hashCommunity[hashTablesOffset + i] = -1;
		// }

		// Manul: TODO: check
		// if (concurrentNeighbours > WARP_SIZE)
		// 	__syncthreads();
		
		// insertCommunityInHashTable(hashCommunity, currentCommunity, prime, hashTablesOffset);

		int neighbourIndex = threadIdx.x + edgesIndex[vertex.index];
		int upperBound = edgesIndex[vertex.index + 1];

		while (neighbourIndex < upperBound) {
			int neighbour = edges[neighbourIndex];
			int community = vertexCommunity[neighbour];

			// if(currentCommunity != community) {
			// 	insertCommunityInHashTable(hashCommunity, community, prime, hashTablesOffset);
			// }

			if(currentCommunity == community) {
				commEval[currentCommunity] = 1;
				nodeEval[vertex.node] = 1;
			}
			neighbourIndex += concurrentNeighbours;
		}
	}
}


nodeIndex* computeCommunitiesDel_gpu(device_structures& deviceStructures, vector<pair<unsigned int, unsigned int>>& newEdges, vector<int>& newEdgesIndex, vector<int>& edgesSV, vector<nodeIndex>& sourceVerticesNI, int *commEval, int *nodeEval) {
	int svCount = sourceVerticesNI.size();
	
	int *newEdgesIndex_d, *edgesSV_d;
	HANDLE_ERROR(cudaMalloc((void**)&newEdgesIndex_d, newEdgesIndex.size() * sizeof(int)));	// free it Done
	HANDLE_ERROR(cudaMemcpy(newEdgesIndex_d, &newEdgesIndex[0], newEdgesIndex.size() * sizeof(int), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&edgesSV_d, edgesSV.size() * sizeof(int)));	// free it Done
	HANDLE_ERROR(cudaMemcpy(edgesSV_d, &edgesSV[0], edgesSV.size() * sizeof(int), cudaMemcpyHostToDevice));


    int lastBucketNum = bucketsSize - 2;
    dim3 lastBlockDimension = dims[lastBucketNum];
    auto predicate = isInBucketSV(buckets[lastBucketNum], buckets[lastBucketNum + 1], newEdgesIndex_d);		// Manul: TODO: check if it works

    nodeIndex *partition;
	HANDLE_ERROR(cudaMalloc((void**)&partition, svCount*sizeof(nodeIndex)));		// free it done
	// thrust::sequence(thrust::device, partition, partition + V, 0);	// ?
	HANDLE_ERROR(cudaMemcpy(partition, &sourceVerticesNI[0], svCount * sizeof(nodeIndex), cudaMemcpyHostToDevice));		

	cout << "comcom gpu part" << endl;
    
    // nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);				//
    // int verticesInLastBucket = thrust::distance(partition, deviceVerticesEnd);

	cout << "comcom gpu part done" << endl;
    
    cout << "Point 1M" << endl;

    // int* primesSV_d;		// Manul change

	// assert(hashOffsetsSV.size() == svCount+1);

	// HANDLE_ERROR(cudaMalloc((void**)&primesSV_d, (svCount + 1) * sizeof(int)));		// free it
	// HANDLE_ERROR(cudaMemcpy(primesSV_d, &hashOffsetsSV[0], (svCount + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // if (verticesInLastBucket > 0) {

	// 	/* Manul change start */
	// 	nodeIndex* partition_h = (nodeIndex*) malloc(verticesInLastBucket * sizeof(nodeIndex));
	// 	int* primes_h = (int*) malloc((verticesInLastBucket + 1) * sizeof(int));
	// 	HANDLE_ERROR(cudaMemcpy(partition_h, partition, verticesInLastBucket * sizeof(nodeIndex), cudaMemcpyDeviceToHost));
	// 	primes_h[0] = 0;
	// 	for(int vi = 0; vi < verticesInLastBucket; vi++) {
	// 		// int v = partition_h[vi].node;
	// 		int v_ind = partition_h[vi].index;
	// 		int degv = newEdgesIndex[v_ind+1] - newEdgesIndex[v_ind];
	// 		primes_h[vi + 1] = primes_h[vi] + getPrime(degv * 1.5);
	// 	}
	// 	HANDLE_ERROR(cudaMalloc((void**)&primesSV_d, (verticesInLastBucket + 1) * sizeof(int)));		// free it Done
	// 	HANDLE_ERROR(cudaMemcpy(primesSV_d, primes_h, (verticesInLastBucket + 1) * sizeof(int), cudaMemcpyHostToDevice));
	// 	/* Manul change end */


	// 	HANDLE_ERROR(cudaMalloc((void**)&hashCommunitySV[bucketsSize-2], primes_h[verticesInLastBucket] * sizeof(int)));		// Manul change free it done
    //     // HANDLE_ERROR(cudaMalloc((void**)&hashWeight, primes_h[verticesInLastBucket] * sizeof(float)));		// Manul change

	// 	free(partition_h);
	// 	free(primes_h);
    // }

	cout << "Point 2M" << endl;

    for(int bucketNum= 0; bucketNum < bucketsSize - 2; bucketNum++) {
        dim3 blockDimension = dims[bucketNum];
        // int prime = primes[bucketNum];
        auto predicate = isInBucketSV(buckets[bucketNum], buckets[bucketNum + 1], newEdgesIndex_d);
        nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);			//
        int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
        if (verticesInBucket > 0) {
			cout << " - bucketNum " << bucketNum << " " << verticesInBucket << endl;
            // int sharedMemSize =
            //         blockDimension.y * prime * (sizeof(float) + sizeof(int)) + blockDimension.y * sizeof(float);
            // if (blockDimension.x > WARP_SIZE)
            //     sharedMemSize += THREADS_PER_BLOCK * (sizeof(int) + sizeof(float));
            int blocksNum = (verticesInBucket + blockDimension.y - 1) / blockDimension.y;

			// HANDLE_ERROR(cudaMalloc((void**)&hashCommunitySV[bucketNum], prime * verticesInBucket * sizeof(int)));		// Manul change free it done

			// computeCommunitiesSVGeneral<<<blocksNum, blockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, prime, deviceStructures, hashCommunitySV[bucketNum]);

			computeCommunitiesDelSV<<<blocksNum, blockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, commEval, nodeEval, deviceStructures);

            // computeMoveShared<<<blocksNum, blockDimension, sharedMemSize>>>(verticesInBucket, partition, prime,
            //                                                                     deviceStructures);
            
			
			// // updating vertex -> community assignment
            // updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, partition,
            //                                                                     deviceStructures);
            // // updating community weight
            // thrust::fill(thrust::device, deviceStructures.communityWeight,
            //                 deviceStructures.communityWeight + hostStructures.V, (float) 0);
            // computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);
        }
    }

    cout << "Point 3M" << endl;


    // last bucket case
    nodeIndex *deviceVerticesEnd = thrust::partition(thrust::device, partition, partition + svCount, predicate);		//
    int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
    if (verticesInBucket > 0) {
        unsigned int blocksNum = (verticesInBucket + lastBlockDimension.y - 1) / lastBlockDimension.y;
        // int sharedMemSize = THREADS_PER_BLOCK * (sizeof(int) + sizeof(float)) + lastBlockDimension.y * sizeof(float);
        // computeMoveGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
                // verticesInBucket, partition, lastBucketPrime,deviceStructures, hashCommunity, hashWeight);	// Manul change
		// computeCommunitiesSVLastBucket<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, primesSV_d, deviceStructures, hashCommunitySV[lastBucketNum]);

		computeCommunitiesDelSV<<<blocksNum, lastBlockDimension>>>(verticesInBucket, partition, edgesSV_d, newEdgesIndex_d, commEval, nodeEval, deviceStructures);

        // computeMoveGlobal2<<<blocksNum, lastBlockDimension, sharedMemSize>>>(
        //         verticesInBucket, partition, primes_d, deviceStructures, hashCommunity, hashWeight);	// Manul change

		// HANDLE_ERROR(cudaFree(primesSV_d));
    }

	HANDLE_ERROR(cudaFree(newEdgesIndex_d));
	HANDLE_ERROR(cudaFree(edgesSV_d));

	// Manul: partition is still unfreed 

	return partition;
}


/* 
 * @param newEdges      List of newly added edges, sorted by source vertices
 */
int nodeEval_del_gpu(device_structures& deviceStructures, host_structures& hostStructures, std::vector<pair<unsigned int, unsigned int>>& newEdges) {
	cout << "nodeEval_del_gpu starts" << endl;

	// remove
	printf("Init MM in nodeEval_del_gpu : %f\n", hostStructures.M);
	
	HANDLE_ERROR(cudaMemcpyToSymbol(MM, &hostStructures.M, sizeof(float)));


	vector<int> sourceVertices, newEdgesIndex, edgesSV;
	vector<nodeIndex> sourceVerticesNI;
    int prevVertex = -1;
	int count = 0;

    for (int i = 0; i < newEdges.size(); i++)
    {
        int vertex = newEdges[i].first;
        if(vertex != prevVertex) {
			sourceVertices.push_back(vertex);
            sourceVerticesNI.emplace_back(vertex, count++);
            
            prevVertex = vertex;
			// if(i > 0) {
			// 	int primeSV = getPrime((i - newEdgesIndex.back()) * 1.5);
			// 	primesSV.push_back(primeSV);
			// 	hashOffsetsSV.push_back(hashOffsetsSV.back() + primeSV);
			// }
			newEdgesIndex.push_back(i);
        }
		edgesSV.push_back(newEdges[i].second);
    }

	cout << "NEG 1" << endl;

	// int primeSV = getPrime((newEdges.size() - newEdgesIndex.back()) * 1.5);
	// primesSV.push_back(primeSV);
	// hashOffsetsSV.push_back(hashOffsetsSV.back() + primeSV);

    int svCount = sourceVerticesNI.size();
    newEdgesIndex.push_back(newEdges.size());

	int *nodeEval, *commEval;

	HANDLE_ERROR(cudaMalloc((void**)&nodeEval, hostStructures.V * sizeof(int))); // free it 
	HANDLE_ERROR(cudaMalloc((void**)&commEval, hostStructures.V * sizeof(int))); // free it 

	// Manul: TODO: These memsets are needed, right?
	HANDLE_ERROR(cudaMemset(nodeEval, 0, hostStructures.V * sizeof(int)));
	HANDLE_ERROR(cudaMemset(commEval, 0, hostStructures.V * sizeof(int)));

	nodeIndex *partition = computeCommunitiesDel_gpu(deviceStructures, newEdges, newEdgesIndex, edgesSV, sourceVerticesNI, commEval, nodeEval);

	int *finalNodeEval = computeFinalNodeEval_gpu(deviceStructures, hostStructures, sourceVerticesNI, nodeEval, partition);

	HANDLE_ERROR(cudaFree(nodeEval));
	HANDLE_ERROR(cudaFree(partition));

	int* R_array = deviceStructures.partition;
	// HANDLE_ERROR(cudaMalloc((void**)&R_array, hostStructures.V * sizeof(int))); // free it
	
	int* R_size;
	HANDLE_ERROR(cudaMalloc((void**)&R_size, sizeof(int))); // free it done
	HANDLE_ERROR(cudaMemset(R_size, 0, sizeof(int)));

	const int V_PER_BLOCK = 512;
	int blocksNum = (hostStructures.V + V_PER_BLOCK - 1) / V_PER_BLOCK;
	computeNodeEval<<<blocksNum, V_PER_BLOCK>>>(hostStructures.V, finalNodeEval, commEval, R_array, R_size, deviceStructures);

	HANDLE_ERROR(cudaFree(commEval));

	// int *nodeEval_h = (int*) malloc(hostStructures.V * sizeof(int));	// free it done
	// HANDLE_ERROR(cudaMemcpy(nodeEval_h, finalNodeEval, hostStructures.V * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(finalNodeEval));

	// R.clear();
	// for(int i = 0; i < hostStructures.V; i++) {
	// 	if(nodeEval_h[i] == 1) R.push_back(i); 
	// }

	// free(nodeEval_h);

	int R_size_h;
	HANDLE_ERROR(cudaMemcpy(&R_size_h, R_size, sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(R_size));

	// R_array unfreed

	// assert(R_size_h == R.size());

	// remove
	// int* R_gpu = (int*) malloc(R_size_h * sizeof(int));	// free it done

	// HANDLE_ERROR(cudaMemcpy(R_gpu, R_array, R_size_h * sizeof(int), cudaMemcpyDeviceToHost));

	// sort(R_gpu, R_gpu + R_size_h);
	// sort(R.begin(), R.end());

	// for(int i = 0; i < R_size_h; i++) { cout << R[i] << " " << R_gpu[i]; assert(R[i] == R_gpu[i]); }

	// free(R_gpu);

	return R_size_h;

}