/**** Added by Manul ****/

#ifndef __GPULOUVAIN__CUH__
#define __GPULOUVAIN__CUH__

#include "../louvain.h"
#include "utils.cuh"

class GPUWrapper {
	host_structures hostStructures;
	device_structures deviceStructures;
    aggregation_phase_structures aggregationPhaseStructures;
	bool initPart;
	Louvain* c;
	float memoryTime;
	int R_size;

public:

    GPUWrapper(Louvain* c, std::vector<int>& n2c, bool initPart);

	int gpuNodeEvalAdd(vector<pair<unsigned int, unsigned int>>& newEdges);

	int gpuNodeEvalDel(vector<pair<unsigned int, unsigned int>>& newEdges);

    void gpuLouvain(std::vector<int>& n2c);
};

// void gpuLouvain(Louvain* c, vector<int>& n2c, bool initPart = false);

#endif /* __GPULOUVAIN__CUH__ */