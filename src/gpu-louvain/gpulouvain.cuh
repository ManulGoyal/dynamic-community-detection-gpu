/**** Added by Manul ****/

#ifndef __GPULOUVAIN__CUH__
#define __GPULOUVAIN__CUH__

#include "../louvain.h"

void gpuLouvain(Louvain* c, vector<int>& n2c, bool initPart = false);

#endif /* __GPULOUVAIN__CUH__ */