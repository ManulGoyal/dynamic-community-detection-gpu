// File: shimalik.h
// -- quality functions (for Shi-Malik criterion) header file
//-----------------------------------------------------------------------------
// Delta-screening, dynamic community detection 
//
// This work is an extension of the Louvain implementation 
// for static community detection. The change incorporates
// the Delta-screening technique that selects subsets of 
// vertices to process at each iteration.
//
// The Louvain implementation for static community detection is 
// based on the article 
// "Fast unfolding of community hierarchies in large networks"
// Copyright (C) 2008 V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre
//
// And based on the article 
// "A Generalized and Adaptive Method for Community Detection"
// Copyright (C) 2014 R. Campigotto, P. Conde Céspedes, J.-L. Guillaume
//
// The Delta-screening technique for dynamic community detection 
// is based on the article 
// "A fast and efficient incremental approach toward dynamic community detection" 
// Copyright (C) 2019 N. Zarayeneh, A. Kalyanaraman
// Proc. IEEE/ACM International Conference on Advances in Social 
// Networks Analysis and Mining, pp. 9-16, 2019.

// This file is part of Louvain algorithm.

// Louvain algorithm is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Louvain algorithm is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Louvain algorithm.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------
// see README.txt for more details



#ifndef SHIMALIK_H
#define SHIMALIK_H

#include "quality.h"

using namespace std;


class ShiMalik: public Quality {
 public:

  vector<long double> in, tot; // used to compute the quality participation of each community
  int kappa; // number of communities

  int kmin;

  ShiMalik(Graph & gr, int kappa_min);
  ~ShiMalik();

  inline void remove(int node, int comm, long double dnodecomm);

  inline void insert(int node, int comm, long double dnodecomm);

  inline long double gain(int node, int comm, long double dnodecomm, long double w_degree);

  long double quality();
};


inline void
ShiMalik::remove(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);

  in[comm]  -= 2.0L*dnodecomm + g.nb_selfloops(node);
  tot[comm] -= g.weighted_degree(node);
  
  if (tot[comm] == 0.0L)
    kappa--;

  n2c[node] = -1;
}

inline void
ShiMalik::insert(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);

  in[comm] += 2.0L*dnodecomm + g.nb_selfloops(node);

  if (tot[comm] == 0.0L)
    kappa++;
  
  tot[comm] += g.weighted_degree(node);

  n2c[node] = comm;
}

inline long double
ShiMalik::gain(int node, int comm, long double dnc, long double degc) {
  assert(node>=0 && node<size);

  long double inc  = in[comm];
  long double totc = tot[comm];
  long double self = g.nb_selfloops(node);
  
  long double gain;
  
  if (totc == 0.0L) {
    gain  = (2.0L*dnc+self) / degc;
    gain -= 1.0L;
  }
  else {
    gain  = (inc + 2.0L*dnc+self) / (totc+degc);
    gain -= inc/totc;
  }

  if (kappa < kmin) return 0.0L;
  else return gain;
}


#endif // SHIMALIK_H
