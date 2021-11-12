// File: zahn.h
// -- quality functions (for Zahn-Condorcet criterion) header file
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



#ifndef ZAHN_H
#define ZAHN_H

#include "quality.h"

using namespace std;


class Zahn: public Quality {
 public:
  
  // used to compute the quality participation of each community
  vector<long double> in;
  vector<int> w;
  long double max; // biggest weight on links

  Zahn(Graph & gr, long double max_w);
  ~Zahn();

  inline void remove(int node, int comm, long double dnodecomm);

  inline void insert(int node, int comm, long double dnodecomm);

  inline long double gain(int node, int comm, long double dnodecomm, long double w_degree);

  long double quality();
};


inline void
Zahn::remove(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);

  in[comm] -= 2.0L*dnodecomm + g.nb_selfloops(node);
  w[comm]  -= g.nodes_w[node];
  
  n2c[node] = -1;
}

inline void
Zahn::insert(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);
  
  in[comm] += 2.0L*dnodecomm + g.nb_selfloops(node);
  w[comm]  += g.nodes_w[node];
  
  n2c[node] = comm;
}

inline long double
Zahn::gain(int node, int comm, long double dnc, long double degc) {
  assert(node>=0 && node<size);
  
  long double wc = (long double)w[comm];
  long double wu = (long double)g.nodes_w[node];
  
  long double gain = 2.0L*dnc - wu*wc*max;

  return gain;
}


#endif // ZAHN_H
