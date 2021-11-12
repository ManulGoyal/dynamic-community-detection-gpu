// File: modularity.h
// -- quality functions (for Modularity criterion) header file
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



#ifndef MODULARITY_H
#define MODULARITY_H

#include "quality.h"

using namespace std;


class Modularity: public Quality {
 public:

  // Manul: for the i-th community, in[i] = sum of e_(k->C(k)) for all nodes k in i
  // Manul: and tot[i] = a_i (where a_i is sum of weighted degrees of all nodes in i)
  vector<long double> in, tot; // used to compute the quality participation of each community

  Modularity(Graph & gr);
  ~Modularity();

  inline void remove(int node, int comm, long double dnodecomm);

  inline void insert(int node, int comm, long double dnodecomm);

  inline long double gain(int node, int comm, long double dnodecomm, long double w_degree);

  long double quality();
};


inline void
Modularity::remove(int node, int comm, long double dnodecomm) {//dnodecomm: weigth of links between node and comm
  assert(node>=0 && node<size);

  // Manul: here 2 is multiplied with dnodecomm which seems correct although in paper
  // Manul: it is not there
  in[comm]  -= 2.0L*dnodecomm + g.nb_selfloops(node);
  tot[comm] -= g.weighted_degree(node);
  
  n2c[node] = -1;
}

inline void
Modularity::insert(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);
  
  in[comm]  += 2.0L*dnodecomm + g.nb_selfloops(node);
  tot[comm] += g.weighted_degree(node);
  
  n2c[node] = comm;
}

inline long double
Modularity::gain(int node, int comm, long double dnc, long double degc) {//dnc:degree of comm, degc:weighted degree of node
 // cerr<<"node: "<<node<<endl<<"size: "<<size<<endl;
  assert(node>=0 && node<size);
  //cerr<<"node: "<< node <<" comm = "<<comm<<" neigh_weight: "<<dnc<<" weighted_degree: "<<degc<<endl;

  long double totc = tot[comm];
  long double m2   = g.total_weight;
  
  // Manul: dnc is sum of weights of edges from node to vertices in comm (same as dnodecomm)
  // Manul: Although the actual expression should have been:
  // Manul: gain = (2*dnc + self_loops_weights_sum(node))/m2 - (2*totc*degc + degc^2)/(m2^2)
  // Manul: but while local maximisation of gain, the constant terms can be dropped
  // Manul: basically we can compare [(gain * m2 - self_loops_weights_sum(node) + degc^2/m2]/2
  // Manul: which is equivalent to dnc - totc*degc/m2
  // Manul: Note that this value CAN'T be used for actual updation of modularity

  return (dnc - totc*degc/m2);
}


#endif // MODULARITY_H
