// File: modularity.cpp
// -- quality functions (for Newman-Girvan Modularity criterion) source file
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



#include "modularity.h"

using namespace std;


Modularity::Modularity(Graph & gr):Quality(gr,"Newman-Girvan Modularity") {
//cerr<<"Debug...start of modularity"<<endl;
  n2c.resize(size);
//cerr<<"Debug...size is: "<<size<<endl;
  in.resize(size);
  tot.resize(size);
  
  // initialization
  for (int i=0 ; i<size ; i++) {
    n2c[i] = i;
    in[i]  = g.nb_selfloops(i);
    tot[i] = g.weighted_degree(i);
  }
}

Modularity::~Modularity() {
  in.clear();
  tot.clear();
}

long double
Modularity::quality() {
	//cerr<<"Debug...start of modularity"<<endl;
  long double q  = 0.0L;

  // Manul-Manan: g.total_weight is twice the sum of weights of all edges in graph
  long double m2 = g.total_weight;

  // Manul-Manan: q = (1/m)*sum_over_all_communities_i(in[i] - tot[i]*tot[i] / m)

  for (int i=0 ; i<size ; i++) {
    // Manul-Manan: since 'size' is the number of nodes, tot[i] > 0.0L is used to check whether i
    // Manul-Manan: represents a community or not (since not all i in [0, size) represent a comm)
    if (tot[i] > 0.0L)
      q += in[i] - (tot[i]*tot[i]) / m2;
  }

  q /= m2;

  return q;
}
