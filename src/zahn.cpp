// File: zahn.cpp
// -- quality functions (for Zahn-Condorcet criterion) source file
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



#include "zahn.h"

using namespace std;


Zahn::Zahn(Graph & gr, long double max_w): Quality(gr,"Zahn-Condorcet"), max(max_w) {  
  n2c.resize(size);

  in.resize(size);
  w.resize(size);


  // initialization
  for (int i=0 ; i<size ; i++) {
    n2c[i] = i;
    in[i]  = g.nb_selfloops(i);
    w[i]   = g.nodes_w[i];
  }
}

Zahn::~Zahn() {
  in.clear();
  w.clear();
}

long double
Zahn::quality() {
  long double q  = 0.0L;
  long double m2 = g.total_weight;
  long double n  = (long double)g.sum_nodes_w;

  for (int i=0 ; i < size ; i++) {
    long double wc = (long double)w[i];
    if (wc > 0.0L)
      q += 2.0L*in[i] - max*wc*wc;
  }
  
  q += n*n*max - m2;
  
  q /= n*n*max;
  
  return q;
}
