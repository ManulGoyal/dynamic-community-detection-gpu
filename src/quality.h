// File: quality.h
// -- quality functions header file
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



#ifndef QUALITY_H
#define QUALITY_H

#include <sstream>

#include "graph_binary.h"

using namespace std;


class Quality {
 public:
  
  Graph & g; // network to compute communities for
  int size; // nummber of nodes in the network and size of all vectors
  string name;
  
  vector<int> n2c; // community to which each node belongs
  /*nzarayeneh 1st change*/
  //vector<vector<int> > nod_com;//store nodes for each comm
  // Manul: set of nodes which should be processed by the passes of Louvain algorithm
  // Manul: this set is computed by delta-screening  
  vector <int> R;
 Quality(Graph &gr, const std::string& n):g(gr),size(g.nb_nodes),name(n){R.resize(g.nb_nodes);}
 /*nzarayeneh end 1st change*/
  virtual ~Quality();
  
  // remove the node from its current community with which it has dnodecomm links
  virtual void remove(int node, int comm, long double dnodecomm)=0;
  
  // insert the node in comm with which it shares dnodecomm links
  virtual void insert(int node, int comm, long double dnodecomm)=0;
  
  // compute the gain of quality by adding node to comm
  virtual long double gain(int node, int comm, long double dnodecomm, long double w_degree)=0;
  
  // compute the quality of the current partition
  virtual long double quality()=0;
};

template<class T>
std::string to_string(T number){
  std::ostringstream oss;
  oss << number;
  return oss.str();
}

#endif // QUALITY_H
