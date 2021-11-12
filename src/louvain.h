// File: louvain.h
// -- community detection header file
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



#ifndef LOUVAIN_H
#define LOUVAIN_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>


#include "graph_binary.h"
#include "quality.h"

using namespace std;



class Louvain {
 public:
  vector<long double> neigh_weight;//the weight of each community
  vector<int> neigh_pos;//
  int neigh_last;

  // number of pass for one level computation
  // if -1, compute as many pass as needed to increase quality
  int nb_pass;

  // a new pass is computed if the last one has generated an increase 
  // better than eps_impr
  // if 0.0L even a minor increase is enough to go for one more pass
  long double eps_impr;
  
  // Quality functions used to compute communities
  Quality* qual;


  // constructors:
  // reads graph from file using graph constructor
  // type defined the weighted/unweighted status of the graph file
  Louvain (int nb_pass, long double eps_impr, Quality* q);

  // initiliazes the partition with something else than all nodes alone
  void init_partition(char *filename_part);
  void init_partition_v(vector<int> v);

  // compute the set of neighboring communities of node
  // for each community, gives the number of links from node to comm
  void neigh_comm(int node);

  // displays the graph of communities as computed by one_level
  void partition2graph();

  // displays the current partition (with communities renumbered from 0 to k-1)
  void display_partition(int count);
  void display_partition1();

  // generates the binary graph of communities as computed by one_level
  Graph partition2graph_binary();

  // compute communities of the graph for one level
  // return true if some nodes have been moved
  bool one_level();
};


#endif // LOUVAIN_H
