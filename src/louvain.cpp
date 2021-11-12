// File: louvain.cpp
// -- community detection source file
//-----------------------------------------------------------------------------
// Delta-screening, dynamic community detection 
//
//This work is an extension of the Louvain implementation for static community detection and the change incorporate Delta-screening method
//The Louvain implementation for static community detection is based on the article "Fast unfolding of community hierarchies in large networks"
// Copyright (C) 2008 V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre
//
// And based on the article "A Generalized and Adaptive Method for Community Detection"
// Copyright (C) 2014 R. Campigotto, P. Conde Céspedes, J.-L. Guillaume
//
// This code has been modified to incorporate the Delta-screening technique, and the Delta-screening approach is based on  
//Zarayeneh, Neda, and Ananth Kalyanaraman. "A fast and efficient incremental approach toward dynamic community detection." 
//Proceedings of the 2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining. 2019.

//This file is part of Louvain algorithm.

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
#include <limits>
#include <sstream>
#include <iomanip>
#include "louvain.h"
#include <sys/time.h>
#include <ctime>
#include <iostream>


using namespace std;

static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000;
}
namespace my{
std::string to_string(int d)
{
	std::ostringstream stm;
	stm <<std::setprecision(numeric_limits<int>::digits10) <<d;
	return stm.str();
}
}

Louvain::Louvain(int nbp, long double epsq, Quality* q) {
  qual = q;

  neigh_weight.resize(qual->size,-1);//the #comm are at most #nodes so iniit with size of nodes and assign -1 for wiegth of each comm
  neigh_pos.resize(qual->size);//each node can have several comm  as neighbor and one node can be neighbor with all comm so its a vector that sav pos of neighbor comm for a node
  neigh_last = 0;

  nb_pass = nbp;
  eps_impr = epsq;
}

void
Louvain::init_partition(char * filename) //it gets the file containing nodes and comms and use this partition instead of trivial partition
{
  ifstream finput;
  finput.open(filename,fstream::in);

  // read partition
  while (!finput.eof()) 
  {
    int node, comm;
    finput >> node >> comm;
    
    if (finput) 
    {
      int old_comm = qual->n2c[node];
      //cerr<<"old_comm: "<<old_comm<<endl;
      neigh_comm(node);
     // cerr<<"neigh_weight["<<node<<"]: "<<neigh_weight[node]<<endl;

      // Manul: neigh_weight[old_comm] holds the sum of edge weights from node to other nodes
      // Manul: in the comm. in which node belongs
      qual->remove(node, old_comm, neigh_weight[old_comm]);
      

      // Manul: this loop checks whether comm is one of the nbring comm's of node
      
      int i=0;
      for (i=0 ; i<neigh_last ; i++) 
      {
	      int best_comm = neigh_pos[i];
	     // cerr<<"best_comm: "<<best_comm<<endl;

      	long double best_nblinks = neigh_weight[neigh_pos[i]];
      	//cerr<<"best_nblinks: "<<best_nblinks<<endl;
	      if (best_comm==comm) 
        {
	        qual->insert(node, best_comm, best_nblinks);
	        break;
	      }
     }
     if (i==neigh_last)
	   qual->insert(node, comm, 0);
    }
  }
  finput.close();
}

// Manul: initialize communities such that the i-th node belongs to comm no. v[i]
void
Louvain::init_partition_v(vector<int> v) //it gets the file containing nodes and comms and use this partition instead of trivial partition
{

  for(int k=0;k<v.size();k++)
  {
      int old_comm = qual->n2c[k];
     // cerr<<"old_comm: "<<old_comm<<endl;
      neigh_comm(k);
      //cerr<<"neigh_weight["<<k<<"]: "<<neigh_weight[k]<<endl;

      qual->remove(k, old_comm, neigh_weight[old_comm]);

      int i=0;
      for (i=0 ; i<neigh_last ; i++)
      {
	      int best_comm = neigh_pos[i];
	     //cerr<<"best_comm: "<<best_comm<<endl;
      	long double best_nblinks = neigh_weight[neigh_pos[i]];
      	//cerr<<"best_nblinks: "<<best_nblinks<<endl;
      	//cerr<<"best_nblinks: "<<best_nblinks<<endl;
	    if(best_comm==v[k])
         {
	        qual->insert(k, best_comm, best_nblinks);
	        break;
	      }
     }
     if (i==neigh_last)
	   qual->insert(k, v[k], 0);
    }
    /*for(int k=0;k<v.size();k++)
    {
      cerr<<"v["<<k<<"] in init_partition_v= "<<v[k]<<endl;
}*/
  }


// Manul: calculates the sum of edge weights from node to each neighbouring community of node
//gives the "neighbor_pos" for name of neighbr comm and "neigh_weight" for the weight of these comm  
void
Louvain::neigh_comm(int node) {//node has #neigh_last comm neighbors

  // Manul: neigh_last is the current number of nbring comm's of node (including comm of node itself)
  // Manul: neigh_pos[i] is the community number of the i-th nbring comm of node
  // Manul: neigh_weight[k] is the sum of edge weights from node to community number k

 // cerr<<"neigh_last: "<<neigh_last<<endl;
  // Manul: This loop resets the weights calculated during the last invocation of this function
  // Manul: to -1, so that weights can be recalculated for the current 'node'
  for (int i=0 ; i<neigh_last ; i++)// assign -1 to weight of comm for all neighbor comm of node
    neigh_weight[neigh_pos[i]]=-1;
  
  neigh_last = 0;

  pair<vector<int>::iterator, vector<long double>::iterator> p = (qual->g).neighbors(node);//
  int deg = (qual->g).nb_neighbors(node);

  // Manul: 0-th nbring comm of node is the comm in which the node belongs
  // Manul: neigh_weight[0] does not include weight of self loop on node

  neigh_pos[0] = qual->n2c[node];//current comm node is in 
  neigh_weight[neigh_pos[0]] = 0;//assign zero to current comm of node 
  neigh_last = 1;// consider one neighbor comm for node 

  for (int i=0 ; i<deg ; i++) {
    int neigh  = *(p.first+i);//ith nighbor of node
    int neigh_comm = qual->n2c[neigh];//ith neighboe comm

    // Manul: weight of edge from node to its i-th neighbor
    long double neigh_w = ((qual->g).weights.size()==0)?1.0L:*(p.second+i);//ith neighbor comm weight
    
    if (neigh!=node) {;
      if (neigh_weight[neigh_comm]==-1) //at first each node is a community with weigh of -1, so if neigh_weight==-1 means just one node
      {
	      neigh_weight[neigh_comm] = 0.0L;
	      neigh_pos[neigh_last++] = neigh_comm;
      }
      neigh_weight[neigh_comm] += neigh_w;
      //cerr<<" neigh_weight["<<neigh_comm<<"]: "<<neigh_weight[neigh_comm]<<endl;
    }
  }

  // Manul: neigh_pos is now of the size neigh_last, and holds the numbers of all the nbring
  // Manul: comm's of node, including the comm of node itself (at index 0)
}

// Manul: deletes empty comm's and renumbers them as 0, 1, ... and then displays the graph of renumbered comm's (the renumbering is not permanent yet, done only for display purposes) 
//after pass 1 weneed to remove comm with zero nodes inside and renumber he comms, this function will do that 
//this will print all neighbor comms 
void
Louvain::partition2graph() 
{
  vector<int> renumber(qual->size, -1);//consider a vec of size nodes and assign -1 to all
  for (int node=0 ; node<qual->size ; node++) 
  {
    renumber[qual->n2c[node]]++;//for each node find its comm let say its j and increament jth pos in renumber vec by one so renumber has comms with number of nodes inside([1,5,6]
                                  //means c0 has 1 node, c1 has 5 nodes and c2 has 6 nodes in this way we want to remove comm with zero elements
  }

  int end=0;
  for (int i=0 ; i< qual->size ; i++)
    if (renumber[i]!=-1)//its a comm with nodes inside
      renumber[i]=end++;//renumber nonzero comms from 1 

  for (int i=0 ; i< qual->size ; i++) 
  {
    pair<vector<int>::iterator, vector<long double>::iterator> p = (qual->g).neighbors(i);

    int deg = (qual->g).nb_neighbors(i);
    for (int j=0 ; j<deg ; j++) 
    {
      int neigh = *(p.first+j);
      cout << renumber[qual->n2c[i]] << " " << renumber[qual->n2c[neigh]] << endl;// now comm of node i might be different ex: before it was 4 now its 3, also comm of jth neighbor of i is diff
    }
  }
}

// Manul: deletes empty comm's and renumbers them as 0, 1, ... and then outputs to file the renumbered comm
// Manul: number for each node (the renumbering is not permanent yet, done only for output to file purposes) 
//sane as partition2graph just print each node and its new name comm(delete those partition that have no node anymore, and renumber comms
void
Louvain::display_partition(int count)
{
 // cerr<<"qual->size inside display_partition: "<<qual->size<<endl; 
  vector<int> renumber(qual->size, -1);
  for(int node=0 ; node < qual->size ; node++) 
  {
    renumber[qual->n2c[node]]++;
  }

  int end=0;
  for (int i=0 ; i < qual->size ; i++)
    if (renumber[i]!=-1)
      renumber[i] = end++;
  //create name
  string name = "graph" + my::to_string(count)+".tree";//c++11 for std::to_string
  ofstream ofile;
  ofile.open(name.c_str(),fstream::app);//in c++11 without .c_str()
  for (int i=0 ; i < qual->size ; i++)
    ofile << i << " " << renumber[qual->n2c[i]] << endl;
  ofile.close();
}

// Manul: deletes empty comm's and renumbers them as 0, 1, ... and then creates a graph of renumbered comm's
Graph
Louvain::partition2graph_binary() //this is for making new graph for new pass
{
  // Renumber communities
  vector<int> renumber(qual->size, -1);
  for (int node=0 ; node < qual->size ; node++)
    renumber[qual->n2c[node]]++;

  int last=0;
  for (int i=0 ; i < qual->size ; i++) 
  {
    if (renumber[i]!=-1)
      renumber[i] = last++;//last is number of comm that we have after pass
  }

  // Compute communities
  vector<vector<int> > comm_nodes(last);//2dim array first vec is comms for each comm a vec of nodes inside it
  vector<int> comm_weight(last, 0);
  
  for (int node = 0 ; node < (qual->size) ; node++) {
    comm_nodes[renumber[qual->n2c[node]]].push_back(node);
    comm_weight[renumber[qual->n2c[node]]] += (qual->g).nodes_w[node];
  }

  // Compute weighted graph
  Graph g2;
  int nbc = comm_nodes.size();//# of comm

  g2.nb_nodes = comm_nodes.size();//grapg for sencond pass the # of nodes are # of comm
  g2.degrees.resize(nbc);
  g2.nodes_w.resize(nbc);
  
  for (int comm=0 ; comm<nbc ; comm++) {
    map<int,long double> m;
    map<int,long double>::iterator it;

    int size_c = comm_nodes[comm].size();

    // Manul: each comm (a node in the new graph) has weight equal to count of original nodes in the comm
    // Manul: for ex, 1st level comm C1 (2 nodes) may be moved to comm of 1st level comm C2 (3 nodes), the resulting
    // Manul: 2nd level comm will have 2 1st level comm's C1 and C2, and its weight will be 2 + 3 = 5
    g2.assign_weight(comm, comm_weight[comm]);//each comm is a node now assign weight of comm to comm as a node to the  new graph 

    for (int node=0 ; node<size_c ; node++) 
    {//for each node in each comm find its neghbors and determine its neighbors for each neighbor check if its comm is in m (its a map) if its not make a pair in m firs element 
      //the comm and sencond the weight of this neigbor, if the comm is already in m just add the weight of this neighbor node
      pair<vector<int>::iterator, vector<long double>::iterator> p = (qual->g).neighbors(comm_nodes[comm][node]); 
      int deg = (qual->g).nb_neighbors(comm_nodes[comm][node]);
      for (int i=0 ; i<deg ; i++) 
      {
	      int neigh = *(p.first+i);
	      int neigh_comm = renumber[qual->n2c[neigh]];
	      long double neigh_weight = ((qual->g).weights.size()==0)?1.0L:*(p.second+i);

	      it = m.find(neigh_comm);//check if neigh_comm is already added to m
	      if (it==m.end())
	        m.insert(make_pair(neigh_comm, neigh_weight));// if not added add it
	      else
	        it->second += neigh_weight;// iff added jus add the neigh_weight to the second pos of pair 
      }
    }

    g2.degrees[comm] = (comm==0)?m.size():g2.degrees[comm-1]+m.size();
    g2.nb_links += m.size();

    for (it = m.begin() ; it!=m.end() ; it++) {
      g2.total_weight += it->second;
      g2.links.push_back(it->first);
      g2.weights.push_back(it->second);
    }
  }

  return g2;
}

bool
Louvain::one_level() {
  bool improvement=false ;
  int nb_moves;
  int nb_pass_done = 0;
  long double new_qual = qual->quality();
  long double cur_qual = new_qual;

  vector<int> random_order;
  /*nzarayeneh 1st chamge*/
  //cerr<< endl<<"qual->R.size()= "<<qual->R.size()<<endl<< "qual->size= "<< qual->size <<endl;

  // Manul: if the set of nodes to be processed (R) contains all nodes, shuffle the order in which the
  // Manul: nodes should be processed, otherwise, the nodes should be processed according to R
  if(qual->size == qual->R.size())
  {
	  random_order.resize(qual->size);
	  for (int i=0 ; i < qual->size ; i++)
		random_order[i]=i;
	  for (int i=0 ; i < qual->size-1 ; i++)
	  {
      // Manul: pick a random number j from the range [i, qual->size-1], and swap i-th and j-th elements
		int rand_pos = rand()%(qual->size-i)+i;
		int tmp = random_order[i];
		random_order[i] = random_order[rand_pos];
		random_order[rand_pos] = tmp;
	  }
    // Manul: now random_order contains a random permutation of [0..qual->size-1] (all nodes)
  }

  else
  {
	  //cerr<<"qual->R.size(): "<<qual->R.size()<<endl;
	  random_order.resize(qual->R.size());
	  random_order = qual->R;
	  //for(size_t i=0 ; i < qual->R.size() ; i++)
	  		//random_order[i]=qual->R[i];
	  //random_shuffle(random_order.begin(),   random_order.end());
  }
  /*nzarayeneh end 1st chamge*/
 // double sTime,eTime;

  // repeat while 
  //   there is an improvement of quality
  //   or there is an improvement of quality greater than a given epsilon 
  //   or a predefined number of pass have been done
  int n_iter=0;
  double tt=0.0;
  do {
    n_iter++;

    cur_qual = new_qual;
    nb_moves = 0;
    nb_pass_done++;

    /*nzarayeneh 2nd change*/
    /*timespec start;
    timespec end;
    double elapsedSeconds;
    clock_gettime(CLOCK_MONOTONIC, &start);*/
    struct timeval start, end;

    gettimeofday(&start, NULL);
    // for each node: remove the node from its community and insert it in the best community
    for(int node_tmp = 0 ; node_tmp < qual->R.size() ; node_tmp++) {
    /*nzarayeneh end 2nd change*/
      int node = random_order[node_tmp];
      int node_comm = qual->n2c[node];
     // cerr<<endl;
      long double w_degree = (qual->g).weighted_degree(node);

      // computation of all neighboring communities of current node
      neigh_comm(node);
      // remove node from its current community
      qual->remove(node, node_comm, neigh_weight[node_comm]);

      // compute the nearest community for node
      // default choice for future insertion is the former community
      int best_comm = node_comm;
      long double best_nblinks  = 0.0L;
      long double best_increase = 0.0L;
      for (int i=0 ; i<neigh_last ; i++)
      {
    	//  cerr<<"neigh_weight[neigh_pos["<<i<<"]]"<<neigh_weight[neigh_pos[i]]<<endl;
    	 // cerr<<"w_degree: "<<w_degree<<endl;
		long double increase = qual->gain(node, neigh_pos[i], neigh_weight[neigh_pos[i]], w_degree);
		//cerr<<"qual->gain: "<<increase<<endl;
		if (increase>best_increase) 
		{
		  best_comm = neigh_pos[i];
		  best_nblinks = neigh_weight[neigh_pos[i]];
		  best_increase = increase;
		}
      }

      // insert node in the nearest community
      qual->insert(node, best_comm, best_nblinks);
     
      if (best_comm!=node_comm)
	nb_moves++;
    }

    new_qual = qual->quality();
    //cerr<<"Debug...new_qual in louvain: "<<new_qual<<endl;
    /*nzarayeneh*/
    //cerr<<"\t iteration "<<nb_pass_done<<endl;
    /*clock_gettime(CLOCK_MONOTONIC, &end);
    elapsedSeconds = TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start);
    cerr<<"\t Time: "<<elapsedSeconds<<endl;*/
    gettimeofday(&end, NULL);
    tt += (double) ((end.tv_sec -start.tv_sec)*1000 + (end.tv_usec- start.tv_usec)/1000) ;
   // cerr<<"\t Time:(milliseconds) "<<tt<<endl; 
   // cerr<<"\t quality: "<<new_qual<<endl;
    /*nzarayeneh*/
    
    if (nb_moves>0)
      improvement=true;
    /*nzarayeneh
    vector<vector<int> > nod_com_t = nod_com;
    nod_com.clear();
    for(int i=0;i<n2c.size();i++)
     nod_com[n2c[i]].push_back[nod_com_t[i]];
     /*nzaryeneh*/
    if(nb_moves <= 0 || new_qual-cur_qual <= eps_impr)
    	cerr <<"number of iterations: "<<nb_pass_done<<endl;
  } while (nb_moves>0 && new_qual-cur_qual > eps_impr);

    cerr<<"\t Time for " << n_iter << " iterations : " << tt << " (milliseconds); Per iteration cost:=> " << tt/n_iter << " millisec" << endl; 

  return improvement;
}

