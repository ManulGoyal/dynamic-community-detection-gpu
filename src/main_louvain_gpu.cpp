// File: main_louvain.cpp
// -- Delta-screening to track communities within dynamic networks, sample main file
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

#include "graph_binary.h"
#include "louvain.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <sstream>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <unordered_set>

#include "modularity.h"
#include "zahn.h"
#include "owzad.h"
#include "goldberg.h"
#include "condora.h"
#include "devind.h"
#include "devuni.h"
#include "dp.h"
#include "shimalik.h"
#include "balmod.h"
#include "timer.h"    // Manul's Change

// Manul start
#include "gpu-louvain/gpulouvain.cuh"
// Manul end

using namespace std;


char *filename = NULL;
char *filename_w = NULL;
char *filename_part = NULL;
char* filename_add = NULL;
char* filename_del = NULL;
int T ;
vector<int> n2c;
int type = UNWEIGHTED;
int nb_pass = 0;
long double precision = 0.001L;
int display_level = -2;
unsigned short id_qual = 0;
long double alpha = 0.5L;
int kmin = 1;
long double sum_se = 0.0L;
long double sum_sq = 0.0L;
long double max_w = 1.0L;
Quality *q;
bool verbose = false;


static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000;
}


namespace my2{
std::string to_string(int d)
{
	std::ostringstream stm;
	stm <<std::setprecision(numeric_limits<int>::digits10) <<d;
	return stm.str();
}
}

bool fexists(string filename)
{
	ifstream ifile(filename.c_str());
	return ifile.is_open();
}

// Manul-Manan: removes -1's from vector
void moveNegativeOnes(vector<int>& nums) {
        int j = 0;
        for(int i =0;i<nums.size();i++)
        {
            if(nums[i] != -1)
                swap(nums[i],nums[j++]);
        }
        nums.resize(j);
    }

// Manul-Manan: removes 0's from vector
void movezeroes(vector<long double>& nums) {
        int j = 0;
        for(int i =0;i<nums.size();i++)
        {
            if(nums[i] != 0)
                swap(nums[i],nums[j++]);
        }
        nums.resize(j);
    }


// Manul-Manan: It updates the given graph by adding the edges and nodes contained in filename to the graph
// Manul-Manan: and also returns the sorted array of newly added edges
// Manul-Manan: IMO, there are a few problems with this function:
// Manul-Manan: 1. Does not account for addition of edges which are incident to the 0-th node
// Manul-Manan: 2. May give SEG fault (read the comments in the body of the function)
// Manul-Manan: 3. Does not check if a newly added edge already exists
// Manul-Manan: 4. does not update weights array
vector<pair<unsigned int,unsigned int> > buildNewGraph_add(string filename,Graph *g)
{
	ifstream finput;
	vector<pair<unsigned int,unsigned int> > vect;
	vector<pair<pair<unsigned int,unsigned int>,long double> > vect_w;
	finput.open(filename.c_str());
	int node,comm;
  // Manul-Manan: count_w = number of newly added intracommunity edges
  // Manul-Manan: conut_b = number of newly added intercommunity edges
  // Manul-Manan: count_on = number of newly added edges between old and new nodes
  // Manul-Manan: count_nn = number of newly added edges between new and new nodes
	int count_w =0,count_b =0,count_nn =0,count_on =0, count_oo=0;
	vector<int> N(g->nb_nodes,0);
	vector<unsigned long long> degrees_o = g->degrees;
	if(finput.is_open())
	{
	    struct timeval start3, end3;
	    double t3;
	    gettimeofday(&start3, NULL);
	  // read partition
	    while (true)
		  {
			  unsigned int start, end;
			  long double w=1.0L;
			  if (type == WEIGHTED)
			  {
				 finput >> start >> end >> w;
				 if(finput.eof())
					 break;
			  }
			  else
			  {
          finput >> start >> end;
          if(finput.eof())
            break;
			  }
			  /*******count different types of new edges for edge sampling*****/
        if(start < g->nb_nodes && end <g->nb_nodes)
        {
          if(n2c[start]==n2c[end])
            count_w++;
          else
            count_b++;
        }
        else if(start > g->nb_nodes-1 && end > g->nb_nodes-1)
          count_nn ++;
        else
          count_on++;

        g->total_weight += 2*w;
        vect.push_back(make_pair(start, end));
        vect.push_back(make_pair(end, start));

        // N holds the increase in cumulative degrees
        if(start < N.size())//if start is
        {
            N[start]++;
        }
        else
        {
          N.resize(start+1,0);
          N[start] +=1;
        }

        if(end < N.size())
        {
          N[end]+=1;
        }
        else
        {
          N.resize(end+1,0);
          N[end]+=1;
        }

	    }//end of while
      // Manul-Manan: SEGFAULT?
      // Manul-Manan: last edge in filename not considered?
	  for(int i=1;i<N.size();i++)
		  N[i]=N[i]+N[i-1];
    
    // Manul-Manan: resize degrees to accommodate new nodes and fill the new elements with the same value
    // Manul-Manan: as the sum of all degrees in the original graph
	  g->degrees.resize(N.size(),g->degrees[g->degrees.size()-1]);

    // Manul-Manan: update cumulative degrees by adding the increments contained in N
	  for(int i=0;i<g->degrees.size();i++)
		  g->degrees[i]=g->degrees[i]+N[i];
	    gettimeofday(&end3, NULL);
	    t3 = (double) ((end3.tv_sec -start3.tv_sec)*1000 + (end3.tv_usec- start3.tv_usec)/1000) ;
    }
	  finput.close();
	 N.clear();
	 vector<int>(N).swap(N);
    struct timeval start4, end4;
	double t4;
	gettimeofday(&start4, NULL);

    // Manul-Manan: sorts newly added edges by source vertices 
  	sort(vect.begin(),vect.end());
  	int k = g->degrees.size()-1;

    // Manul-Manan: d is sum of degrees of newly added nodes, which is also equal to
    // Manul-Manan: the number of edges from new to old nodes + 2 * (num of edges from new to new nodes) 
  	int d=g->degrees[g->degrees.size()-1]-g->degrees[degrees_o.size()-1];
  	g->links.resize(g->links.size()+vect.size());
  	vector<int> v1(vect.size());
  	for(int i=0;i<vect.size();i++)
  		v1[i]=vect[i].second;
    
    // Manul-Manan: since v is sorted, the last d entries of v would be of the form (x, y), where x
    // Manul-Manan: is a newly added node. The last d entries would contain all edges between new and old nodes
    // Manul-Manan: once, and all edges between new and new nodes twice. Hence, the below call will copy
    // Manul-Manan: neighbours of all newly added nodes in the last d positions of the links array
    // Manul-Manan: the new edges from old to new nodes and old to old nodes still need to be added to links
    /* Manul: Suppose the last node in the old graph was n_old, and in new graph is n_new
     * after calling below function, the links array will look like this
     * [nbrs_of_0 nbrs_of_1 ... nbrs_of_n_old garb garb ... garb nbrs_of_(n_old+1) ... nbrs_of_n_new]
     *  |-----------------------------------| |----------------| |---------------------------------|
     *    total entries = sum of old degrees   still need update           total entries = d
     */
  	copy(v1.end()-d,v1.end(),g->links.end()-d);
  	int p=d,q=d;
    // Manul-Manan: Why i = 0 not considered?
  	for(int i=degrees_o.size()-1;i>0;i--)
  	{
  		if((g->degrees[i]-g->degrees[i-1])==(degrees_o[i]-degrees_o[i-1]))
  		{
  			q=q+(g->degrees[i]-g->degrees[i-1]);
  			copy(g->links.begin()+degrees_o[i-1],g->links.begin()+degrees_o[i],g->links.begin()+(g->links.size()-q));
        /* Manul: example: assuming that i = n_old has same degree as before
         * [nbrs_of_0 ... nbrs_of_n_old garb garb ... garb nbrs_of_(n_old+1) ... nbrs_of_n_new]
         *                ^            ^        |--------| |---------------------------------|
         *         l.b()+d_o[i-1]              deg of n_old                d
         *                                      |--------------------------------------------|
         *                                            q = d + degree of node n_old
         */
  		}
  		else
  		{
  			int pp=p;
  			p=p+((g->degrees[i]-g->degrees[i-1])-(degrees_o[i]-degrees_o[i-1]));
  			q=q+((g->degrees[i]-g->degrees[i-1])-(degrees_o[i]-degrees_o[i-1]));
  			copy(v1.begin()+(v1.size()-p),v1.begin()+(v1.size()-pp),g->links.begin()+(g->links.size()-q));
  			/* Manul: example: assuming that i = n_old has greater degree than before
         * v1 array:
         * [new_nbrs_of_other_old_nodes new_nbrs_of_n_old nbrs_of_new_nodes]
         *                             |-----------------||---------------|
         *                                   portion x         pp (=d)
         *                             |--------------------------------------|
         *                                              p (=q)  
         * links array:
         * [nbrs_of_0 ... nbrs_of_n_old garb garb ... garb nbrs_of_(n_old+1) ... nbrs_of_n_new]
         *                                      |--------| |---------------------------------|
         *                                       portion y                 d
         *                                      |--------------------------------------------|
         *                                              q = d + inc in deg of n_old
         * portion x is copied to portion y after execution upto this point
         */
        
        q=q+(degrees_o[i]-degrees_o[i-1]);
  			copy(g->links.begin()+degrees_o[i-1],g->links.begin()+degrees_o[i],g->links.begin()+(g->links.size()-q));
        /* Manul: example: assuming that i = n_old has greater degree than before
         * [nbrs_of_0 ... nbrs_of_n_old garb garb ... garb nbrs_of_(n_old+1) ... nbrs_of_n_new]
         *                |-----------|  |-----||--------| |---------------------------------|
         *                 portion z   portion w portion y                 d
         *                               |---------------------------------------------------|
         *                                   q = d + inc in deg of n_old + old deg of n_old
         * portion z is copied to portion w after execution upto this point
         */
  		}
  	}
    // Manul-Manan: the links array now includes all the new edges
    gettimeofday(&end4, NULL);
    t4 = (double) ((end4.tv_sec -start4.tv_sec)*1000 + (end4.tv_usec- start4.tv_usec)/1000) ;
	g->nb_nodes = g->degrees.size();
	g->nb_links = g->links.size();
	degrees_o.clear();
	vector<unsigned long long>(degrees_o).swap(degrees_o);
    return vect;
}

// Manul-Manan: given the graph{i}.tree file as input, where i is the time step, it returns a 
// Manul-Manan: vector of nodes contained in each community. The graph0.tree file holds the initial
// Manul-Manan: partitioning at each level, before the first time step. The format of each such
// Manul-Manan: file is as follows:
// Manul-Manan: 0 [comm of 0]     |
// Manul-Manan: 1 [comm of 1]     |  
// Manul-Manan: 2 [comm of 2]     | renumbered comm's [0..k-1] after calling one_level on original
// Manul-Manan: 3 [comm of 3]     | graph of n nodes
// Manul-Manan: ...               |
// Manul-Manan: n-1 [comm of n-1] |__________________________________________________________
// Manul-Manan: 0 [comm of 0]     |
// Manul-Manan: 1 [comm of 1]     |  
// Manul-Manan: 2 [comm of 2]     | renumbered comm's [0..k2-1] after calling one_level on
// Manul-Manan: 3 [comm of 3]     | composite graph whose nodes represent the first-level comm's
// Manul-Manan: ...               |
// Manul-Manan: k-1 [comm of k-1] |__________________________________________________________
// Manul-Manan: 0 [comm of 0]     |
// Manul-Manan: 1 [comm of 1]     |  
// Manul-Manan: 2 [comm of 2]     | renumbered comm's [0..k3-1] after calling one_level on
// Manul-Manan: 3 [comm of 3]     | composite graph whose nodes represent the second-level comm's
// Manul-Manan: ...               |
// Manul-Manan: k2-1 [comm of k-1]|__________________________________________________________
// Manul-Manan: ...
// Manul-Manan: 0 0
// Manul-Manan: 1 1
// Manul-Manan: 2 2
// Manul-Manan: ...
// Manul-Manan: k_l-1 k_l-1
vector<vector<int> >find_NodCom(string filename,int n)
 {
	n2c.resize(n);
    vector<vector<int> >levels;
    vector<vector<int> > nod_com;
	vector<vector<int> > nod_com_t;

	int l = -1;
    ifstream finput;
     finput.open(filename.c_str());
     int node, comm;
     if (finput.is_open())
      {
         while (!finput.eof())
         {
            finput >> node>>comm;
            if (finput)
            {
              if (node==0)
              {
                l++;
                levels.resize(l+1);
              }
              levels[l].push_back(comm);
            }
         }
    }
    finput.close();
    // Manul-Manan: now, levels[i][j] contains the (renumbered) comm. of the j-th node at the i-th level
    
    // Manul-Manan: the number of distinct comm. at the i-th level is the same as the number of
    // Manul-Manan: nodes in the graph just before carrying out the (i+1)-th level
    nod_com.resize(levels[1].size());
    for(int j=0;j<levels[0].size();j++)
	{
	  	nod_com[levels[0][j]].push_back(j);
	}
  // Manul-Manan: nod_com[k] contains all the nodes in the comm. k at the first level, where 
  // Manul-Manan: k is in [0, levels[1].size()-1]
	nod_com_t = nod_com;
    nod_com.clear();
    vector<vector<int> >(nod_com).swap(nod_com);
    // Manul-Manan: iterating over levels (we don't need to consider the last level, as it maps
    // Manul-Manan: each node to a singleton comm. containing only itself)
    for(int i=1;i<levels.size()-1;i++)
    {
      // Manul-Manan: number of distinct comm's at level i is same as no. of nodes at level i+1
	   nod_com.resize(levels[i+1].size());
	   // Manul-Manan: iterating over nodes of level i
     for(int j=0;j<levels[i].size();j++)
	   {
          // Manul-Manan: j-th node of level i belongs to comm. levels[i][j]. So, to nod_com[levels[i][j]],
          // Manul-Manan: we should add the entire set of original nodes represented by the composite node j,
          // Manul-Manan: which is contained in nod_com_t[j]
          nod_com[levels[i][j]].insert(nod_com[levels[i][j]].end(), nod_com_t[j].begin(), nod_com_t[j].end());
	   }
	  nod_com_t = nod_com;
      nod_com.clear();
    }
    nod_com.clear();
    vector<vector<int> >(nod_com).swap(nod_com);

    // Manul-Manan: nod_com_t now contains the original nodes assigned to each final comm (obtained after the final level) 
    return nod_com_t;
 }

// Manul-Manan: updates given graph by deleting the edges mentioned in filename from graph
// Manul-Manan: and returns sorted list of edges removed
// Manul-Manan: note: if (a, b) is present in filename, removes both (a, b) and (b, a) from graph,
// Manul-Manan: if these edges exist. But in the returned vector, only (a, b) is present.
// Manul-Manan: Does not properly update the weights array, and does not remove isolated nodes
vector<pair<unsigned int,unsigned int> > buildNewGraph_del(string filename, Graph *g, vector<pair<unsigned int,unsigned int> >& delEdges)
{
	ifstream finput;
	vector<pair<unsigned int,unsigned int> > vect;
	vector<int>::iterator it;
	finput.open(filename.c_str());
	vector<int> N(g->nb_nodes,0);//save the number of edges that we remove from each node
	int cl=0,cw=0;
	if(finput.is_open())
	{
	    struct timeval start3, end3;
	    double t3;
	    int count_notexist=0;
	    gettimeofday(&start3, NULL);
	  // read partition
	    while (true)
		{
			  unsigned int start, des;
			  long double w = 1.0L;
			  if (type == WEIGHTED)
			  {
				 finput >> start >> des >> w;
				 if(finput.eof())
					 break;
			  }
			  else
			  {
				finput >> start >> des;
				if(finput.eof())
					break;
			  }
              bool flag = false;
                //remove the edge (start,end) (in links and weights we should find start and end and erase them)
            if(start >  g->nb_nodes-1 ) cout<<"wrong start deletion"<<endl;
            else if(start == g->nb_nodes-1 ){
              // Manul-Manan: find the destination node in the segment of links containing neighbours of source
                it = find(g->links.begin() + g->degrees[start-1], g->links.end(), des);//check does not exist
                if(g->degrees[start] != g->degrees[start-1] && it != g->links.end())//if this edge exists
                  {
                     flag = true;
                     // Manul-Manan: set the dest corresponding to source to -1
                    *it = -1;
                    if (type == WEIGHTED)
                       {
                         // Manul-Manan: subtract the weight read from file from the weight of the edge to be deleted. Why this is done is a mystery
                           g->weights[distance(g->links.begin(),it)] -= w;
                           g->total_weight-=w;
                       }
                    else
                       {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                  }

            }
            else if(start == 0){
              // Manul-Manan: Issue: why is 1 subtracted from the the upper limit?
                it = find(g->links.begin(), g->links.begin()+g->degrees[start]-1, des);
                if(g->degrees[start] != 0 &&  distance(g->links.begin(),it)<g->degrees[start])//   it != g->links.begin()+g->degrees[start])//if this edge exists
                  {
                    flag = true;
                    *it = -1;
                    if (type == WEIGHTED)
                    {
                        g->weights[distance(g->links.begin(),it)] -= w;
                        g->total_weight-=w;
                    }
                    else
                        {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                  }

            }
            else{
              // Manul-Manan: Issue: why is 1 subtracted from the the upper limit?
                it = find(g->links.begin()+g->degrees[start-1], g->links.begin()+ g->degrees[start]-1, des);
                if(g->degrees[start] != g->degrees[start-1] && distance(g->links.begin()+g->degrees[start-1],it) < (g->degrees[start]-g->degrees[start-1]))//it != g->links.begin()+g->degrees[start])//if this edge exists
                  {
                    flag = true;
                    *it = -1;
                    if (type == WEIGHTED)
                    {
                        g->weights[distance(g->links.begin(),it)] -= w;
                        g->total_weight-=w;
                    }
                    else
                        {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                  }

            }

            // Manul-Manan: uptil now (start, des) was deleted, now repeat same procedure with (des, start)
            if(des >  g->nb_nodes-1 ) cout<<"wrong end deletion"<<endl;
            else if(des == g->nb_nodes -1){
                  it = find(g->links.begin() + g->degrees[des-1], g->links.end(), start);//check does not exist
                if(g->degrees[start] != g->degrees[des-1] && it != g->links.end())//if this edge exists
                  {
                      flag = true;
                    *it = -1;
                    if (type == WEIGHTED)
                       {
                           g->weights[distance(g->links.begin(),it)] -= w;
                           g->total_weight-=w;
                       }
                    else
                       {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                  }

            }
            else if(des == 0){
              // Manul-Manan: Issue: why is 1 subtracted from the the upper limit?
                  it = find(g->links.begin(), g->links.begin()+g->degrees[des]-1, start);
                if(g->degrees[des] != 0 &&  distance(g->links.begin(),it)<g->degrees[des])//   it != g->links.begin()+g->degrees[start])//if this edge exists
                  {
                      flag = true;
                    *it = -1;
                    if (type == WEIGHTED)
                    {
                        g->weights[distance(g->links.begin(),it)] -= w;
                        g->total_weight-=w;
                    }
                    else
                        {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                    cerr<<"Debug... start: "<< start <<" does not have any edge"<<endl;
                  }

            }
            else{
              // Manul-Manan: Issue: why is 1 subtracted from the the upper limit?
                  it = find(g->links.begin()+g->degrees[des-1], g->links.begin()+ g->degrees[des]-1, start);
                if(g->degrees[des] != g->degrees[des-1] && distance(g->links.begin()+g->degrees[des-1],it) < (g->degrees[des]-g->degrees[des-1]))//it != g->links.begin()+g->degrees[start])//if this edge exists
                  {
                      flag = true;
                    *it = -1;
                    if (type == WEIGHTED)
                    {
                        g->weights[distance(g->links.begin(),it)] -= w;
                        g->total_weight-=w;
                    }
                    else
                        {g->total_weight -= 1;cw++;}
                  }
                  else{
                    count_notexist++;
                    cerr<<"Debug... start: "<<start<<" does not have any edge"<<endl;
                  }
            }

            // Manul-Manan: flag is true if the edge (start, des) or (des, start) exists
            if(flag)//if the edges exists then reduce the degrees
            {
                N[start]++;//we increase one when we remove an edge
                N[des]++;
                // Manul-Manan: why (des, start) is not added?
                vect.push_back(make_pair(start, des));

                delEdges.push_back(make_pair(start, des));
                delEdges.push_back(make_pair(des, start));

            }

	  }//end of while

	 for(int i=0;i<g->links.size();i++)
	  {
	   	if(g->links[i]==-1) ++cl;
	  }
    
    // Manul-Manan: this removes all -1's from links array and resizes it after removal
	    moveNegativeOnes(g->links);

        for(int i=1;i<N.size();i++)
		  N[i]=N[i]+N[i-1];//as we want the accumulated degree, we add up the number of removed edges
        for(int i=0;i<g->degrees.size();i++)//update the degrees after removing the batch of edges from the graph
		  g->degrees[i] = g->degrees[i]-N[i];
        gettimeofday(&end3, NULL);
        t3 = (double) ((end3.tv_sec - start3.tv_sec)*1000 + (end3.tv_usec- start3.tv_usec)/1000) ;
     } // Manul-Manan: end of if(finput.isopen())
     finput.close();
     N.clear();
     vector<int>(N).swap(N);
     struct timeval start4, end4;
     double t4;
     gettimeofday(&start4, NULL);
     sort(vect.begin(),vect.end());
     sort(delEdges.begin(), delEdges.end());
     gettimeofday(&end4, NULL);
     t4 = (double) ((end4.tv_sec -start4.tv_sec)*1000 + (end4.tv_usec- start4.tv_usec)/1000) ;
     g->nb_nodes = g->degrees.size();
     g->nb_links = g->links.size();
    return vect;
}


vector<int> nodToEval_b(vector<vector<int> > nodes_in_comm,Graph *g,Louvain *louvain, int type, vector<int> n2c,vector<pair<unsigned int,unsigned int> > vect,int nb_nodes)
{
  for(int i=0;i<n2c.size();i++)
	louvain->qual->R[i]=i;
  return louvain->qual->R;
}

// Manul-Manan: compute the set of nodes to be evaluated by Louvain when some edges (contained in vect) are
// Manul-Manan: added to the graph at the current time step
vector<int> nodToEval_add(vector<vector<int> > nodes_in_comm,Graph *g,Louvain *louvain, int type, vector<int> n2c,vector<pair<unsigned int,unsigned int> > vect, int nb_nodes)
{
    louvain->qual->R.clear();
   vector<int>::iterator it;
   vector<int> Re(n2c.size(),0);
    int i=0;
    int max = 0;
    int comm_to ;
    int comm1,comm2;
    double gain_start_to_comm1, gain_start_to_comm2;
    long double w_degree1,w_degree2;
    vector<int> flag_neig(g->nb_nodes,0);
    vector<int> flag_comm(g->nb_nodes,0);
    double gain_time = 0, neig_time = 0, comm_time =0;

    // Manul-Manan: iterating over all source nodes of the newly added edges. vect is already sorted wrt source nodes 
    while(i<vect.size())
	{
      // Manul-Manan: the neigh comm of source node which gives maximum gain upon moving the source node to it
    	  comm_to = -1;
    	  int flag =0;
		  unsigned int start, end, max_end; // Manul-Manan: max_end is the target node corr. to which gain is maximum
		  start = vect[i].first;
		  comm2 = n2c[start];
		  struct timeval sTime1, eTime1;

      // Manul-Manan: iterating over all target nodes corresponding to source node 'start'
		  while(start == vect[i].first)
		  {
			  end = vect[i].second;
			  comm1 = n2c[end];
			  if(comm1==comm2)
			  {
          // Manul-Manan: if edge added is intracommunity, do nothing
				  i++;
				  continue;
			  }
			  else
			  {
				 if(start >= nb_nodes && end >= nb_nodes)
				  {
            // Manul-Manan: this condition checks that whether the current edge is between two new nodes
            // Manul-Manan: and if so, add start to the node set for evaulation and continue
		  			  Re[start] = 1;
		  			  i++;
		  			  continue;
				  }
				 else
				 {
            // this condition ensures that the body of this 'if' runs only once for each source node
					  if(flag == 0)
					  {
						  Re[start] =1;
						  flag =1;
						  gettimeofday(&sTime1, NULL);
						  w_degree1 = g->weighted_degree(start);
              // Manul-Manan: Issue: louvain->neigh_comm[start] has nowhere been called. So, neigh_weight won't contain the expected values. Also, start has not been removed from its current comm (comm2) yet, so calculating gain produced by addition of start to comm2 does not make sense. Also, why w_degree1 is being subtracted in third argument?
						  gain_start_to_comm2 = louvain->qual->gain(start, comm2, louvain->neigh_weight[comm2]-w_degree1, w_degree1);
						  gettimeofday(&eTime1, NULL);
						  gain_time += (double) ((eTime1.tv_sec -sTime1.tv_sec )*1000 + (eTime1.tv_usec -  sTime1.tv_usec)/1000);
					  }

					  gettimeofday(&sTime1, NULL);
            // Manul-Manan: Issue: neigh_weight does not contain expected values 
					  gain_start_to_comm1 = louvain->qual->gain(start, comm1, louvain->neigh_weight[comm1], w_degree1);
					  gettimeofday(&sTime1, NULL);
					  gain_time += (double) ((eTime1.tv_sec -sTime1.tv_sec )*1000 + (eTime1.tv_usec -  sTime1.tv_usec)/1000);
            // Manul-Manan: if moving start to neigh comm yields greater gain than staying in its own comm
					  if(gain_start_to_comm1 > gain_start_to_comm2)
					  {
						  if(max < gain_start_to_comm1)
						  {
							  max = gain_start_to_comm1;
							  comm_to = comm1;
                                            max_end = end;
						  }
					  }
					  i++;
			       } // Manul-Manan: end else
		       } // Manul-Manan: end else
	      } // Manul-Manan: end inner while (iteration over targets)

		  if(comm_to == -1)
		  {
        // Manul-Manan: no positive gain can be obtained by moving start to one of the comm. of its target
        // Manul-Manan: nodes. Thus, proceed to the next source node
			 continue;
		  }
		  else
		  { // Manul-Manan: Issue: it is not checked that moving start to max_end's comm yields greater gain
        // Manul-Manan: than moving max_end to start's comm
        //start is going to comm_to
			       gettimeofday(&sTime1,NULL);
				   Re[start]=1;
				   //add neighbors of start
				   if(flag_neig[start] != 1)
				   {
					   flag_neig[start] = 1;
					   pair<vector<int>::iterator, vector<long double>::iterator > p = (louvain->qual->g).neighbors(start);
					   vector<int>::iterator ngbr = p.first;
						//add all neighbors of start node in R except those that are neighbor with end node in its community or single nodes
						for (int k = 0; k < (louvain->qual->g).nb_neighbors(start); k++)
						{
					      int nb = ngbr[k];
                // Manul-Manan: checking if the k-th neighbour of start belongs to same comm as max_end
                // Manul-Manan: Issue: shouldn't it be comm_to?
						  if (louvain->qual->n2c[ngbr[k]] != comm1)
						  {
							  Re[nb]=1;
						  }
						  else
						  {
							 Re[nb]=-1;
						  }
						}
				   }
				   gettimeofday(&eTime1,NULL);
				   neig_time += (double) ((eTime1.tv_sec -sTime1.tv_sec )*1000 + (eTime1.tv_usec - sTime1.tv_usec)/1000);
				   //add all nodes in comm1
				   gettimeofday(&sTime1,NULL);
           // Manul-Manan: this condition handles the case when new edge is from old node to new node
					if(comm1>nodes_in_comm.size()-1)//end is a single new node as a community
					{
            // Manul-Manan: Issue: why are we not adding max_end to R in this case?
						continue;
					}

					else
					{//end is located in a community with more than one member
						if(flag_comm[comm1] !=1)//check if we have already added the nodes in this community
						{
							flag_comm[comm1]= 1;
							for(int k = 0; k < nodes_in_comm[comm1].size(); k++)
							{//add all nodes in community of end node(not repeated in R, not single node)
								  //check weather we have already pushed it into R
								int node=nodes_in_comm[comm1][k];
								if((louvain->qual->g).nb_neighbors(node)>1 && Re[node] != -1)//?should we exclude singular nodes?
								{
								  Re[node]=1;
								}
							}
						}
			         }
					 gettimeofday(&eTime1,NULL);
					 comm_time += (double) ((eTime1.tv_sec -sTime1.tv_sec )*1000 + (eTime1.tv_usec - sTime1.tv_usec)/1000);
				   }


	} // Manul-Manan: end of outer while (iteration over vect)
   int reeval_count=0;
   for(int i=0;i<Re.size();i++)
   {
	   if(Re[i]==1)
         {
           reeval_count++;
		   louvain->qual->R.push_back(i);
         }
   }
   return Re;
}

//it will return a vector including nodes to be reevaluated
vector< int > nodToEval_del(vector<vector<int> > nodes_in_comm,Graph *g,Louvain *louvain, int type, vector<int> n2c,vector<pair<unsigned int,unsigned int> > vect, int nb_nodes)
{
	louvain->qual->R.clear();
	//vect is a vector of pairs containing all edges to be deleted and edge_to_check is a vector telling which of these edges should be consider for reevaluation
	vector<int> edge_to_check(vect.size());
	//com_to_check is a vector of size all communities and it is 1 if the community needs to be reevaluated
	vector<int> com_to_check(nodes_in_comm.size());
	unordered_set<int> nodes_to_check;
	unordered_set<int>::iterator it;
	//check whether the deleted edge is between two different communities
	for(int i=0;i<vect.size();i++)
	{
		if(n2c[vect[i].first] != n2c[vect[i].second])
			continue;
		else
		{
		    if(com_to_check[n2c[vect[i].first]] ==1 )
                continue;
			com_to_check[n2c[vect[i].first]] = 1;//this comm should be recomputed
			nodes_to_check.insert(nodes_in_comm[n2c[vect[i].first]].begin(),nodes_in_comm[n2c[vect[i].first]].end());
			vector<int> k = {vect[i].first, vect[i].second};//for each comm i it holds the number of nodes
			for(int j=0;j<k.size();j++)
			{
				//for each node in the selected community find its neighbors wihin the community that form an edge and add this edge to the subgraph
				pair<vector<int>::iterator, vector<long double>::iterator > p = (louvain->qual->g).neighbors(k[j]);
				for(int v=0;v<(louvain->qual->g).nb_neighbors(k[j]);v++)//add all neighbors of each node in community that needs to be reevaluate
				{
					if(n2c[*(p.first+v)]!=n2c[k[j]]){//if the neighbors of a node in the current comm is in other communities add them to nodes_to_check
                       nodes_to_check.insert(*(p.first+v));
					}
				}
			}
		}
	}
    for(auto i=nodes_to_check.begin();i!=nodes_to_check.end();i++)
	    {
 		   louvain->qual->R.push_back(*i);
	    }
	return louvain->qual->R;
}



void
usage(char *prog_name, const char *more) {
  cerr << more;
  cerr << "usage: " << prog_name << " input_file [-q id_qual] [-c alpha] [-k min] [-w weight_file] [-p part_file] [-e epsilon] [-l display_level] [-v] [-h] [-x graph_delta_add] [-a num_change][-b graph_delta_del]"<<endl <<endl;
  cerr << "input_file: file containing the graph to decompose in communities" << endl;
  cerr << "-q id\tthe quality function used to compute partition of the graph (modularity is chosen by default):" << endl << endl;
  cerr << "\tid = 0\t -> the classical Newman-Girvan criterion (also called \"Modularity\")" << endl;
  cerr << "\tid = 1\t -> the Zahn-Condorcet criterion" << endl;
  cerr << "\tid = 2\t -> the Owsinski-Zadrozny criterion (you should specify the value of the parameter with option -c)" << endl;
  cerr << "\tid = 3\t -> the Goldberg Density criterion" << endl;
  cerr << "\tid = 4\t -> the A-weighted Condorcet criterion" << endl;
  cerr << "\tid = 5\t -> the Deviation to Indetermination criterion" << endl;
  cerr << "\tid = 6\t -> the Deviation to Uniformity criterion" << endl;
  cerr << "\tid = 7\t -> the Profile Difference criterion" << endl;
  cerr << "\tid = 8\t -> the Shi-Malik criterion (you should specify the value of kappa_min with option -k)" << endl;
  cerr << "\tid = 9\t -> the Balanced Modularity criterion" << endl;
  cerr << endl;
  cerr << "-c al\tthe parameter for the Owsinski-Zadrozny quality function (between 0.0 and 1.0: 0.5 is chosen by default)" << endl;
  cerr << "-k min\tthe kappa_min value (for Shi-Malik quality function) (it must be > 0: 1 is chosen by default)" << endl;
  cerr << endl;
  cerr << "-w file\tread the graph as a weighted one (weights are set to 1 otherwise)" << endl;
  cerr << "-p file\tstart the computation with a given partition instead of the trivial partition" << endl;
  cerr << "\tfile must contain lines \"node community\"" << endl;
  cerr << "-e eps\ta given pass stops when the quality is increased by less than epsilon" << endl;
  cerr << "-l k\tdisplays the graph of level k rather than the hierachical structure" << endl;
  cerr << "\tif k=-1 then displays the hierarchical structure rather than the graph at a given level" << endl;
  cerr << "-v\tverbose mode: gives computation time, information about the hierarchy and quality" << endl;
  cerr << "-h\tshow this usage message" << endl;
  cerr << "-x file \t read the changes including adding edges" << endl;
  cerr << "-b name \t read the changes including deleting edges"<<endl;
  cerr << "-a num \t gets the number of delta files "<<endl;
  exit(0);
}

void
parse_args(int argc, char **argv) {
  if (argc<2)
    usage(argv[0], "Bad arguments number\n");

  for (int i = 1; i < argc; i++) {
    if(argv[i][0] == '-') {
    switch(argv[i][1]) {
    case 'w':
        type = WEIGHTED;
        filename_w = argv[i+1];
        i++;
        break;
    case 'q':
        id_qual = (unsigned short)atoi(argv[i+1]);
        i++;
        break;
    case 'c':
        alpha = atof(argv[i+1]);
        i++;
        break;
    case 'k':
        kmin = atoi(argv[i+1]);
        i++;
        break;
    case 'p':
        filename_part = argv[i+1];
        i++;
        break;
    case 'e':
        precision = atof(argv[i+1]);
        i++;
        break;
    case 'l':
        display_level = atoi(argv[i+1]);
        i++;
        break;
    case 'v':
        verbose = true;
        break;
    case 'x':
        filename_add = argv[i+1];
        i++;
        break;
    case 'o':
    	alpha = atof(argv[i+1]);
    	i++;
    	break;
    case 'h':
        usage(argv[0], "");
        break;
    case 'a':
        T = atoi(argv[i+1]);
        i++;
        break;
    case 'b':
        filename_del = argv[i+1];
        i++;
        break;
    default:
        usage(argv[0], "Unknown option\n");
      }
    } else {
      if (filename==NULL)
        filename = argv[i];
      else
        usage(argv[0], "More than one filename\n");
    }
  }
  if (filename == NULL)
    usage(argv[0], "No input file has been provided\n");
}

void
display_time(const char *str) {
  time_t rawtime;
  time ( &rawtime );
  cerr << str << ": " << ctime (&rawtime);
}

void
init_quality(Graph *g, unsigned short nbc) {

  if (nbc > 0)
    delete q;

  switch (id_qual) {
  case 0:
    q = new Modularity(*g);
    break;
  case 1:
    if (nbc == 0)
      max_w = g->max_weight();
    q = new Zahn(*g, max_w);
    break;
  case 2:
    if (nbc == 0)
      max_w = g->max_weight();
    if (alpha <= 0. || alpha >= 1.0)
      alpha = 0.5;
    q = new OwZad(*g, alpha, max_w);
    break;
  case 3:
    if (nbc == 0)
      max_w = g->max_weight();
    q = new Goldberg(*g, max_w);
    break;
  case 4:
    if (nbc == 0) {
      g->add_selfloops();
      sum_se = CondorA::graph_weighting(g);
    }
    q = new CondorA(*g, sum_se);
    break;
  case 5:
    q = new DevInd(*g);
    break;
  case 6:
    q = new DevUni(*g);
    break;
  case 7:
    if (nbc == 0) {
      max_w = g->max_weight();
      sum_sq = DP::graph_weighting(g);
    }
    q = new DP(*g, sum_sq, max_w);
    break;
  case 8:
    if (kmin < 1)
      kmin = 1;
    q = new ShiMalik(*g, kmin);
    break;
  case 9:
    if (nbc == 0)
      max_w = g->max_weight();
    q = new BalMod(*g, max_w);
    break;
  default:
    q = new Modularity(*g);
    break;
  }
}
int
main(int argc, char **argv) {

	int count=0;
	for(int j=0;j<=T ;j++)//we will make all graph.tree files empty before starting
	  {
		  		string str = "graph"+ my2::to_string(j)+ ".tree";
		  	    if(fexists(str))
		  	     {
		  	      	ifstream infile;
		  	       	infile.open(str.c_str(),std::ifstream::out | std::ifstream::trunc);
		  	       	if(!infile.is_open() || infile.fail())
		  	       	{
		  	       		infile.close();
		  	       		printf("\n Error: failed to erase file comtent.");
		  	       	}
		  	       	infile.close();
		  	      }
		  	    else
		  	    	cerr<<"not exists"<<endl;
	  }
  struct timeval startTime, endTime,level_time_s,level_time_e;
  double elapsedSeconds, level_time=0;
  gettimeofday(&startTime, NULL);
  srand(time(NULL)+getpid());
  parse_args(argc, argv);//it gets filename_w, filename_part
  unsigned short nb_calls = 0;//# of iteration per pass

  if (verbose)
    display_time("Begin");
  Graph g(filename, filename_w, type);
  Graph gr = g;
  int n = gr.nb_nodes;
  // Manul-Manan: this will create a new quality class (modularity by default), and initialize it with graph
  // Manul-Manan: g. Initially, each node is a community by itself (q->n2c[i] == i)
  init_quality(&g, nb_calls);

  // Manul-Manan: R is the set of nodes to be evaluated by Louvain. For time step 0, all nodes should be
  // Manul-Manan: evaluated
  for(int i=0; i< q->size;i++)
  {
 	  //assign all nodes to R for time t0
  	  q->R[i] = i;
  }
  nb_calls++;

  if (verbose)
    cerr << endl << "Computation of communities with the " << q->name << " quality function" << endl << endl;

  // Manul-Manan: -1 means do as many passes at a given level as required, precision is the threshold for gain below which no more passes will be done 
  Louvain c(-1, precision, q);
  if (filename_part!=NULL)
    // Manul-Manan: TODO: initial partition file at the start is not handled currently with GPU louvain
    c.init_partition(filename_part);
  bool improvement = true;
  // cout << "Gpu start" << endl;

  // Manul-Manan: quality of initial partition
  long double quality = (c.qual)->quality();
  long double new_qual;
  int level = 0;
  gettimeofday(&level_time_s,NULL);

  

  // Manul start
  n2c.resize(n);
  GTIC("Init part mem");
  GPUWrapper gpuWrapper(&c, n2c, false);
  GTOC;

  GTIC("Initial partitioning");
  // cout << "Gpu start 2" << endl;
  gpuWrapper.gpuLouvain(n2c);
  GTOC;
  // Manul end

  cout << "Gpu done" << endl;

  /**********Commented*****************/
  // GTIC("Initial partitioning");   // Manul's Change
  // // Manul-Manan: computation of comm's before the first time step (before any changes happen)
  // do {
  //   GTIC("Overall level " + to_string(level));    // Manul's Change
  //   if (verbose) {
  //     cerr <<"level " << level << ":\n";
  //     cerr << "\t network size: "
	//    << (c.qual)->g.nb_nodes << " nodes, "
	//    << (c.qual)->g.nb_links << " links, "
	//    << (c.qual)->g.total_weight << " weight" << endl;
  //   }

  //   GTIC("one_level");  // Manul's Change
  //   // Manul-Manan: carry out one level (consisting of one or more passes). The passes stop if
  //   // Manul-Manan: either no nodes were moved in the last pass, or the gain in the last pass 
  //   // Manul-Manan: was less than the specified threshold. Improvement is true if at least one of 
  //   // Manul-Manan: the nodes was moved and is false if no nodes were moved. In the latter case, we  
  //   // Manul-Manan: do not need to compute the next level
  //   improvement = c.one_level();

  //   GTOC;                 // Manul's Change

  //   // Manul-Manan: now some of the nodes may have been reassigned to different comm's, but graph is the same

  //   new_qual = (c.qual)->quality();
  //   if (++level==display_level)
  //     (c.qual)->g.display();
  //   if (display_level==-1)
  //     c.display_partition(count);


  //   GTIC("part2graph");  // Manul's Change
  //   // Manul-Manan: create a new graph whose nodes represent the computed comm's. If no nodes were moved
  //   // Manul-Manan: and each node still forms a singleton comm, this would not change the graph
  //   g = c.partition2graph_binary();
  //   GTOC;      // Manul's Change

  //   // Manul-Manan: a completely new quality function corr. to the new graph, with each node as a singleton comm
  //   init_quality(&g, nb_calls);
  //   nb_calls++;

  //   // Manul-Manan: a new value for q requires a new Louvain object with the new q
  //   c = Louvain(-1, precision, q);
  //   quality = new_qual;
  //   if (filename_part!=NULL && level==1)
  //   // Manul-Manan: this is done because if an custom initial partitioning was provided, then it may have 
  //   // Manul-Manan: happened that no further movement of nodes was done by one_level on top of the initial
  //   // Manul-Manan: custom paritioning, and so improvement would be false in such a case, but partition2graph_binary 
  //   // Manul-Manan: would give a different graph than the original, and so we would like to continue the 
  //   // Manul-Manan: procedure with this new graph
  //     improvement=true;
  //   GTOC;   // Manul's Change
  // } while(improvement);
  // GTOC;   // Manul's Change
  /**********Commented*****************/

  gettimeofday(&level_time_e,NULL);
  level_time += (double) ((level_time_e.tv_sec -level_time_s.tv_sec )*1000 + (level_time_e.tv_usec -  level_time_s.tv_usec)/1000);
  gettimeofday(&endTime, NULL);
  double time_for_t0 = (double) ((endTime.tv_sec -startTime.tv_sec )*1000 + (endTime.tv_usec -  startTime.tv_usec)/1000);
  cerr<<"\t Time for step 0:(milliseconds) "<<time_for_t0<<endl;
  int count_add = 0;
  int count_del = 0;
  vector<int>::iterator it,it2;
  vector<int> nodes_to_check;
  // Manul-Manan: it2 is never being used anywhere
  it2 = max_element(q->n2c.begin(), q->n2c.end());
  vector<int> Re;
  vector<pair<unsigned int,unsigned int> > vect_new_edges;
  vector<vector<int> > nodes_in_comm;//for each community holds their nodes
  int n_comm;
  Graph g1 = gr;//gr hold the changed graph at the start of each time step
  string s = "graph0.tree";
  string s_old= "graph0.tree";
  string clusters;
  string name;
  string comm;
  ofstream ofile8;
  string n2cInit = "n2c0.txt";
  int nb_nodes_b = gr.nb_nodes;

  GTIC("All time steps");   // Manul's Change
  for(int j=0;j<=T;j++)
  {
    GTIC("Overall ts " + to_string(j)); // Manul's Change
      cerr<<endl <<"**********************************************"<<endl;
	  cerr<< "Timestep " <<j<<" deletion starts"<<endl;

    GTIC("Deletion");       // Manul's Change
	  double time_per_step2;
	  level_time = 0;
	  struct timeval sTime, eTime;
	  double time_per_step;
	  gettimeofday(&sTime, NULL);
	  count_del++;
      name = filename_del+my2::to_string(j)+".txt";//remove  +".txt" if your delta changes have not a suffix
	  nb_nodes_b = gr.nb_nodes;
      s = "graph"+my2::to_string(count_del-1)+".tree";
      clusters = "cluster"+my2::to_string(count_del-1)+".txt";
      comm = "comm"+my2::to_string(count_del-1)+".txt";
      vector<vector<int> > nodes_in_comm;

      /**********Commented*****************/
      // GTIC("find_NodCom del");   // Manul's Change
      // // Manul-Manan: after the next line, nodes_in_comm[i] is the set of all original nodes
      // // Manul-Manan: contained in community i (based on the last level of the last time step)
      // nodes_in_comm = find_NodCom(s,nb_nodes_b);
      // GTOC;   // Manul's Change
      /**********Commented*****************/

      // Manul start
      int num_of_comm = (*max_element(n2c.begin(), n2c.end())) + 1;
      nodes_in_comm.resize(num_of_comm);
      for(int node_i = 0; node_i < n2c.size(); node_i++) {
        nodes_in_comm[n2c[node_i]].push_back(node_i);
      }
      // Manul end

      /**********Commented*****************/
      // ofstream myfile (clusters.c_str(), ios::app);
      // if (myfile.is_open())
      // {
      //          for(int i=0;i<nodes_in_comm.size();i++)
      //          {
      //              myfile<<nodes_in_comm[i].size()<<endl;
      //              /**********Commented*****************/
      //             //  for(int j=0;j<nodes_in_comm[i].size();j++)
      //             //      n2c[nodes_in_comm[i][j]] = i;
      //             /**********Commented*****************/
      //          }
      // }
      // myfile.close();
      /**********Commented*****************/

      int idx=0;
      /**********Commented*****************/
      // ofstream myfile2(comm.c_str(), ios::app);
      // if (myfile2.is_open())
      // {
      //       for(int i=0;i<nodes_in_comm.size();i++)
      //       {
      //           for(int j=0;j<nodes_in_comm[i].size();j++)
      //           {
      //               myfile2<<nodes_in_comm[i][j]<<endl;
      //           }
      //       }
      // }
      // myfile2.close();
      /**********Commented*****************/

	  //read each file of changes called name, create the new graph, find the affected vertices and apply louvain for them using the previous community structure
	  if(fexists(name))
	  {

      vector<pair<unsigned int,unsigned int> > delEdges;

    GTIC("buildNew_del"); // Manul's Change
      // Manul-Manan: modify gr by deleting the edges specified in file 'name'
		vect_new_edges = buildNewGraph_del(name,&gr,delEdges);
    GTOC;    // Manul's Change

		Graph g1 = gr;
		int nb_nodes_a = gr.nb_nodes;
		vector<int>::const_iterator it;
		it = max_element(n2c.begin(), n2c.end());

    // Manul-Manan: n_comm is the total comm's formed after the last level of the last time step
		int n_comm = *it +1;

    // Manul-Manan: initialize quality on the new graph (with deleted edges)
		init_quality(&g1, nb_calls);
		nb_calls++;
		Louvain c1(-1, precision, q);//call louvain constructor

    // Manul-Manan: n2c contains the comm corr. to each original node as per the last level of the last time step
    // Manul-Manan: this should be used as the initial partition at the beginning of this time step
		c1.init_partition_v(n2c);
		q->n2c = n2c;
		struct timeval sTime2, eTime2;
		gettimeofday(&sTime2, NULL);
		//Re = nodToEval_b(nodes_in_comm,&g1,&c1, type,n2c,vect_new_edges, nb_nodes_b);

    // Manul start
      GTIC("part after del mem");
      GPUWrapper gpuWrapper(&c1, n2c, true);
      GTOC;

      GTIC("node eval del gpu");
      int R_size = gpuWrapper.gpuNodeEvalDel(delEdges);
      GTOC;

    // Manul end

    /**********Commented*****************/
    // GTIC("nodToEval_del");  // Manul's Change
    // // Manul-Manan: compute the set of nodes to be re-evaulated by using the new graph (with deleted edges), and
    // // Manul-Manan: the partition computed by the last level of the last time step
		// Re = nodToEval_del(nodes_in_comm,&g1,&c1, type,n2c,vect_new_edges, nb_nodes_b);
    // GTOC;    // Manul's Change
    /**********Commented*****************/

      /**********Commented*****************/
      // // Manul-Manan: if re-evaluation set is empty, simply copy graph.tree file for prev time step
      // // Manul-Manan: to the graph.tree file for the current time step
      //   if((c1.qual)->R.size() == 0)//if  deletion did not make any effect at time step j!=0
      //    {
      //       s = "graph"+my2::to_string(count_del)+".tree";
      //       string line;
      //       ifstream myfile (s_old);
      //       ofstream myfile2 (s);
      //       if (myfile.is_open())
      //       {
      //           while ( getline (myfile,line) )
      //           {
      //               if (myfile2.is_open())
      //                   myfile2 << line << '\n';
      //               else
      //                   cout << "Unable to open output file";

      //           }
      //           myfile.close();
      //       }
      //       else
      //           cout << "Unable to open file";
      //       myfile2.close();
      //   }
      /**********Commented*****************/

		gettimeofday(&eTime2, NULL);
		time_per_step2 = (double) ((eTime2.tv_sec -sTime2.tv_sec )*1000 + (eTime2.tv_usec -  sTime2.tv_usec)/1000);
		cerr<<"\t Time for nodToEval(milliseconds): "<<time_per_step2<<endl;
		nodes_in_comm.clear();
		vector<vector<int> >(nodes_in_comm).swap(nodes_in_comm);
		vect_new_edges.clear();
		vector<pair<unsigned int,unsigned int> >(vect_new_edges).swap(vect_new_edges);

		cerr<<"R percentage : "<< ((double)R_size/(double)(c1.qual)->g.nb_nodes)*100  <<endl;
		if(R_size!=0)
		{
		    s = "graph"+my2::to_string(count_del)+".tree";
			nb_calls =0;
			improvement = true;
			quality = (c1.qual)->quality();// modularity of current partition( for the very first step each node in one comm)
			level = 0;
			gettimeofday(&level_time_s,NULL);

      /**********Commented*****************/
      // // Manul start
      // GTIC("part after del mem");
      // // Manul-Manan: isolated nodes are not removed after deletion, and n2c is perfectly valid after deletion
      // GPUWrapper gpuWrapper(&c1, n2c, true);
      // GTOC;
      /**********Commented*****************/

      GTIC("partitioning after del");
      gpuWrapper.gpuLouvain(n2c);
      GTOC;
      // Manul end

      /**********Commented*****************/
      // GTIC("partitioning after del")
			// do {
      //   GTIC("Level " + to_string(level)); // Manul's Change
			// 	if (verbose)
			// 	{
			// 		cerr <<"level " << level << ":\n";
			// 		cerr << "\t network size: "
			// 		     << (c1.qual)->g.nb_nodes << " nodes, "
			// 		     << (c1.qual)->g.nb_links << " links, "
			// 		     << (c1.qual)->g.total_weight << " weight" << endl;
			// 	}

      //   GTIC("one_level");  // Manul's Change
			// 	improvement = c1.one_level();// do the first pass
      //   GTOC; // Manul's Change
			// 	new_qual = (c1.qual)->quality();// new_qual is the new modularity
			// 	if (++level==display_level)
			// 		(c1.qual)->g.display();
			// 	if (display_level==-1)
      //     // Manul-Manan: this method appends to the graph{count_del}.tree file the renumbered comm
      //     // Manul-Manan: associated with each node at the current level
			// 		c1.display_partition(count_del);
        
      //   GTIC("part2graph");   // Manul's Change
			// 	g1 = c1.partition2graph_binary();// find the new graph for the next pass
      //   GTOC;   // Manul's Change
			// 	init_quality(&g1, nb_calls);
			// 	nb_calls++;

			// 	c1 = Louvain(-1, precision, q);
			// 	quality = new_qual;
			// 	if (n2c.size() != 0 && level==1) // do at least one more computation if partition is provided
			// 		improvement=true;
      //   GTOC;   // Manul's Change
			// } while(improvement);
      // GTOC;   // Manul's Change
      /**********Commented*****************/

			gettimeofday(&level_time_e,NULL);
			level_time += (double) ((level_time_e.tv_sec -level_time_s.tv_sec )*1000 + (level_time_e.tv_usec -  level_time_s.tv_usec)/1000);
      // Manul-Manan: new_qual would not contain correct value if GPU louvain is used
			// cerr << "\t new_qual= "<<new_qual << endl;

		}
		else
		{
			cerr<<"\t There is no node to reevaluate"<<endl;
      /**********Commented*****************/
			// s = "graph"+my2::to_string(count_del)+".tree";
      //       string line;
      //       ifstream myfile (s_old);
      //       ofstream myfile2 (s);
      //       if (myfile.is_open())
      //       {
      //           while ( getline (myfile,line) )
      //           {
      //               if (myfile2.is_open())
      //                   myfile2 << line << '\n';
      //               else
      //                   cout << "Unable to open output file";

      //           }
      //           myfile.close();
      //       }
      //       else
      //           cout << "Unable to open file";
      //       myfile2.close();
      /**********Commented*****************/
		}

	  }//if(fexists(name))
	  else
          {//if
            cerr<< "There is no delta_file associated with "<<name<<endl;
            /**********Commented*****************/
            // s = "graph"+my2::to_string(count_del)+".tree";
            // string line;
            // ifstream myfile (s_old);
            // ofstream myfile2 (s);
            // if (myfile.is_open())
            // {
            //     while ( getline (myfile,line) )
            //     {
            //         if (myfile2.is_open())
            //             myfile2 << line << '\n';
            //         else
            //             cout << "Unable to open output file";

            //     }
            //     myfile.close();
            // }
            // else
            //     cout << "Unable to open file";
            // myfile2.close();
            /**********Commented*****************/
      }

    GTOC;   // Manul's Change
	  gettimeofday(&eTime, NULL);
	  time_per_step = (double) ((eTime.tv_sec -sTime.tv_sec )*1000 + (eTime.tv_usec -  sTime.tv_usec)/1000);
	  cerr<<endl <<"**********************************************";
	  cerr<<endl<< "Timestep " <<j <<" addition starts"<<endl;

    GTIC("Addition");   // Manul's Change

	  count_add++;
	  name = filename_add+my2::to_string(j)+".txt";
	  nb_nodes_b = gr.nb_nodes;
	  string ss = s;
	  if(fexists(name))
	  {
		s = "graph"+my2::to_string(count_add)+".tree";

    /**********Commented*****************/
    // GTIC("find_NodCom add");    // Manul's Change
	  // 	nodes_in_comm = find_NodCom(s,nb_nodes_b);
    // GTOC;   // Manul's Change
    /**********Commented*****************/

    cout << "Point 1" << endl;

    // Manul start
    int num_of_comm = (*max_element(n2c.begin(), n2c.end())) + 1;
    nodes_in_comm.resize(num_of_comm);
    for(int node_i = 0; node_i < n2c.size(); node_i++) {
      nodes_in_comm[n2c[node_i]].push_back(node_i);
    }
    // Manul end

    cout << "Point 2" << endl;

    GTIC("buildNew_add");   // Manul's Change
	  	vect_new_edges = buildNewGraph_add(name,&gr);
    GTOC;   // Manul's Change
	  	
      cout << "Point 3" << endl;
      
      Graph g1 = gr;
	  	int nb_nodes_a = gr.nb_nodes;
	  	it = max_element(n2c.begin(), n2c.end());
	  	n_comm = *it +1;

      // Manul-Manan: since new nodes may be added, resizing is necessary
	  	n2c.resize(gr.nb_nodes);//resize the n2c at the next time step
      // Manul-Manan: iterating over the newly added nodes (if any)
	  	for(int i=nb_nodes_b;i<nb_nodes_a;i++)
	  	{
	  		pair<vector<int>::iterator, vector<long double>::iterator > p = gr.neighbors(i);
        // Manul-Manan: if the node has only one neighbour and that nbr is an old node, assign
        // Manul-Manan: this new node to the comm of the old nbr node
	  		if((gr.degrees[i]-gr.degrees[i-1] == 1) && (*(p.first) < nb_nodes_b)) //if the node has just one neighbor assign it to the community of its neighbor
	  		{
	  			n2c[i] = n2c[*(p.first)];
	  		}
	  		else//assign the node to a new community
	  		{
	  			n2c[i] = n_comm;
	  			n_comm++;
	  		}
	  	}

      cout << "Point 4" << endl;

      // Manul-Manan: Issue: where is n2c being updated? If the deletion of edges leads to changes
      // Manul-Manan: in the comm's, n2c must be updated before being used for edge addition
	  	init_quality(&g1, nb_calls);
	  	nb_calls++;
	  	Louvain c1(-1, precision, q);//call louvain constructor
	  	c1.init_partition_v(n2c);
	  	q->n2c = n2c;
	  	struct timeval sTime2, eTime2;
	  	gettimeofday(&sTime2, NULL);

      cout << "Point 5" << endl;

	  	//Re = nodToEval_b(nodes_in_comm,&g1,&c1, type,n2c,vect_new_edges, nb_nodes_b);
      /**********Commented*****************/
      // GTIC("nodeToEval_add");   // Manul's Change
	  	// Re = nodToEval_add(nodes_in_comm,&g1,&c1, type,n2c,vect_new_edges, nb_nodes_b);
      // GTOC;   // Manul's Change
      /**********Commented*****************/

      cout << "Point 6" << endl;
    
    /**********Commented*****************/
    // // Manul-Manan: Issue: what does this check accomplish?
	  // if(*max_element(Re.begin(),Re.end()) != 0)
    //   {
    //         //s.erase(remove_if(s.begin(), s.end(), static_cast<int(*)(int)>(isspace)), s.end());
	  //   if(fexists(s))
	  //   {
	  // 	 ifstream infile;
	  // 	 infile.open(s.c_str(),std::ifstream::out | std::ifstream::trunc);
	  // 	 if(!infile.is_open() || infile.fail())
	  // 	 {
	  // 	    infile.close();
	  // 	    printf("\n Error: failed to erase file content.");
	  // 	 }
	  // 	 infile.close();
	  // 	 }
	  //     else
	  // 	      cerr<<"not exists"<<endl;
    //     }
    /**********Commented*****************/

        s = "graph"+my2::to_string(count_add)+".tree";
	  	gettimeofday(&eTime2, NULL);
	  	time_per_step2 = (double) ((eTime2.tv_sec -sTime2.tv_sec )*1000 + (eTime2.tv_usec -  sTime2.tv_usec)/1000);
      cout << "Point 7" << endl;
	  	cerr<<"\t Time for nodToEval(milliseconds): "<<time_per_step2<<endl;

      // Manul start
      GTIC("part after add mem");
      GPUWrapper gpuWrapper(&c1, n2c, true);
      GTOC;

      GTIC("node eval add gpu");
      int R_size = gpuWrapper.gpuNodeEvalAdd(vect_new_edges);
      GTOC;

      // Manul end

	  	nodes_in_comm.clear();
	  	vector<vector<int> >(nodes_in_comm).swap(nodes_in_comm);
	  	vect_new_edges.clear();
	  	vector<pair<unsigned int,unsigned int> >(vect_new_edges).swap(vect_new_edges);
	  	cerr<<"R percentage : "<<(double)R_size/(double)(c1.qual)->g.nb_nodes *100 <<endl;

	  	if(R_size!=0)
	  	{
	  		nb_calls =0;
	  		improvement = true;
	  		quality = (c1.qual)->quality();// modularity of current partition( for the very first step each node in one comm)
	  		level = 0;
	  		gettimeofday(&level_time_s,NULL);

        // Manul start
        GTIC("partitioning after add");
        gpuWrapper.gpuLouvain(n2c);
        GTOC;
        // Manul end

        /**********Commented*****************/
        // GTIC("partitioning after add");   // Manul's Change
	  		// do {
        //   GTIC("Level " + to_string(level));  // Manul's Change
	  		// 	if (verbose)
	  		// 	{
	  		// 		cerr <<"level " << level << ":\n";
	  		// 		cerr << "\t network size: "
	  		// 				<< (c1.qual)->g.nb_nodes << " nodes, "
	  		// 				<< (c1.qual)->g.nb_links << " links, "
	  		// 				<< (c1.qual)->g.total_weight << " weight" << endl;
	  		// 	}
        //   GTIC("one_level");    // Manul's Change
	  		// 	improvement = c1.one_level();// do the first pass
        //   GTOC;   // Manul's Change
	  		// 	new_qual = (c1.qual)->quality();// new_qual is the new modularity
	  		// 	if (++level==display_level)
	  		// 		(c1.qual)->g.display();
	  		// 	if (display_level==-1)
	  		// 		c1.display_partition(count_add);

        //   GTIC("part2graph");   // Manul's Change
	  		// 	g1 = c1.partition2graph_binary();// find the new graph for the next pass
        //   GTOC;

	  		// 	init_quality(&g1, nb_calls);
	  		// 	nb_calls++;

	  		// 	c1 = Louvain(-1, precision, q);
	  		// 	quality = new_qual;
	  		// 	if (n2c.size() != 0 && level==1) // do at least one more computation if partition is provided
	  		// 		improvement=true;
          
        //     GTOC; // Manul's Change
	  		// 	} while(improvement);
        //   GTOC; // Manul's Change
        /**********Commented*****************/

	  			gettimeofday(&level_time_e,NULL);
	  			level_time += (double) ((level_time_e.tv_sec -level_time_s.tv_sec )*1000 + (level_time_e.tv_usec -  level_time_s.tv_usec)/1000);
	  			// cerr << "\t new_qual= "<<new_qual << endl;
	  		}
	  		else
	  		{
	  			cerr<<"\t There is no node to reevaluate"<<endl;
	  		}

	  	  }//if(fexists(name))
	  	  else
          {
            cerr<< "There is no delta_file associated with "<<name<<endl;
            s = "graph"+my2::to_string(count_add)+".tree";

          }

        GTOC;   // Manul's Change

	  	  gettimeofday(&eTime, NULL);
	  	  time_per_step = (double) ((eTime.tv_sec -sTime.tv_sec )*1000 + (eTime.tv_usec -  sTime.tv_usec)/1000);
	  	  cerr<<"\t Time for step "<< j+1<<":(milliseconds) "<<time_per_step<<endl;
    GTOC;   // Manul's Change
  }//end of time step j
  GTOC;     // Manul's Change

  gettimeofday(&endTime, NULL);
  elapsedSeconds = (double) ((endTime.tv_sec-startTime.tv_sec) * 1000 + (endTime.tv_usec - startTime.tv_usec)/1000);
  cerr<< "execution time (milliseconds): "<<elapsedSeconds<<endl;

  cout << "\n\n\n";       // Manul's Change
  Timer::display_stat_h(); // Manul's Change
  Timer::display_summary(0);
  cout << '\n';
  Timer::display_summary(1);
}

