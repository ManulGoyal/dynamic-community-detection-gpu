// File: graph.cpp
// -- simple graph handling source file
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


#include "graph.h"

using namespace std;


Graph::Graph(char *filename, int type) {
	cerr << "graph creating starts"<<endl;
  ifstream finput;
  finput.open(filename,fstream::in);
  if (finput.is_open() != true) {
    cerr << "The file " << filename << " does not exist" << endl;
    exit(EXIT_FAILURE);
  }

  unsigned long long nb_links = 0ULL;

  unsigned long long self_loops_n = 0ULL;

  while (!finput.eof()) {
    unsigned int src, dest;
    long double weight = 1.0L;

    if (type==WEIGHTED) {
      finput >> src >> dest >> weight;
    } else {
      finput >> src >> dest;
    }
    
    if (finput) {
      if (links.size()<=max(src,dest)+1) {
        links.resize(max(src,dest)+1);
      }

      links[src].push_back(make_pair(dest,weight));
      if (src!=dest) {
        links[dest].push_back(make_pair(src,weight));
      } else self_loops_n++;

      nb_links += 1ULL;
    }
  }

  finput.close();
  cerr<<"nb_links: "<< nb_links<<endl;
  cerr<<"nb_nodes: "<< links.size()<<endl;
  cerr<<"nb_self_loops: "<< self_loops_n<<endl;
}

void
Graph::renumber(int type, char *filename) 
{
  vector<int> linked(links.size(),-1);
  vector<int> renum(links.size(),-1);
  int nb = 0;

  ofstream foutput;
  foutput.open(filename, fstream::out);
  
  for (unsigned int i=0 ; i<links.size() ; i++) 
  {
    if (links[i].size() > 0)
      linked[i] = 1;
  }
  
  for (unsigned int i=0 ; i<links.size() ; i++)
  {
    if (linked[i]==1) 
    { 
      renum[i] = nb++;
      foutput << i << " " << renum[i] << endl;
    }
  }

  for (unsigned int i=0 ; i<links.size() ; i++) 
  {
    if (linked[i]==1) 
    {
      for (unsigned int j=0 ; j<links[i].size() ; j++) 
      {
  	   links[i][j].first = renum[links[i][j].first];
      }
      links[renum[i]] = links[i];
    }
  }
  links.resize(nb);
}

void
Graph::clean(int type) {
  unsigned long long old_tot = 0ULL, new_tot = 0ULL; 
  for (unsigned int i=0 ; i<links.size() ; i++) {
    map<int, long double> m;
    map<int, long double>::iterator it;

    for (unsigned int j=0 ; j<links[i].size() ; j++) {
      it = m.find(links[i][j].first);
      if (it==m.end())
	m.insert(make_pair(links[i][j].first, links[i][j].second));
      else if (type==WEIGHTED)
      	it->second+=links[i][j].second;
    }
    
    vector<pair<int, long double> > v;
    for (it = m.begin() ; it!=m.end() ; it++)
      v.push_back(*it);
    
    // if(links[i].size() != 2 * v.size()) {
    //   cerr << "ERR: Increasing during clean: node " << i << " " << links[i].size() << " to " << v.size() << '\n';
      
    // }
    old_tot += links[i].size();
    new_tot += v.size();
    links[i].clear();
    links[i] = v;
  }
  cout << "edges before clean " << old_tot << '\n';
  cout << "edges after clean " << new_tot << '\n';
}

void
Graph::display(int type) {
  for (unsigned int i=0 ; i<links.size() ; i++) {
    for (unsigned int j=0 ; j<links[i].size() ; j++) {
      int dest = links[i][j].first;
      long double weight = links[i][j].second;
      if (type==WEIGHTED)
	cout << i << " " << dest << " " << weight << endl;
      else
	cout << i << " " << dest << endl;
    }
  }
}

void
Graph::display_binary(char *filename, char *filename_w, int type) {
  ofstream foutput;
  foutput.open(filename, fstream::out | fstream::binary);

  int s = links.size();

  cout << "display_binary nodes (after clean): " << links.size() << endl;

  // outputs number of nodes
  foutput.write((char *)(&s),sizeof(int));
  
  // outputs cumulative degree sequence
  unsigned long long tot = 0ULL;
  for (int i=0 ; i<s ; i++) {
    tot += (unsigned long long)links[i].size();
    foutput.write((char *)(&tot),sizeof(unsigned long long));
  }

  cout << "display_binary links (after clean): " << tot << endl;

  // outputs links
  for (int i=0 ; i<s ; i++) {
    for (unsigned int j=0 ; j<links[i].size() ; j++) {
      int dest = links[i][j].first;
      foutput.write((char *)(&dest),sizeof(int));
    }
  }
  foutput.close();

  // outputs weights in a separate file
  if (type==WEIGHTED) {
    ofstream foutput_w;
    foutput_w.open(filename_w,fstream::out | fstream::binary);
    for (int i=0 ; i<s ; i++) {
      for (unsigned int j=0 ; j<links[i].size() ; j++) {
	long double weight = links[i][j].second;
	foutput_w.write((char *)(&weight),sizeof(long double));
      }
    }
    foutput_w.close();
  }
}
