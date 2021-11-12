#include <iostream>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stack>
#include "timer.h"
using namespace std;
using namespace chrono;

ostream & operator<<(ostream & out, Metadata const & meta) { 
  if(!meta.is_init) return out;

  out << meta.context << ":\n"
      << "Execution count: " << meta.exec_count << '\n' 
      << "Total execution duration: " << fixed << setprecision(3) << meta.exec_duration << " ms\n"; 

  return out;
}

ostream & operator<<(ostream & out, Metanode const & meta) { 

  if(meta.parent) {
    double frac = meta.exec_duration * 100 / meta.parent->exec_duration;
    out << meta.context << ": " << fixed << setprecision(3) << meta.exec_duration << " ms [" << frac << "%]\n"; 
  }
  else {
    out << meta.context << ": " << fixed << setprecision(3) << meta.exec_duration << " ms\n";
  }
  return out;
}

namespace Timer {
    const int MAX_USAGE = 100000;
    vector<Metadata> monitor(MAX_USAGE);
    stack<Metanode*> timer;
    vector<Metanode*> roots;

    chrono::_V2::system_clock::time_point start;
    bool is_running_global = false;

    void start_timer(int id, string const & context) {
        if(id >= MAX_USAGE) { cerr << "Exceeded timer usage limit\n"; return; }
        if(!monitor[id].is_init) {
            monitor[id].is_init = true;
            monitor[id].is_running = true;
            monitor[id].context = context;
            monitor[id].last_begin = monitor[id].first_begin = chrono::high_resolution_clock::now();
        } else if(!monitor[id].is_running) {
            monitor[id].is_running = true;
            monitor[id].last_begin = chrono::high_resolution_clock::now();
        } else {
            cerr << "Timer for \"" << monitor[id].context << "\" is already running" << endl; 
        }
    }

    void stop_timer(int id) {
        auto now = chrono::high_resolution_clock::now();
        if(id >= MAX_USAGE) { cerr << "Exceeded timer usage limit\n"; return; }

        if(!monitor[id].is_init) {
            cerr << "Timer for id = " << id << " was never started" << endl; 
        } else if(monitor[id].is_running) {
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now-monitor[id].last_begin).count();
            monitor[id].is_running = false;
            monitor[id].exec_count++;

            monitor[id].exec_duration += duration / 1000.0;
        } else {
            cerr << "Timer for \"" << monitor[id].context << "\" is already stopped" << endl;
        }
    }

    void display_stats(int id) {
        if(id >= MAX_USAGE) { cerr << "Exceeded timer usage limit\n"; return; }

        if(!monitor[id].is_init) {
            cerr << "Timer for id = " << id << " was never started\n";
        } else {
            cout << monitor[id] << '\n';
        }
    }

    void display_stats() {

        sort(monitor.begin(), monitor.end(), [](
            Metadata const & meta1, 
            Metadata const & meta2) {
                return meta1.first_begin < meta2.first_begin;
            });
        
        for(auto& meta: monitor) {
            if(meta.is_init) cout << meta << '\n';
        }
    }

    void start_timer(string const & context) {
        Metanode* meta = new Metanode(context);
        if(timer.empty()) {
            roots.push_back(meta);
        } else {
            meta->parent = timer.top();
            timer.top()->adj.push_back(meta);
        }
        timer.push(meta);
        meta->begin = chrono::high_resolution_clock::now();
    }

    double stop_timer() {
        auto end = chrono::high_resolution_clock::now();

        if(timer.empty()) {
            cerr << "No timer is running\n";
            return -1.0;
        }

        Metanode* meta = timer.top();

        timer.pop();
        auto start = meta->begin;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

        meta->exec_duration = duration / 1000.0;

        return meta->exec_duration;
        
    }

    void display_stat_h_util(Metanode* meta, int level) {
        for (int i = 0; i < level; i++)
        {
            cout << '\t';
        }
        cout << (*meta);
        
        for(auto x : meta->adj) {
            display_stat_h_util(x, level+1);
        }
    }

    void summarize_by_context(Metanode* meta, bool root, unordered_map<string, double> &m) {
        if(!root) m[meta->context] += (meta->exec_duration);

        for(auto x : meta->adj) {
            summarize_by_context(x, false, m);
        }
    }

    void display_stat_h() {
        int level = 0;

        for (auto x : roots)
        {
            display_stat_h_util(x, 0);
        }
        
    }

    void display_summary_util(Metanode* meta, int level, int req_level) {
        if(level == req_level) {
            unordered_map<string, double> m;
            summarize_by_context(meta, true, m);

            cout << (meta->context) << ": " << fixed << setprecision(3) << meta->exec_duration << "\n";
            double parent_dur = meta->exec_duration;
            for (auto x : m)
            {
                double frac = x.second * 100 / parent_dur;
                cout << '\t' << x.first << ": " << fixed << setprecision(3) << x.second << " [" << frac << "%]\n";  
            }
            return;
        }

        for(auto x : meta->adj) {
            display_summary_util(x, level+1, req_level);
        }
    }

    void display_summary(int level) {
        for (auto root : roots)
        {
            display_summary_util(root, 0, level);
        }
        
    }
};

