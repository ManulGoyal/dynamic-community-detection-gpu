#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <unordered_map>

#define TIC(id, context) Timer::start_timer(id, context);
#define TOC(id) Timer::stop_timer(id);
#define GTIC(context) Timer::start_timer(context);
#define GTOC Timer::stop_timer();

using namespace std;

class Metadata {
public:
    string context;
    int exec_count;
    double exec_duration;       // in milliseconds
    chrono::_V2::system_clock::time_point last_begin, first_begin;
    bool is_init, is_running;

    Metadata() {
        exec_count = 0;
        exec_duration = 0.0;
        is_running = false;
        is_init = false;
    } 
};

class Metanode {
public:
    string context;
    chrono::_V2::system_clock::time_point begin;
    double exec_duration;       // in milliseconds

    bool is_running;
    Metanode* parent;
    vector<Metanode*> adj;

    Metanode(string const & context) : 
        context(context), parent(NULL), exec_duration(0.0), is_running(false) {}
};

namespace Timer {

    void start_timer(int id, string const & context);

    void stop_timer(int id);

    void start_timer(string const & context);

    double stop_timer();

    void display_stats();

    void display_stat_h();

    void display_summary(int level);
};

#endif