#ifndef __STATE_H
#define __STATE_H

#include <iostream>
#include <deque>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <libxml++/libxml++.h>
#include <libxml++/parsers/textreader.h>
#include "parsetree.h"
#include "sentence.h"

using namespace std;

class Feature
{
  string field_name;
  string dependency_function;
  string data_structure;
  int offset;

 public:
  Feature();
  void from_string(string s);
  string to_string();
  string get_value(deque<int>& S, deque<int>& Q, ParseTree* parsetree);
};

class FeatureModel
{
 public:
  vector<Feature> features;
  FeatureModel(string filename);
  string to_string();
};

class Connection
{
 public:
  Feature present_feature;
  Feature past_feature;
  int offset;

  Connection(Feature _present_feature, Feature _past_feature, int _offset);
  string to_string();
};

class ConnectionModel
{
 public:
  vector<Connection> connections;
  ConnectionModel(string filename);
  string to_string();
};


class State
{
  FeatureModel* feature_model;
  ConnectionModel* connection_model;
  deque<int> S;
  deque<int> Q;
  string previous_operation;
  vector<string> features;
  vector< pair<string,string> > connection_features;  //pair<past_feature, present_feature>

 public:
  State(FeatureModel* _feature_model, ConnectionModel* _connection_model);
  ~State();
  State* clone();
  string to_string();
  void stack_push(int val);
  int stack_top();
  bool stack_empty();
  void stack_pop();
  void queue_enqueue(int val);
  int queue_front();
  int queue_element(int position);
  bool queue_empty();
  int queue_size();
  void queue_dequeue();
  string get_previous_operation();
  void set_previous_operation(string op);
  vector<string> get_features();
  void recalculate_features(ParseTree* parsetree);
  vector<int> get_compatible_connections(vector<State*> past_states);
};


#endif
