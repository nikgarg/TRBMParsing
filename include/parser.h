#ifndef __PARSER_H
#define __PARSER_H

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <algorithm>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "state.h"
#include "learning.h"
#include "data.h"
#include "parsetree.h"
#include "evaluator.h"

using namespace std;

class ParserConfiguration
{
  LearningModel *learning_model;

 public:
  LearningInstance *learning_instance;
  ParseTree *partial_parse_tree;
  State* state;
  vector< pair<string,double> > history;
  double log_probability;

  ParserConfiguration();
  ParserConfiguration(vector<Word*>& words, FeatureModel *feature_model, ConnectionModel *connection_model, LearningModel *learning_model);
  ~ParserConfiguration();
  ParserConfiguration* clone();
  void bury();
  void make_transition(string action, double probability, int labelid);
  int possible_transitions(unordered_map<string,bool>& transitions);
};


struct LessParserConfiguration : public binary_function<ParserConfiguration*, ParserConfiguration*, bool>
{
  bool operator()(const ParserConfiguration* lhs, const ParserConfiguration* rhs) const
  {
    return lhs->log_probability < rhs->log_probability;
  }
};

struct GreaterParserConfiguration : public binary_function<ParserConfiguration*, ParserConfiguration*, bool>
{
  bool operator()(const ParserConfiguration* lhs, const ParserConfiguration* rhs) const
  {
    return lhs->log_probability > rhs->log_probability;
  }
};

struct LessLabelProbability : public binary_function<pair<int,double>*, pair<int,double>*, bool>
{
  bool operator()(const pair<int,double>& lhs, const pair<int,double>& rhs) const
  {
    return lhs.second < rhs.second;
  }
};

struct GreaterLabelProbability : public binary_function<pair<int,double>*, pair<int,double>*, bool>
{
  bool operator()(const pair<int,double>& lhs, const pair<int,double>& rhs) const
  {
    return lhs.second > rhs.second;
  }
};

class Parser
{
  FeatureModel *feature_model;
  ConnectionModel *connection_model;
  LearningModel *learning_model;
  Evaluator *evaluator;
  Data *data;
  int beam_size;
  int label_branch_size;
  float learning_rate_init;
  float learning_rate_red_rate;
  float learning_rate_max_red;
  float learning_rate;
  float weight_decay_init;
  float weight_decay_red_rate;
  float weight_decay_max_red;
  float weight_decay;
  float momentum;

  unordered_map<string,string> params;

  unordered_map<string, int> decisions2id;
  unordered_map<int, int> decision_size;
  unordered_map<string, int> action2id;

  double simulate_sentence(vector<Word*>& words, ParseTree *parsetree, bool learn, bool get_repr, vector< vector<double>* >& repr, int iter);
  ParseTree* predict_parse_tree(vector<Word*>& words);
  void read_from_file();
  void send_weight_changes(string filename);
  void get_file_from_server(string filename);

 public:
  Parser(unordered_map<string,string> _params, bool train_new, string model_type);
  ~Parser();
  void train();
  void test();
  void print_word_representations(string data_file, string repr_file);
  string get_results();
  void save();
};

#endif
