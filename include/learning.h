#ifndef __LEARNING_H
#define __LEARNING_H

#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <libxml++/libxml++.h>
#include <libxml++/parsers/textreader.h>
#include "state.h"

using namespace std;

class Layer
{
 public:
  string id;
  int type;
  int n_nodes;
  void (*activation)(gsl_vector*);
  int target_id;
  gsl_vector* target;
  gsl_vector* output;
  gsl_vector* delta;
  vector<string> features;
  vector< vector<Layer*> > past_connections;

  Layer();
  Layer(string _id, int _type, int _n_nodes, void (*_activation)(gsl_vector*), int _target_id, vector<string>& _features, vector< vector<Layer*> >& _past_connections);
  ~Layer();
  Layer* clone(unordered_map<string,Layer*>& past_layers);
  void set_target(int _target_id);
  string to_string();
  vector<double>* get_probability_vector(bool output_vector);
  double get_probability();
};

class Step
{
 public:
  int id;
  State* state;
  vector<Layer*> hidden_layers;
  vector<Layer*> visible_layers;

  Step(int _id, State* _state);
  ~Step();
  Step* clone(unordered_map<string,Layer*>& past_layers);
  double get_probability();
};

class LearningInstance
{
 public:
  vector<Step*> steps;

  LearningInstance();
  ~LearningInstance();
  LearningInstance* clone(unordered_map<string,Layer*>& past_layers);
  double get_probability();
  double get_probability_of_last_step();
  double get_probability_of_last_decision(int decision);
  vector<double>* get_probability_vector_of_last_decision(int decision);
  vector<double>* get_hidden_vector_of_last_step();
};

class LearningModel
{
  float learning_rate;
  float weight_decay;
  float momentum;
  float init_range;
  float init_range_emiss;
  int n_features;
  int n_connections;
  bool isTRBM;  //FF=false, TRBM=true
  bool useCD;  //set to false

  unordered_map<int, int> layer_size;
  int hidden_layer_type;
  int dummy_layer_type;
  Layer *dummy_layer;

  gsl_rng * rng;
 
  bool bias_initialized;
  unordered_map<int, gsl_vector*> __initB;
  unordered_map<int, gsl_vector*> B;
  unordered_map< string, unordered_map<int,gsl_vector*> > *W_feature;
  unordered_map< int, unordered_map<int,gsl_matrix*> > W;
  unordered_map< int, unordered_map<int,gsl_matrix*> > W_RBM;
  unordered_map< int, unordered_map<int,gsl_matrix*> > *W_past;

  unordered_map<int, gsl_vector*> dB;
  unordered_map< string, unordered_map<int,gsl_vector*> > *dW_feature;
  unordered_map< int, unordered_map<int,gsl_matrix*> > dW;
  unordered_map< int, unordered_map<int,gsl_matrix*> > dW_RBM;
  unordered_map< int, unordered_map<int,gsl_matrix*> > *dW_past;
  unordered_map<int, int> n_dB;
  unordered_map< string, unordered_map<int,int> > *n_dW_feature;
  unordered_map< int, unordered_map<int,int> > n_dW;
  unordered_map< int, unordered_map<int,int> > n_dW_RBM;
  unordered_map< int, unordered_map<int,int> > *n_dW_past;

  void read_from_file(string filename);
  void initialize_W();
  void create_dW();
  void reset_dW();
  void update_W();
  int update_network(LearningInstance* instance, State* _state, int decision, int target, bool new_step);
  gsl_vector* get_network_bias(Layer* layer, bool use_past_targets);
  void set_output(Layer* layer, bool use_past_targets);
  void set_directed_output(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets);
  void set_RBM_output_iterative(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets);
  void set_RBM_output_marginalized(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets);
  void set_RBM_target(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets);
  void forward_propagate_directed(Step *step, bool train);
  void forward_propagate_RBM(Step *step, bool train);
  void learn_from_layer(Layer* layer);
  void learn_directed(vector<Layer*> visible_layers, vector<Layer*> hidden_layers);
  void TRBM_contrastive_divergence(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, int N);
  void TRBM_direct_gradient(vector<Layer*> visible_layers, vector<Layer*> hidden_layers);
  void learn_from_step(Step *step);
  void learn_from_step_directed(Step *step);
  void learn_from_step_RBM(Step *step);
  void learn_bias(Step *step);
  string write_vector(gsl_vector* V, string prefix);
  string write_matrix(gsl_matrix* M, string prefix);

 public:
  LearningModel(string filename);
  LearningModel(string _model_type, int _n_hidden, int _n_features, int _n_connections, unordered_map<int,int>* _decision_size);
  ~LearningModel();
  string to_string();
  LearningInstance* get_new_instance();
  LearningInstance* clone_instance(LearningInstance* instance);
  void save(string filename);
  void save_weights_binary(string filename_info, string filename);
  void read_weights_binary(string filename_info, string filename);
  void update(LearningInstance* instance, State* state, int decision, int target, bool new_step);
  void process_last_step(LearningInstance* instance, bool training);
  void learn(LearningInstance* instance, int iter);
  int set_target_of_last_decision(LearningInstance* instance, int decision, int target);
  void set_learning_rate(float _learning_rate);
  void set_weight_decay(float _weight_decay);
  void set_momentum(float _momentum);
  void clear_momentum();
};

void sigmoid(gsl_vector* X);
void derivative_sigmoid(gsl_vector* sigmoidX);
void soft_max(gsl_vector* X);
void sample(gsl_vector* X, gsl_rng * rng);
void print_vector(FILE *stream, const gsl_vector * V);
double matrix_norm(gsl_matrix * M);

#endif
