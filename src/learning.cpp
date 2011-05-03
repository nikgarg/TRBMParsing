#include "learning.h"

Layer::Layer()
{
}

Layer::Layer(string _id, int _type, int _n_nodes, void (*_activation)(gsl_vector*), int _target_id, vector<string>& _features, vector< vector<Layer*> >& _past_connections)
{
  id = _id;
  type = _type;
  n_nodes = _n_nodes;
  activation = _activation;
  target_id = _target_id;
  target = gsl_vector_calloc(n_nodes);
  output = gsl_vector_calloc(n_nodes);
  delta = gsl_vector_calloc(n_nodes);
  if(target_id!=-1)
    gsl_vector_set(target, target_id, 1);
  features = _features;
  past_connections = _past_connections;
}

Layer::~Layer()
{
  gsl_vector_free(target);
  gsl_vector_free(output);
  gsl_vector_free(delta);
}

Layer* Layer::clone(unordered_map<string,Layer*>& past_layers)
{
  Layer *C = new Layer();
  C->id = id;
  C->type = type;
  C->n_nodes = n_nodes;
  C->activation = activation;
  C->target_id = target_id;
  C->target = gsl_vector_alloc(n_nodes);
  gsl_vector_memcpy(C->target, target);
  C->output = gsl_vector_alloc(n_nodes);
  gsl_vector_memcpy(C->output, output);
  C->delta = gsl_vector_alloc(n_nodes);
  gsl_vector_memcpy(C->delta, delta);
  C->features = features;
  for(unsigned int i=0;i<past_connections.size();i++)
    {
      vector<Layer*> p = past_connections[i];
      vector<Layer*> p_clone(p.size());
      for (unsigned int j=0;j<p.size();j++)
	{
	  p_clone[j] = past_layers[p[j]->id];
	}
      C->past_connections.push_back(p_clone);
    }
  return C;
}

void Layer::set_target(int _target_id)
{
  target_id = _target_id;
  if(target_id!=-1)
    gsl_vector_set(target, target_id, 1);
}


string Layer::to_string()
{
  ostringstream result;
  result<<"Layer-> id:\""<<id<<"\" type:\""<<type<<"\" nodes:"<<n_nodes<<" target:"<<target_id;
  result<<" features:[";
  for(unsigned int i=0;i<features.size();i++)
    result<<features[i]<<",";
  result<<"]";
  return result.str();
}


vector<double>* Layer::get_probability_vector(bool output_vector)
{
  vector<double>* prob = new vector<double>(n_nodes, 0);
  if(output_vector)
    for(int i=0;i<n_nodes;i++)
      (*prob)[i] = log(gsl_vector_get(output, i));
  else
    for(int i=0;i<n_nodes;i++)
      (*prob)[i] = log(gsl_vector_get(target, i));
  return prob;
}

double Layer::get_probability()
{
  double prob = 0;
  if(target_id>=0 && target_id<n_nodes)
    prob = log(gsl_vector_get(output, target_id));
  return prob;
}


Step::Step(int _id, State* _state)
{
  id = _id;
  state = _state;
}

Step::~Step()
{
  for(unsigned i=0;i<hidden_layers.size();i++)
    delete hidden_layers[i];
  for(unsigned i=0;i<visible_layers.size();i++)
    delete visible_layers[i];
  delete state;
}

Step* Step::clone(unordered_map<string,Layer*>& past_layers)
{
  Step* C = new Step(id, state->clone());

  for(unsigned i=0;i<hidden_layers.size();i++)
    {
      Layer* clone_layer = hidden_layers[i]->clone(past_layers);
      C->hidden_layers.push_back(clone_layer);
      past_layers[clone_layer->id] = clone_layer;      
    }

  for(unsigned i=0;i<visible_layers.size();i++)
    {
      Layer* clone_layer = visible_layers[i]->clone(past_layers);
      C->visible_layers.push_back(clone_layer);
      past_layers[clone_layer->id] = clone_layer;      
    }

  return C;
}

double Step::get_probability()
{
  double prob = 0;
  for(unsigned d=0; d<visible_layers.size(); d++)
    prob += visible_layers[d]->get_probability();      
  return prob;
}


LearningInstance::LearningInstance()
{
}

LearningInstance::~LearningInstance()
{
  for(unsigned int i=0;i<steps.size();i++)
    delete steps[i];
}

LearningInstance* LearningInstance::clone(unordered_map<string,Layer*>& past_layers)
{
  LearningInstance *C = new LearningInstance();
  for (unsigned int k=0;k<steps.size();k++)
    {
      Step* clone_step = steps[k]->clone(past_layers);
      C->steps.push_back(clone_step);
    }
  return C;
}

double LearningInstance::get_probability()
{
  double prob = 0;
  for (unsigned int i=0; i<steps.size(); i++)
    prob += steps[i]->get_probability();
  return prob;
}

double LearningInstance::get_probability_of_last_step()
{
  return steps.back()->get_probability();
}

double LearningInstance::get_probability_of_last_decision(int decision)
{
  Step *step = steps.back();
  for(unsigned i=0;i<step->visible_layers.size();i++)
    {
      Layer *layer = step->visible_layers[i];
      if(layer->type==decision)
	return layer->get_probability();
    }
  return 0;
}

vector<double>* LearningInstance::get_probability_vector_of_last_decision(int decision)
{
  Step *step = steps.back();
  for(unsigned i=0;i<step->visible_layers.size();i++)
    {
      Layer *layer = step->visible_layers[i];
      if(layer->type==decision)
	return layer->get_probability_vector(true);
    }
  return NULL;
}

vector<double>* LearningInstance::get_hidden_vector_of_last_step()
{
  Layer *layer = steps.back()->hidden_layers[0];
  return layer->get_probability_vector(false);
}


LearningModel::LearningModel(string filename)
{
  learning_rate = 0.005;
  weight_decay = 0.08;
  momentum = 0.9;
  init_range = 5e-2;
  init_range_emiss = 5e-4;
  hidden_layer_type = -1;
  dummy_layer_type = -2;

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rng, time(NULL));

  read_from_file(filename);
  bias_initialized = true;
  cerr<<to_string();
}


LearningModel::LearningModel(string _model_type, int _n_hidden, int _n_features, int _n_connections, unordered_map<int,int>* _decision_size)
{
  learning_rate = 0.005;
  weight_decay = 0.08;
  momentum = 0.9;
  init_range = 5e-2;
  init_range_emiss = 5e-4;
  hidden_layer_type = -1;
  dummy_layer_type = -2;

  gsl_rng_env_setup();
  rng = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rng, time(NULL));

  n_features = _n_features;
  n_connections = _n_connections;
  isTRBM = true;
  if(_model_type=="FF")
    isTRBM = false;
  useCD = false;

  bias_initialized = false;

  cerr<<"initializing Learning Model...";

  W_feature = new unordered_map< string, unordered_map<int,gsl_vector*> >[n_features];
  W_past = new unordered_map< int, unordered_map<int,gsl_matrix*> >[n_connections];

  layer_size[hidden_layer_type] = _n_hidden;
  layer_size[dummy_layer_type] = _n_hidden;
  for(unordered_map<int,int>::iterator it=_decision_size->begin(); it!=_decision_size->end(); it++)
    layer_size[it->first] = it->second;

  vector<string> _features;
  vector< vector<Layer*> > _past_connections(n_connections);
  dummy_layer = new Layer("dummy", dummy_layer_type, layer_size[dummy_layer_type], NULL, -1, _features, _past_connections);
  gsl_vector_set_all(dummy_layer->output, 1);
  gsl_vector_set_all(dummy_layer->target, 1);

  cerr<<"allocating weights memory...";
  for (int k=0;k<n_features;k++)
    W_feature[k]["#IGNORE#"][hidden_layer_type] = gsl_vector_calloc(layer_size[hidden_layer_type]);
  
  for(unordered_map<int, int>::iterator it=layer_size.begin(); it!=layer_size.end(); it++)
    {
      __initB[it->first] = gsl_vector_alloc(it->second);
      B[it->first] = gsl_vector_alloc(it->second);
      W[hidden_layer_type][it->first] = gsl_matrix_alloc(layer_size[hidden_layer_type], it->second);
      W_RBM[hidden_layer_type][it->first] = gsl_matrix_alloc(layer_size[hidden_layer_type], it->second);
    }

  for (int k=0;k<n_connections;k++)
    W_past[k][hidden_layer_type][hidden_layer_type] = gsl_matrix_alloc(layer_size[hidden_layer_type], layer_size[hidden_layer_type]);

  cerr<<"initializing weights...";
  initialize_W();
  create_dW();
  cerr<<"done!\n";
  cerr<<to_string();
}

LearningModel::~LearningModel()
{
  gsl_rng_free (rng);
  delete dummy_layer;

  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_matrix* M;
  int l1,l2;
  string l3;
  int k;

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      l1 = it_V->first;
      V = it_V->second;
      gsl_vector_free(V);
      gsl_vector_free(dB[l1]);
      gsl_vector_free(__initB[l1]);
    }

  for (k=0;k<n_features;k++)
    {
      for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
	{
	  l3 = it1->first;
	  for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	    {
	      l2 = it_V->first;
	      V = it_V->second;
	      gsl_vector_free(V);
	      gsl_vector_free(dW_feature[k][l3][l2]);
	    }
	}
    }
  delete[] W_feature;
  delete[] dW_feature;
  delete[] n_dW_feature;
  
  for(it=W.begin(); it!=W.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  gsl_matrix_free(M);
	  gsl_matrix_free(dW[l1][l2]);
	}
    }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  gsl_matrix_free(M);
	  gsl_matrix_free(dW_RBM[l1][l2]);
	}
    }

  for(k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      {
	l1 = it->first;
	for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	  {
	    l2 = it_M->first;
	    M = it_M->second;
	    gsl_matrix_free(M);
	    gsl_matrix_free(dW_past[k][l1][l2]);
	  }
      }
  delete[] W_past;
  delete[] dW_past;
  delete[] n_dW_past;
}

string LearningModel::to_string()
{
  ostringstream result;

  result<<"Learning Model:\n"
	<<"Model type: "<<(isTRBM?"TRBM":"FF")<<"\n"
	<<"TRBM Training: "<<(useCD?"Constrastive Divergence":"Direct Gradient")<<"\n"
	<<"learning rate: "<<learning_rate<<"\n"
	<<"weight decay:  "<<weight_decay<<"\n"
	<<"momentum:      "<<momentum<<"\n"
	<<"# features types:   "<<n_features<<"\n"
	<<"# connection types: "<<n_connections<<"\n"
	<<"# hidden nodes: "<<layer_size[hidden_layer_type]<<"\n"
	<<"# dummy nodes:  "<<layer_size[dummy_layer_type]<<"\n";
 
  return result.str();
}

LearningInstance* LearningModel::get_new_instance()
{
  LearningInstance* instance = new LearningInstance();
  return instance;
}

LearningInstance* LearningModel::clone_instance(LearningInstance* instance)
{
  unordered_map<string,Layer*> past_layers;
  past_layers["dummy"] = dummy_layer;
  LearningInstance* C = instance->clone(past_layers);
  return C;
}

int LearningModel::update_network(LearningInstance* instance, State* _state, int decision, int target, bool new_step)
{
  if(new_step)
    {
      State *state = _state->clone();
      vector<State*> past_states(instance->steps.size()); 
      for(unsigned i=0;i<instance->steps.size();i++)
	past_states[i] = instance->steps[i]->state;
      vector<int> compatible_connections = state->get_compatible_connections(past_states);
      vector< vector<Layer*> > past_connections(n_connections);
      for(int k=0;k<n_connections;k++)
	{
	  if(compatible_connections[k]!=-1)
	    {
	      Step* past_step = instance->steps[compatible_connections[k]];
	      //cout<<"connection type:"<<k<<": step "<<past_step->id<<" -> present\n";
	      past_connections[k].insert(past_connections[k].end(), past_step->hidden_layers.begin(), past_step->hidden_layers.end());
	    }
	  else
	    {
	      //past_connections_visible[k].push_back(dummy_layer);
	      //past_connections_hidden[k].push_back(dummy_layer);
	    }
	}
      
      vector<string> feature_vector = state->get_features();
      
      int step_id = instance->steps.size();
      Step* step = new Step(step_id, state);
      instance->steps.push_back(step);
       
      char buf[50];
      sprintf(buf, "Step:%d-%d",step_id,hidden_layer_type);
      Layer *hl = new Layer(string(buf), hidden_layer_type, layer_size[hidden_layer_type], sigmoid, -1, feature_vector, past_connections);
      step->hidden_layers.push_back(hl);
    }

  Step* step = instance->steps.back();
  vector<string> _feature_vector;
  vector< vector<Layer*> > _past_connections;
  char buf[50];
  sprintf(buf, "Step:%d-%d",step->id,decision);
  Layer *vl = new Layer(string(buf), decision, layer_size[decision], soft_max, target, _feature_vector, _past_connections);
  step->visible_layers.push_back(vl);
  
  return 0;
}

gsl_vector* LearningModel::get_network_bias(Layer* layer, bool use_past_targets)
{
  unsigned int i,k;
  gsl_vector* bias = gsl_vector_alloc(layer->n_nodes);
  gsl_vector_memcpy(bias, B[layer->type]);
  for(k=0;k<layer->features.size();k++)
    {
      gsl_vector_add(bias, W_feature[k][layer->features[k]][layer->type]);
    }

  for(k=0;k<layer->past_connections.size();k++)
    {
      for(i=0;i<layer->past_connections[k].size(); i++)
	{
	  Layer* past_layer = layer->past_connections[k][i];
	  if (use_past_targets)
	    gsl_blas_dgemv(CblasTrans, 1, W_past[k][past_layer->type][layer->type], past_layer->target, 1, bias);
	  else
	    gsl_blas_dgemv(CblasTrans, 1, W_past[k][past_layer->type][layer->type], past_layer->output, 1, bias);
	}
    }
  return bias;
}

void LearningModel::set_output(Layer* layer, bool use_past_targets)
{
  //cout<<"set_output() for layer:"<<layer->to_string()<<"\n";
  gsl_vector* bias = get_network_bias(layer, use_past_targets);
  gsl_vector_memcpy(layer->output, bias);
  layer->activation(layer->output);
  gsl_vector_free(bias);
}


void LearningModel::set_directed_output(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets)
{
  unsigned int v,h;
  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* bias = get_network_bias(hidden_layers[h], use_past_targets);
      gsl_vector_memcpy(hidden_layers[h]->output, bias);
      gsl_vector_free(bias);
      hidden_layers[h]->activation(hidden_layers[h]->output);
    }
  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector* bias = get_network_bias(visible_layers[v], use_past_targets);
      gsl_vector_memcpy(visible_layers[v]->output, bias);
      gsl_vector_free(bias);
      for(h=0;h<hidden_layers.size();h++)
	gsl_blas_dgemv(CblasTrans, 1, W[hidden_layers[h]->type][visible_layers[v]->type], hidden_layers[h]->output, 1, visible_layers[v]->output);
      visible_layers[v]->activation(visible_layers[v]->output);   
    }
}


void LearningModel::set_RBM_output_iterative(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets)
{
  vector<gsl_vector*> visible_layers_bias(visible_layers.size());
  vector<gsl_vector*> hidden_layers_bias(hidden_layers.size());
  vector<gsl_vector*> visible_layers_previous_output(visible_layers.size());
  unsigned int v,h, iter;
  double norm, previous_norm, change, relative_change;

  for(v=0;v<visible_layers.size();v++)
    visible_layers_bias[v] = get_network_bias(visible_layers[v], use_past_targets);
  for(h=0;h<hidden_layers.size();h++)
    hidden_layers_bias[h] = get_network_bias(hidden_layers[h], use_past_targets);

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector_memcpy(hidden_layers[h]->output, hidden_layers_bias[h]);
      hidden_layers[h]->activation(hidden_layers[h]->output);
    }
  previous_norm = 0;
  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_memcpy(visible_layers[v]->output, visible_layers_bias[v]);
      for(h=0;h<hidden_layers.size();h++)
	gsl_blas_dgemv(CblasTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], hidden_layers[h]->output, 1, visible_layers[v]->output);
      visible_layers[v]->activation(visible_layers[v]->output);
      previous_norm += gsl_blas_dnrm2(visible_layers[v]->output);
      visible_layers_previous_output[v] = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector_memcpy(visible_layers_previous_output[v], visible_layers[v]->output);
    }

  relative_change = 1;
  iter = 0;
  while (relative_change>0.0001)
    {
      for(h=0;h<hidden_layers.size();h++)
	{
	  gsl_vector_memcpy(hidden_layers[h]->output, hidden_layers_bias[h]);
	  for(v=0;v<visible_layers.size();v++)
	    gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], visible_layers[v]->output, 1, hidden_layers[h]->output);
	  hidden_layers[h]->activation(hidden_layers[h]->output);
	}

      norm = 0;
      change = 0;
      for(v=0;v<visible_layers.size();v++)
	{
	  gsl_vector_memcpy(visible_layers[v]->output, visible_layers_bias[v]);
	  for(h=0;h<hidden_layers.size();h++)
	    gsl_blas_dgemv(CblasTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], hidden_layers[h]->output, 1, visible_layers[v]->output);
	  visible_layers[v]->activation(visible_layers[v]->output);
	  norm += gsl_blas_dnrm2(visible_layers[v]->output);
	  gsl_vector_sub(visible_layers_previous_output[v], visible_layers[v]->output);
	  change += gsl_blas_dnrm2(visible_layers_previous_output[v]);
	  gsl_vector_memcpy(visible_layers_previous_output[v], visible_layers[v]->output);
	}
      relative_change = change/previous_norm;
      previous_norm = norm;
      iter += 1;
    }

  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_free(visible_layers_bias[v]);
      gsl_vector_free(visible_layers_previous_output[v]);
    }
  for(h=0;h<hidden_layers.size();h++)
    gsl_vector_free(hidden_layers_bias[h]);
}


void LearningModel::set_RBM_output_marginalized(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets)
{
  vector<gsl_vector*> visible_layers_bias(visible_layers.size());
  vector<gsl_vector*> hidden_layers_bias(hidden_layers.size());
  unsigned int v,h,i,j;

  for(v=0;v<visible_layers.size();v++)
    visible_layers_bias[v] = get_network_bias(visible_layers[v], use_past_targets);
  for(h=0;h<hidden_layers.size();h++)
    hidden_layers_bias[h] = get_network_bias(hidden_layers[h], use_past_targets);

  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_memcpy(visible_layers[v]->output, visible_layers_bias[v]);
      gsl_vector* X = gsl_vector_alloc(visible_layers[v]->n_nodes);
      for(h=0;h<hidden_layers.size();h++)
	{
	  if(v!=0)
	    {
	      gsl_vector* Y = gsl_vector_alloc(hidden_layers[h]->n_nodes);
	      gsl_matrix_get_col(Y, W_RBM[hidden_layers[h]->type][visible_layers[v-1]->type], visible_layers[v-1]->target_id);
	      gsl_vector_add(hidden_layers_bias[h], Y);
	      gsl_vector_free(Y);
	    }
	  for(i=0;i<(unsigned int)hidden_layers[h]->n_nodes;i++)
	    {
	      gsl_matrix_get_row(X, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], i);
	      double b = gsl_vector_get(hidden_layers_bias[h], i);
	      gsl_vector_add_constant(X, b);
	      for(j=0;j<(unsigned int)visible_layers[v]->n_nodes;j++)
		{
		  double x = gsl_vector_get(X,j);
                  if(x>100)
		    x = x; 
                  else
                    x = log(1+exp(x));
		  gsl_vector_set(X, j, x);
		}
	      gsl_vector_add(visible_layers[v]->output, X);
	    }
	}
      gsl_vector_free(X);
      double m = gsl_vector_max(visible_layers[v]->output);
      double sum = 0;
      for(j=0;j<(unsigned int)visible_layers[v]->n_nodes;j++)
	{
	  double x = gsl_vector_get(visible_layers[v]->output, j);
	  sum += exp(x-m);
	}
      sum = log(sum);
      sum += m;

      for(j=0;j<(unsigned int)visible_layers[v]->n_nodes;j++)
	{
	  double x = gsl_vector_get(visible_layers[v]->output, j);
	  x -= sum;
	  gsl_vector_set(visible_layers[v]->output, j, exp(x));
	}     
    }

  for(v=0;v<visible_layers.size();v++)
    gsl_vector_free(visible_layers_bias[v]);
  for(h=0;h<hidden_layers.size();h++)
    gsl_vector_free(hidden_layers_bias[h]);
}


void LearningModel::set_RBM_target(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, bool use_past_targets)
{
  unsigned int v,h;

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* bias = get_network_bias(hidden_layers[h], use_past_targets);
      gsl_vector_memcpy(hidden_layers[h]->target, bias);
      for(v=0;v<visible_layers.size();v++)
	if(use_past_targets)
	  gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], visible_layers[v]->target, 1, hidden_layers[h]->target);
	else
	  gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], visible_layers[v]->output, 1, hidden_layers[h]->target);
      hidden_layers[h]->activation(hidden_layers[h]->target);
      gsl_vector_free(bias);
    }
}


void LearningModel::forward_propagate_directed(Step *step, bool train)
{
  for(unsigned i=0;i<step->hidden_layers.size();i++)
    gsl_vector_set_zero(step->hidden_layers[i]->delta);
  for(unsigned i=0;i<step->visible_layers.size();i++)
    gsl_vector_set_zero(step->visible_layers[i]->delta);

  set_directed_output(step->visible_layers, step->hidden_layers, true);

  for(unsigned i=0;i<step->hidden_layers.size();i++)
    gsl_vector_memcpy(step->hidden_layers[i]->target, step->hidden_layers[i]->output);
}

void LearningModel::forward_propagate_RBM(Step *step, bool train)
{
  for(unsigned i=0;i<step->hidden_layers.size();i++)
    gsl_vector_set_zero(step->hidden_layers[i]->delta);
  for(unsigned i=0;i<step->visible_layers.size();i++)
    gsl_vector_set_zero(step->visible_layers[i]->delta);

  if(useCD)
    set_RBM_output_iterative(step->visible_layers, step->hidden_layers, true);
  else
    set_RBM_output_marginalized(step->visible_layers, step->hidden_layers, true);
  set_RBM_target(step->visible_layers, step->hidden_layers, true);
}

void LearningModel::learn_from_layer(Layer* layer)
{
  unsigned int k,i;

  gsl_vector* delta = gsl_vector_alloc(layer->n_nodes);
  gsl_vector_memcpy(delta, layer->target);
  gsl_vector_sub(delta, layer->output);

  gsl_vector_add(dB[layer->type], delta);
  n_dB[layer->type] += 1;

  for(k=0;k<layer->features.size();k++)
    {
      gsl_vector_add(dW_feature[k][layer->features[k]][layer->type], delta);
      n_dW_feature[k][layer->features[k]][layer->type] += 1;
    }

  for(k=0;k<layer->past_connections.size();k++)
    {
      for(i=0;i<layer->past_connections[k].size(); i++)
	{
	  Layer* past_layer = layer->past_connections[k][i];
	  gsl_blas_dger(1, past_layer->target, delta, dW_past[k][past_layer->type][layer->type]);   //try using past outputs
	  n_dW_past[k][past_layer->type][layer->type] += 1;
	}
    }
  gsl_vector_free(delta);
}


void LearningModel::learn_directed(vector<Layer*> visible_layers, vector<Layer*> hidden_layers)
{
  unsigned int v,h,k,i;

  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_memcpy(visible_layers[v]->delta, visible_layers[v]->target);
      gsl_vector_sub(visible_layers[v]->delta, visible_layers[v]->output);
      
      gsl_vector_add(dB[visible_layers[v]->type], visible_layers[v]->delta);
      n_dB[visible_layers[v]->type] += 1;
      
      for(k=0;k<visible_layers[v]->features.size();k++)
	{
	  gsl_vector_add(dW_feature[k][visible_layers[v]->features[k]][visible_layers[v]->type], visible_layers[v]->delta);
	  n_dW_feature[k][visible_layers[v]->features[k]][visible_layers[v]->type] += 1;
	}
      
      for(k=0;k<visible_layers[v]->past_connections.size();k++)
	{
	  for(i=0;i<visible_layers[v]->past_connections[k].size(); i++)
	    {
	      Layer* past_layer = visible_layers[v]->past_connections[k][i];
	      gsl_blas_dger(1, past_layer->target, visible_layers[v]->delta, dW_past[k][past_layer->type][visible_layers[v]->type]);
	      n_dW_past[k][past_layer->type][visible_layers[v]->type] += 1;
	    }
	}
      
      for(h=0;h<hidden_layers.size();h++)
	{
	  gsl_blas_dger(1, hidden_layers[h]->target, visible_layers[v]->delta, dW[hidden_layers[h]->type][visible_layers[v]->type]);
	  n_dW[hidden_layers[h]->type][visible_layers[v]->type] += 1;
	  gsl_blas_dgemv(CblasNoTrans, 1, W[hidden_layers[h]->type][visible_layers[v]->type], visible_layers[v]->delta, 1, hidden_layers[h]->delta);
	}
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* output_derivative = gsl_vector_alloc(hidden_layers[h]->n_nodes);
      gsl_vector_memcpy(output_derivative, hidden_layers[h]->output);
      derivative_sigmoid(output_derivative);
      gsl_vector_mul(hidden_layers[h]->delta, output_derivative);
      gsl_vector_free(output_derivative);

      gsl_vector_add(dB[hidden_layers[h]->type], hidden_layers[h]->delta);
      n_dB[hidden_layers[h]->type] += 1;
      
      for(k=0;k<hidden_layers[h]->features.size();k++)
	{
	  gsl_vector_add(dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type], hidden_layers[h]->delta);
	  n_dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type] += 1;
	}
      
      for(k=0;k<hidden_layers[h]->past_connections.size();k++)
	{
	  for(i=0;i<hidden_layers[h]->past_connections[k].size(); i++)
	    {
	      Layer* past_layer = hidden_layers[h]->past_connections[k][i];
	      gsl_blas_dger(1, past_layer->target, hidden_layers[h]->delta, dW_past[k][past_layer->type][hidden_layers[h]->type]);
	      n_dW_past[k][past_layer->type][hidden_layers[h]->type] += 1;
	      gsl_blas_dgemv(CblasNoTrans, 1, W_past[k][past_layer->type][hidden_layers[h]->type], hidden_layers[h]->delta, 1, past_layer->delta);
	    }
	}
    }
}


void LearningModel::TRBM_contrastive_divergence(vector<Layer*> visible_layers, vector<Layer*> hidden_layers, int N=1)
{
  vector<gsl_vector*> visible_layers_bias(visible_layers.size());
  vector<gsl_vector*> visible_layers_recon(visible_layers.size());

  vector<gsl_vector*> hidden_layers_bias(hidden_layers.size());
  vector<gsl_vector*> hidden_layers_data(hidden_layers.size());
  vector<gsl_vector*> hidden_layers_recon(hidden_layers.size());

  unsigned int v,h,k,i;

  for(v=0;v<visible_layers.size();v++)
    {
      visible_layers_bias[v] = get_network_bias(visible_layers[v], true);
      visible_layers_recon[v] = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector_memcpy(visible_layers_recon[v], visible_layers[v]->target);
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      hidden_layers_bias[h] = get_network_bias(hidden_layers[h], true);
      hidden_layers_recon[h] = gsl_vector_alloc(hidden_layers[h]->n_nodes);
      hidden_layers_data[h] = gsl_vector_alloc(hidden_layers[h]->n_nodes);
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector_memcpy(hidden_layers_recon[h], hidden_layers_bias[h]);
      for(v=0;v<visible_layers.size();v++)
	gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], visible_layers_recon[v], 1, hidden_layers_recon[h]);
      hidden_layers[h]->activation(hidden_layers_recon[h]);
      gsl_vector_memcpy(hidden_layers_data[h], hidden_layers_recon[h]);
    }

  for(int iter=0;iter<N;iter++)
    {
      for(h=0;h<hidden_layers.size();h++)
        {
	  sample(hidden_layers_recon[h], rng);
        }
      for(v=0;v<visible_layers.size();v++)
	{
	  gsl_vector_memcpy(visible_layers_recon[v], visible_layers_bias[v]);
	  for(h=0;h<hidden_layers.size();h++)
	    gsl_blas_dgemv(CblasTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], hidden_layers_recon[h], 1, visible_layers_recon[v]);
	  visible_layers[v]->activation(visible_layers_recon[v]);
	}
      for(h=0;h<hidden_layers.size();h++)
        {
          gsl_vector_memcpy(hidden_layers_recon[h], hidden_layers_bias[h]);
          for(v=0;v<visible_layers.size();v++)
            gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], visible_layers_recon[v], 1, hidden_layers_recon[h]);
          hidden_layers[h]->activation(hidden_layers_recon[h]);
        }
    }


  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector* delta = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector_memcpy(delta, visible_layers[v]->target);
      gsl_vector_sub(delta, visible_layers_recon[v]);
      gsl_vector_memcpy(visible_layers[v]->delta, delta);
      //gsl_vector_scale(delta, 0.1);

      gsl_vector_add(dB[visible_layers[v]->type], delta);
      n_dB[visible_layers[v]->type] += 1;
      
      for(k=0;k<visible_layers[v]->features.size();k++)
	{
	  gsl_vector_add(dW_feature[k][visible_layers[v]->features[k]][visible_layers[v]->type], delta);
	  n_dW_feature[k][visible_layers[v]->features[k]][visible_layers[v]->type] += 1;
	}
      
      for(k=0;k<visible_layers[v]->past_connections.size();k++)
	{
	  for(i=0;i<visible_layers[v]->past_connections[k].size(); i++)
	    {
	      Layer* past_layer = visible_layers[v]->past_connections[k][i];
	      gsl_blas_dger(1, past_layer->target, delta, dW_past[k][past_layer->type][visible_layers[v]->type]);
	      n_dW_past[k][past_layer->type][visible_layers[v]->type] += 1;
	    }
	}

      for(h=0;h<hidden_layers.size();h++)
	{
	  gsl_blas_dgemv(CblasNoTrans, 1, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], delta, 1, hidden_layers[h]->delta);
	}

      gsl_vector_free(delta);
    }


  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* output_derivative = gsl_vector_alloc(hidden_layers[h]->n_nodes);
      gsl_vector_memcpy(output_derivative, hidden_layers_data[h]);
      derivative_sigmoid(output_derivative);
      gsl_vector_mul(hidden_layers[h]->delta, output_derivative);
      gsl_vector_free(output_derivative);

      for(v=0;v<visible_layers.size();v++)
	{
	  gsl_blas_dger(1, hidden_layers_data[h], visible_layers[v]->target, dW_RBM[hidden_layers[h]->type][visible_layers[v]->type]);  //could use hidden samples instead
	  gsl_blas_dger(-1, hidden_layers_recon[h], visible_layers_recon[v], dW_RBM[hidden_layers[h]->type][visible_layers[v]->type]);
	  gsl_blas_dger(1, hidden_layers_recon[h], visible_layers[v]->delta, dW_RBM[hidden_layers[h]->type][visible_layers[v]->type]);
	  //gsl_blas_dger(1, hidden_layers[h]->delta, visible_layers_recon[v], dW_RBM[hidden_layers[h]->type][visible_layers[v]->type]);
	  n_dW_RBM[hidden_layers[h]->type][visible_layers[v]->type] += 2; //2;
	  //cout<<"RBM weight change: "<<hidden_layers[h]->type<<","<<visible_layers[v]->type<<" : "<<matrix_norm(dW_RBM[hidden_layers[h]->type][visible_layers[v]->type])<<"\n";
	}
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* delta = gsl_vector_alloc(hidden_layers[h]->n_nodes);
      gsl_vector_memcpy(delta, hidden_layers_data[h]);   //could use hidden samples instead
      gsl_vector_sub(delta, hidden_layers_recon[h]);

      gsl_vector_add(dB[hidden_layers[h]->type], delta);
      gsl_vector_add(dB[hidden_layers[h]->type], hidden_layers[h]->delta);
      n_dB[hidden_layers[h]->type] += 2; //2;
      
      for(k=0;k<hidden_layers[h]->features.size();k++)
	{
	  gsl_vector_add(dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type], delta);
	  gsl_vector_add(dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type], hidden_layers[h]->delta);
	  n_dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type] += 2; //2;
	}
      
      for(k=0;k<hidden_layers[h]->past_connections.size();k++)
	{
	  for(i=0;i<hidden_layers[h]->past_connections[k].size(); i++)
	    {
	      Layer* past_layer = hidden_layers[h]->past_connections[k][i];
	      gsl_blas_dger(1, past_layer->target, delta, dW_past[k][past_layer->type][hidden_layers[h]->type]);
	      gsl_blas_dger(1, past_layer->target, hidden_layers[h]->delta, dW_past[k][past_layer->type][hidden_layers[h]->type]);
	      n_dW_past[k][past_layer->type][hidden_layers[h]->type] += 2; //2;
	      gsl_blas_dgemv(CblasNoTrans, 1, W_past[k][past_layer->type][hidden_layers[h]->type], hidden_layers[h]->delta, 1, past_layer->delta);
	    }
	}

      gsl_vector_free(delta);
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector_free(hidden_layers_bias[h]);
      gsl_vector_free(hidden_layers_data[h]);
      gsl_vector_free(hidden_layers_recon[h]);
    }

  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_free(visible_layers_bias[v]);
      gsl_vector_free(visible_layers_recon[v]);
    }
}

void LearningModel::TRBM_direct_gradient(vector<Layer*> visible_layers, vector<Layer*> hidden_layers)
{
  //vector<gsl_vector*> visible_layers_bias(visible_layers.size());
  vector<gsl_vector*> hidden_layers_bias(hidden_layers.size());
  unsigned int v,h, k, k1;
  int i;
  /*
  for(v=0;v<visible_layers.size();v++)
    {
      visible_layers_bias[v] = get_network_bias(visible_layers[v], true);
    }
  */
  for(h=0;h<hidden_layers.size();h++)
    {
      hidden_layers_bias[h] = get_network_bias(hidden_layers[h], true);
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector* output_derivative = gsl_vector_alloc(hidden_layers[h]->n_nodes);
      gsl_vector_memcpy(output_derivative, hidden_layers[h]->target);
      derivative_sigmoid(output_derivative);
      gsl_vector_mul(hidden_layers[h]->delta, output_derivative);
      gsl_vector_free(output_derivative);

      for(v=0;v<visible_layers.size();v++)
	{
	  gsl_blas_dger(1, hidden_layers[h]->delta, visible_layers[v]->target, dW_RBM[hidden_layers[h]->type][visible_layers[v]->type]);
	  n_dW_RBM[hidden_layers[h]->type][visible_layers[v]->type] += 1;
	}
    }

  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector* t_V = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector_memcpy(t_V, visible_layers[v]->output);
      gsl_vector_scale(t_V, -1);
      gsl_vector_add(t_V, visible_layers[v]->target);
      
      gsl_vector* dE_dBv = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector_memcpy(dE_dBv, t_V);
      gsl_vector_add(dB[visible_layers[v]->type], dE_dBv);
      n_dB[visible_layers[v]->type] += 1;

      gsl_vector* dE_dW = gsl_vector_alloc(visible_layers[v]->n_nodes);
      gsl_vector* S = gsl_vector_alloc(visible_layers[v]->n_nodes);
      for(h=0;h<hidden_layers.size();h++)
	{
	  if(v!=0)
	    {
	      gsl_vector* Y = gsl_vector_alloc(hidden_layers[h]->n_nodes);
	      gsl_matrix_get_col(Y, W_RBM[hidden_layers[h]->type][visible_layers[v-1]->type], visible_layers[v-1]->target_id);
	      gsl_vector_add(hidden_layers_bias[h], Y);
	      gsl_vector_free(Y);
	    }

	  gsl_vector* dE_dBias = gsl_vector_alloc(hidden_layers[h]->n_nodes);
	  for(i=0;i<hidden_layers[h]->n_nodes;i++)
	    {
	      gsl_matrix_get_row(S, W_RBM[hidden_layers[h]->type][visible_layers[v]->type], i);
	      gsl_vector_add_constant(S, gsl_vector_get(hidden_layers_bias[h],i));
	      sigmoid(S);

	      gsl_vector* t_V__S = gsl_vector_alloc(visible_layers[v]->n_nodes);
	      gsl_vector_memcpy(t_V__S, t_V);
	      gsl_vector_mul(t_V__S, S);
	      double sum_t_V__S = 0;
	      for(int t=0;t<visible_layers[v]->n_nodes;t++)
		sum_t_V__S += gsl_vector_get(t_V__S, t);

	      gsl_vector_memcpy(dE_dW, t_V__S);
	      gsl_vector_view dW = gsl_matrix_row(dW_RBM[hidden_layers[h]->type][visible_layers[v]->type], i);
	      gsl_vector_add(&dW.vector, dE_dW);
	      gsl_vector_free(t_V__S);

	      gsl_vector_set(dE_dBias, i, sum_t_V__S);
	    }

	  n_dW_RBM[hidden_layers[h]->type][visible_layers[v]->type] += 1;

	  gsl_vector_add(hidden_layers[h]->delta, dE_dBias);

	  gsl_vector_free(dE_dBias);
	}

      gsl_vector_free(t_V);
      gsl_vector_free(dE_dBv);
      gsl_vector_free(S);
      gsl_vector_free(dE_dW);
    }

  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector_add(dB[hidden_layers[h]->type], hidden_layers[h]->delta);
      n_dB[hidden_layers[h]->type] += 1;

      for(k=0;k<hidden_layers[h]->features.size();k++)
	{
	  gsl_vector_add(dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type], hidden_layers[h]->delta);
	  n_dW_feature[k][hidden_layers[h]->features[k]][hidden_layers[h]->type] += 1;
	}
      
      for(k=0;k<hidden_layers[h]->past_connections.size();k++)
	{
	  for(k1=0;k1<hidden_layers[h]->past_connections[k].size(); k1++)
	    {
	      Layer* past_layer = hidden_layers[h]->past_connections[k][k1];
	      gsl_blas_dger(1, past_layer->target, hidden_layers[h]->delta, dW_past[k][past_layer->type][hidden_layers[h]->type]);
	      n_dW_past[k][past_layer->type][hidden_layers[h]->type] += 1;
	      gsl_blas_dgemv(CblasNoTrans, 1, W_past[k][past_layer->type][hidden_layers[h]->type], hidden_layers[h]->delta, 1, past_layer->delta);
	    }
	}
    }
  
  for(h=0;h<hidden_layers.size();h++)
    {
      gsl_vector_free(hidden_layers_bias[h]);
    }

  /*
  for(v=0;v<visible_layers.size();v++)
    {
      gsl_vector_free(visible_layers_bias[v]);
    }
  */
}



void LearningModel::learn_from_step_directed(Step *step)
{
  learn_directed(step->visible_layers, step->hidden_layers);
}


void LearningModel::learn_from_step_RBM(Step *step)
{
  if(useCD)
    TRBM_contrastive_divergence(step->visible_layers, step->hidden_layers, 1);
  else
    TRBM_direct_gradient(step->visible_layers, step->hidden_layers);
}

void LearningModel::learn_bias(Step *step)
{
  unsigned int i;
  double x, y, sum;

  for(unsigned d=0;d<step->visible_layers.size();d++)
    {
      int decision = step->visible_layers[d]->type;
      gsl_vector_add(__initB[decision], step->visible_layers[d]->target);
      sum = 0;
      for(i=0;i<__initB[decision]->size;i++)
	sum += gsl_vector_get(__initB[decision],i);
      for(i=0;i<__initB[decision]->size;i++)
	{
	  x = gsl_vector_get(__initB[decision],i);
	  y = log((1.0+x)/(__initB[decision]->size+sum-x));
	  gsl_vector_set(B[decision], i, y);
	}      
    }
}

string LearningModel::write_vector(gsl_vector* V, string prefix)
{
  ostringstream result;
  for(unsigned int i=0;i<V->size;i++)
    result<<prefix<<"<value i=\""<<i<<"\">"<<gsl_vector_get(V,i)<<"</value>\n";
  return result.str();
}

string LearningModel::write_matrix(gsl_matrix* M, string prefix)
{
  ostringstream result;
  for(unsigned i=0;i<M->size1;i++)
    for(unsigned j=0;j<M->size2;j++)
      result<<prefix<<"<value i=\""<<i<<"\" j=\""<<j<<"\">"<<gsl_matrix_get(M,i,j)<<"</value>\n";
  return result.str();
}

void LearningModel::save(string filename)
{
  string bkp_file = filename + ".bkp";
  rename(filename.c_str(), bkp_file.c_str());
  ofstream fd;
  fd.open(filename.c_str());
  fd<<"<LearningModel>\n";
  fd<<"\t<ModelType>"<<(isTRBM?"TRBM":"FF")<<"</ModelType>\n";
  fd<<"\t<TRBM_Training>"<<(useCD?"Constrastive Divergence":"Direct Gradient")<<"</TRBM_Training>\n";
  fd<<"\t<NumberOfFeatures>"<<n_features<<"</NumberOfFeatures>\n"
    <<"\t<NumberOfConnections>"<<n_connections<<"</NumberOfConnections>\n";

  for(unordered_map<int,int>::iterator it_ls=layer_size.begin(); it_ls!=layer_size.end(); it_ls++)
    fd<<"\t<LayerSize layer=\""<<it_ls->first<<"\">"<<it_ls->second<<"</LayerSize>\n";

  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      fd<<"\t<LayerBias layer=\""<<it_V->first<<"\" rows=\""<<it_V->second->size<<"\">\n";
      fd<<write_vector(it_V->second, "\t\t");
      fd<<"\t</LayerBias>\n";
    }

  for (int k=0;k<n_features;k++)
    {
      fd<<"\t<FeatureConnection type=\""<<k<<"\">\n";
      for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
	{
	  fd<<"\t\t<Feature>\n";
	  fd<<"\t\t\t<value><![CDATA["<<it1->first<<"]]></value>\n";
	  for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	    {
	      fd<<"\t\t\t<connection layer=\""<<it_V->first<<"\" rows=\""<<it_V->second->size<<"\">\n";
	      fd<<write_vector(it_V->second, "\t\t\t\t");
	      fd<<"\t\t\t</connection>\n";
	    }
	  fd<<"\t\t</Feature>\n";
	}
      fd<<"\t</FeatureConnection>\n";
    }
    
  for(it=W.begin(); it!=W.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	fd<<"\t<Connection layer1=\""<<it->first<<"\" layer2=\""<<it_M->first<<"\" rows=\""<<it_M->second->size1<<"\" cols=\""<<it_M->second->size2<<"\">\n";
	fd<<write_matrix(it_M->second, "\t\t");
	fd<<"\t</Connection>\n";
      }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	fd<<"\t<RBM layer1=\""<<it->first<<"\" layer2=\""<<it_M->first<<"\" rows=\""<<it_M->second->size1<<"\" cols=\""<<it_M->second->size2<<"\">\n";
	fd<<write_matrix(it_M->second, "\t\t");
	fd<<"\t</RBM>\n";
      }

  for(int k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  fd<<"\t<PastConnection type=\""<<k<<"\" layer1=\""<<it->first<<"\" layer2=\""<<it_M->first<<"\" rows=\""<<it_M->second->size1<<"\" cols=\""<<it_M->second->size2<<"\">\n";
	  fd<<write_matrix(it_M->second, "\t\t");
	  fd<<"\t</PastConnection>\n";
	}

  fd<<"</LearningModel>\n";
  fd.close();
  remove(bkp_file.c_str());
}


void LearningModel::save_weights_binary(string filename_info, string filename)
{
  FILE *fp_info;
  fp_info=fopen(filename_info.c_str(), "w");

  FILE *fp;
  fp=fopen(filename.c_str(), "w");

  fprintf(fp_info, "<LearningModel>\n");

  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      fprintf(fp_info, "\t<LayerBias layer=\"%d\" rows=\"%d\">\n", it_V->first, (int) it_V->second->size);
      gsl_vector_fwrite (fp, it_V->second);
      fprintf(fp_info, "\t</LayerBias>\n");
    }

  for (int k=0;k<n_features;k++)
    {
      fprintf(fp_info, "\t<FeatureConnection type=\"%d\">\n", k);
      for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
	{
	  fprintf(fp_info, "\t\t<Feature>\n");
	  fprintf(fp_info, "\t\t\t<value><![CDATA[%s]]></value>\n", it1->first.c_str());
	  for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	    {
	      fprintf(fp_info, "\t\t\t<connection layer=\"%d\" rows=\"%d\">\n", it_V->first, (int) it_V->second->size);
	      gsl_vector_fwrite(fp, it_V->second);
	      fprintf(fp_info, "\t\t\t</connection>\n");
	    }
	  fprintf(fp_info, "\t\t</Feature>\n");
	}
      fprintf(fp_info, "\t</FeatureConnection>\n");
    }
    
  for(it=W.begin(); it!=W.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	fprintf(fp_info, "\t<Connection layer1=\"%d\" layer2=\"%d\" rows=\"%d\" cols=\"%d\">\n", it->first, it_M->first, (int) it_M->second->size1, (int) it_M->second->size2);
	gsl_matrix_fwrite(fp, it_M->second);
	fprintf(fp_info, "\t</Connection>\n");
      }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	fprintf(fp_info, "\t<RBM layer1=\"%d\" layer2=\"%d\" rows=\"%d\" cols=\"%d\">\n", it->first, it_M->first, (int) it_M->second->size1, (int) it_M->second->size2);
	gsl_matrix_fwrite(fp, it_M->second);
	fprintf(fp_info, "\t</RBM>\n");
      }

  for(int k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  fprintf(fp_info, "\t<PastConnection type=\"%d\" layer1=\"%d\" layer2=\"%d\" rows=\"%d\" cols=\"%d\">\n", k, it->first, it_M->first, (int) it_M->second->size1, (int) it_M->second->size2);
	  gsl_matrix_fwrite(fp, it_M->second);
	  fprintf(fp_info, "\t</PastConnection>\n");
	}

  fprintf(fp_info, "</LearningModel>\n");
  fclose(fp);
  fclose(fp_info);
}


void LearningModel::read_from_file(string filename)
{
  cerr<<"reading Learning Model from file: "<<filename<<"...";
  ifstream fd;
  fd.open(filename.c_str());
  try
    {
      xmlpp::TextReader reader(filename);
      while(reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="ModelType")
		{
		  reader.move_to_element();
		  reader.read();
		  isTRBM = true;
		  if(reader.get_value()=="FF")
		    isTRBM = false;
		}
	      else if(reader.get_name()=="TRBM_Training")
		{
		  reader.move_to_element();
		  reader.read();
		  useCD = false;
		  if(reader.get_value()=="Contrastive Divergence")
		    useCD = true;
		}
	      else if(reader.get_name()=="NumberOfFeatures")
		{
		  reader.move_to_element();
		  reader.read();
		  n_features = atoi(reader.get_value().c_str());
		  W_feature = new unordered_map< string, unordered_map<int,gsl_vector*> >[n_features];
		}
	      else if(reader.get_name()=="NumberOfConnections")
		{
		  reader.move_to_element();
		  reader.read();
		  n_connections = atoi(reader.get_value().c_str());
		  W_past = new unordered_map< int, unordered_map<int,gsl_matrix*> >[n_connections];
		}
	      else if(reader.get_name()=="LayerSize")
		{
		  reader.move_to_attribute("layer");
		  int l = atoi(reader.get_value().c_str());
		  reader.move_to_element();
		  reader.read();
		  layer_size[l] = atoi(reader.get_value().c_str());
		  if(l==dummy_layer_type)
		    {
		      vector<string> _features;
		      vector< vector<Layer*> > _past_connections(n_connections);
		      dummy_layer = new Layer("dummy", dummy_layer_type, layer_size[dummy_layer_type], NULL, -1, _features, _past_connections);
		      gsl_vector_set_all(dummy_layer->output, 1);
		      gsl_vector_set_all(dummy_layer->target, 1);
		    }
		}
	      else if(reader.get_name()=="LayerBias")
		{
		  reader.move_to_attribute("layer");
		  int l = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  int n = atoi(reader.get_value().c_str());
		  B[l] = gsl_vector_alloc(n);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="LayerBias")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="value")
			{
			  reader.move_to_attribute("i");
			  int i = atoi(reader.get_value().c_str());
			  reader.move_to_element();
			  reader.read();
			  double v = atof(reader.get_value().c_str());
			  gsl_vector_set(B[l], i, v);
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="FeatureConnection")
		{
		  reader.move_to_attribute("type");
		  int type = atoi(reader.get_value().c_str());
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="FeatureConnection")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="Feature")
			{
			  reader.move_to_element();
			  reader.read();
			  string fv = "";
			  while(reader.get_name()!="Feature")
			    {
			      if(reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
				{
				  if(reader.get_name()=="value")
				    {
				      reader.move_to_element();
				      reader.read();
				      fv = reader.get_value();
				    }
				  else if(reader.get_name()=="connection")
				    {
				      reader.move_to_attribute("layer");
				      int l = atoi(reader.get_value().c_str());
				      reader.move_to_attribute("rows");
				      int n = atoi(reader.get_value().c_str());
				      W_feature[type][fv][l] = gsl_vector_alloc(n);
				      reader.move_to_element();
				      reader.read();
				      while(reader.get_name()!="connection")
					{
					  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="value")
					    {
					      reader.move_to_attribute("i");
					      int i = atoi(reader.get_value().c_str());
					      reader.move_to_element();
					      reader.read();
					      double v = atof(reader.get_value().c_str());
					      gsl_vector_set(W_feature[type][fv][l], i, v);
					    }
					  reader.read();
					}
				    }
				}
			      reader.read();
			    }
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="Connection")
		{
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  int cols = atoi(reader.get_value().c_str());
		  W[l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="Connection")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="value")
			{
			  reader.move_to_attribute("i");
			  int i = atoi(reader.get_value().c_str());
			  reader.move_to_attribute("j");
			  int j = atoi(reader.get_value().c_str());
			  reader.move_to_element();
			  reader.read();
			  double v = atof(reader.get_value().c_str());
			  gsl_matrix_set(W[l1][l2], i, j, v);
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="RBM")
		{
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  int cols = atoi(reader.get_value().c_str());
		  W_RBM[l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="RBM")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="value")
			{
			  reader.move_to_attribute("i");
			  int i = atoi(reader.get_value().c_str());
			  reader.move_to_attribute("j");
			  int j = atoi(reader.get_value().c_str());
			  reader.move_to_element();
			  reader.read();
			  double v = atof(reader.get_value().c_str());
			  gsl_matrix_set(W_RBM[l1][l2], i, j, v);
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="PastConnection")
		{
		  reader.move_to_attribute("type");
		  int type = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  int cols = atoi(reader.get_value().c_str());
		  W_past[type][l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="PastConnection")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="value")
			{
			  reader.move_to_attribute("i");
			  int i = atoi(reader.get_value().c_str());
			  reader.move_to_attribute("j");
			  int j = atoi(reader.get_value().c_str());
			  reader.move_to_element();
			  reader.read();
			  double v = atof(reader.get_value().c_str());
			  gsl_matrix_set(W_past[type][l1][l2], i, j, v);
			}
		      reader.read();
		    }
		}
	    }
	}
    }
  catch(const std::exception& e)
    {
      cerr << "Exception caught: " << e.what() << endl;
    }  

  fd.close();

  create_dW();
  cerr<<"done!\n";
}


void LearningModel::read_weights_binary(string filename_info, string filename)
{
  FILE *fp;
  fp=fopen(filename.c_str(), "r");

  try
    {
      xmlpp::TextReader reader(filename_info);
      while(reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="LayerBias")
		{
		  reader.move_to_attribute("layer");
		  int l = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  //int n = atoi(reader.get_value().c_str());
		  //B[l] = gsl_vector_alloc(n);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="LayerBias")
		    {
		      reader.read();
		    }
		  gsl_vector_fread (fp, B[l]);
		}
	      else if(reader.get_name()=="FeatureConnection")
		{
		  reader.move_to_attribute("type");
		  int type = atoi(reader.get_value().c_str());
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="FeatureConnection")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="Feature")
			{
			  reader.move_to_element();
			  reader.read();
			  string fv = "";
			  while(reader.get_name()!="Feature")
			    {
			      if(reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
				{
				  if(reader.get_name()=="value")
				    {
				      reader.move_to_element();
				      reader.read();
				      fv = reader.get_value();
				    }
				  else if(reader.get_name()=="connection")
				    {
				      reader.move_to_attribute("layer");
				      int l = atoi(reader.get_value().c_str());
				      reader.move_to_attribute("rows");
				      //int n = atoi(reader.get_value().c_str());
				      //W_feature[type][fv][l] = gsl_vector_alloc(n);
				      reader.move_to_element();
				      reader.read();
				      while(reader.get_name()!="connection")
					{
					  reader.read();
					}
				      gsl_vector_fread (fp, W_feature[type][fv][l]);
				    }
				}
			      reader.read();
			    }
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="Connection")
		{
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  //int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  //int cols = atoi(reader.get_value().c_str());
		  //W[l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="Connection")
		    {
		      reader.read();
		    }
		  gsl_matrix_fread (fp, W[l1][l2]);
		}
	      else if(reader.get_name()=="RBM")
		{
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  //int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  //int cols = atoi(reader.get_value().c_str());
		  //W_RBM[l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="RBM")
		    {
		      reader.read();
		    }
		  gsl_matrix_fread (fp, W_RBM[l1][l2]);
		}
	      else if(reader.get_name()=="PastConnection")
		{
		  reader.move_to_attribute("type");
		  int type = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer1");
		  int l1 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("layer2");
		  int l2 = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("rows");
		  //int rows = atoi(reader.get_value().c_str());
		  reader.move_to_attribute("cols");
		  //int cols = atoi(reader.get_value().c_str());
		  //W_past[type][l1][l2] = gsl_matrix_alloc(rows, cols);
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="PastConnection")
		    {
		      reader.read();
		    }
		  gsl_matrix_fread (fp, W_past[type][l1][l2]);
		}
	    }
	}
    }
  catch(const std::exception& e)
    {
      cerr << "Exception caught: " << e.what() << endl;
    }  

  fclose(fp);
}


void LearningModel::update(LearningInstance* instance, State* state, int decision, int target, bool new_step)
{
  int check = update_network(instance, state, decision, target, new_step);
  if (check!=0)
    {
      cerr<<"Error in updating network...exiting\n";
      exit(0);
    }
}

void LearningModel::process_last_step(LearningInstance* instance, bool training)
{
  Step *step = instance->steps.back();

  for(unsigned h=0;h<step->hidden_layers.size();h++)
    for(int k=0;k<n_features;k++)
      {
	string fv = step->hidden_layers[h]->features[k];
	if (W_feature[k].find(fv)==W_feature[k].end() || W_feature[k][fv].find(hidden_layer_type)==W_feature[k][fv].end())
	  {
	    if (training)
	      {
		W_feature[k][fv][hidden_layer_type] = gsl_vector_alloc(layer_size[hidden_layer_type]);
		for(int i=0;i<layer_size[hidden_layer_type];i++)
		  gsl_vector_set(W_feature[k][fv][hidden_layer_type], i, init_range*(2*gsl_rng_uniform(rng)-1));
		dW_feature[k][fv][hidden_layer_type] = gsl_vector_calloc(layer_size[hidden_layer_type]);
		n_dW_feature[k][fv][hidden_layer_type] = 0;
	      }
	    else
	      {
		step->hidden_layers[h]->features[k] = "#IGNORE#";
	      }
	  }
      }
  
  if(isTRBM)
    forward_propagate_RBM(step, true);
  else
    forward_propagate_directed(step, true);
}



void LearningModel::learn(LearningInstance* instance, int iter)
{
  if(iter==1 && !bias_initialized)
    {
      for (unsigned i=0;i<instance->steps.size();i++)
	learn_bias(instance->steps[i]);
    }
  else
    {
      bias_initialized = true;
      reset_dW();
      if(isTRBM)
	for (int i=instance->steps.size()-1;i>=0;i--)
	  learn_from_step_RBM(instance->steps[i]);
      else
	for (int i=instance->steps.size()-1;i>=0;i--)
	  learn_from_step_directed(instance->steps[i]);
      update_W();
    }
}

int LearningModel::set_target_of_last_decision(LearningInstance* instance, int decision, int target)
{
  Step *step = instance->steps.back();
  for(unsigned i=0;i<step->visible_layers.size();i++)
    {
      Layer *layer = step->visible_layers[i];
      if(layer->type==decision)
	{
	  layer->set_target(target);
	  if(isTRBM)
	    set_RBM_target(step->visible_layers, step->hidden_layers, true);
	  return 1;
	}
    }

  return 0;
}

void LearningModel::initialize_W()
{
  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_matrix* M;
  unsigned int i,j;
  int k;

  for(it_V=__initB.begin(); it_V!=__initB.end(); it_V++)
    {
      V = it_V->second;
      gsl_vector_set_all(V, 1);
    }

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      V = it_V->second;
      gsl_vector_set_zero(V);
    }

  for (k=0;k<n_features;k++)
    for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
      for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	{
	  V = it_V->second;
	  for (i=0;i<V->size;i++)
	    gsl_vector_set(V, i, init_range*(2*gsl_rng_uniform(rng)-1));
	}
  
  for(it=W.begin(); it!=W.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	M = it_M->second;
	for (i=0;i<M->size1;i++)
	  for (j=0;j<M->size2;j++)
	    gsl_matrix_set(M, i, j, init_range_emiss*(2*gsl_rng_uniform(rng)-1));
      }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
      {
	M = it_M->second;
	for (i=0;i<M->size1;i++)
	  for (j=0;j<M->size2;j++)
	    gsl_matrix_set(M, i, j, init_range*(2*gsl_rng_uniform(rng)-1));
      }

  for(k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  M = it_M->second;
	  for (i=0;i<M->size1;i++)
	    for (j=0;j<M->size2;j++)
	      gsl_matrix_set(M, i, j, init_range*(2*gsl_rng_uniform(rng)-1));
	}
}

void LearningModel::create_dW()
{
  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_matrix* M;
  int l1,l2;
  string l3;
  int k;

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      l1 = it_V->first;
      V = it_V->second;
      dB[l1] = gsl_vector_calloc(V->size);
      n_dB[l1] = 0;
    }

  dW_feature = new unordered_map< string, unordered_map<int,gsl_vector*> >[n_features];
  n_dW_feature = new unordered_map< string, unordered_map<int,int> >[n_features];
  for (k=0;k<n_features;k++)
    for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
      {
	l3 = it1->first;
	for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	  {
	    l2 = it_V->first;
	    V = it_V->second;
	    dW_feature[k][l3][l2] = gsl_vector_calloc(V->size);
	    n_dW_feature[k][l3][l2] = 0;
	  }
      }
  
  for(it=W.begin(); it!=W.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  dW[l1][l2] = gsl_matrix_calloc(M->size1, M->size2);
	  n_dW[l1][l2] = 0;
	}
    }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  dW_RBM[l1][l2] = gsl_matrix_calloc(M->size1, M->size2);
	  n_dW_RBM[l1][l2] = 0;
	}
    }

  dW_past = new unordered_map< int, unordered_map<int,gsl_matrix*> >[n_connections];
  n_dW_past = new unordered_map< int, unordered_map<int,int> >[n_connections];
  for(k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      {
	l1 = it->first;
	for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	  {
	    l2 = it_M->first;
	    M = it_M->second;
	    dW_past[k][l1][l2] = gsl_matrix_calloc(M->size1, M->size2);
	    n_dW_past[k][l1][l2] = 0;
	  }
      }
}


void LearningModel::clear_momentum()
{
  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_matrix* M;
  int l1,l2;
  string l3;
  int k;

  for(it_V=dB.begin(); it_V!=dB.end(); it_V++)
    {
      l1 = it_V->first;
      V = it_V->second;
      gsl_vector_set_zero(V);
      n_dB[l1] = 0;
    }

  for (k=0;k<n_features;k++)
    for(it1=dW_feature[k].begin(); it1!=dW_feature[k].end(); it1++)
      {
	l3 = it1->first;
	for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	  {
	    l2 = it_V->first;
	    V = it_V->second;
	    gsl_vector_set_zero(V);
	    n_dW_feature[k][l3][l2] = 0;
	  }
      }
  
  for(it=dW.begin(); it!=dW.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  gsl_matrix_set_zero(M);
	  n_dW[l1][l2] = 0;
	}
    }

  for(it=dW_RBM.begin(); it!=dW_RBM.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  gsl_matrix_set_zero(M);
	  n_dW_RBM[l1][l2] = 0;
	}
    }

  for(k=0;k<n_connections;k++)
    for(it=dW_past[k].begin(); it!=dW_past[k].end(); it++)
      {
	l1 = it->first;
	for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	  {
	    l2 = it_M->first;
	    M = it_M->second;
	    gsl_matrix_set_zero(M);
	    n_dW_past[k][l1][l2] = 0;
	  }
      }
}


void LearningModel::reset_dW()
{
  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_matrix* M;
  int l1,l2;
  string l3;
  int k;

  for(it_V=dB.begin(); it_V!=dB.end(); it_V++)
    {
      l1 = it_V->first;
      V = it_V->second;
      if (n_dB[l1])
	{
	  gsl_vector_scale(V, momentum);
	  n_dB[l1] = 0;
	}
    }

  for (k=0;k<n_features;k++)
    for(it1=dW_feature[k].begin(); it1!=dW_feature[k].end(); it1++)
      {
	l3 = it1->first;
	for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	  {
	    l2 = it_V->first;
	    V = it_V->second;
	    if (n_dW_feature[k][l3][l2])
	      {
		gsl_vector_scale(V, momentum);
		n_dW_feature[k][l3][l2] = 0;
	      }
	  }
      }
  
  for(it=dW.begin(); it!=dW.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  if (n_dW[l1][l2])
	    {
	      gsl_matrix_scale(M, momentum);
	      n_dW[l1][l2] = 0;
	    }
	}
    }

  for(it=dW_RBM.begin(); it!=dW_RBM.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  if (n_dW_RBM[l1][l2])
	    {
	      gsl_matrix_scale(M, momentum);
	      n_dW_RBM[l1][l2] = 0;
	    }
	}
    }

  for(k=0;k<n_connections;k++)
    for(it=dW_past[k].begin(); it!=dW_past[k].end(); it++)
      {
	l1 = it->first;
	for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	  {
	    l2 = it_M->first;
	    M = it_M->second;
	    if (n_dW_past[k][l1][l2])
	      {
		gsl_matrix_scale(M, momentum);
		n_dW_past[k][l1][l2] = 0;
	      }
	  }
      }
}


void LearningModel::update_W()
{
  unordered_map<int, gsl_vector*>::iterator it_V;
  unordered_map<int,gsl_matrix*>::iterator it_M;
  unordered_map< int, unordered_map<int,gsl_matrix*> >::iterator it;
  unordered_map< string, unordered_map<int,gsl_vector*> >::iterator it1;
  gsl_vector* V;
  gsl_vector* dV;
  gsl_matrix* M;
  gsl_matrix* dM;
  int l1,l2;
  string l3;
  int k, n_dV, n_dM;

  for(it_V=B.begin(); it_V!=B.end(); it_V++)
    {
      l1 = it_V->first;
      V = it_V->second;
      dV = dB[l1];
      n_dV = n_dB[l1]>0;
      if (n_dV)
	{
	  gsl_vector_scale(dV, learning_rate/n_dV);
	  //gsl_vector_scale(V, (1-learning_rate*weight_decay));
	  gsl_vector_add(V, dV);
	}
    }

  for (k=0;k<n_features;k++)
    for(it1=W_feature[k].begin(); it1!=W_feature[k].end(); it1++)
      {
	l3 = it1->first;
	for(it_V=it1->second.begin(); it_V!=it1->second.end(); it_V++)
	  {
	    l2 = it_V->first;
	    V = it_V->second;
	    dV = dW_feature[k][l3][l2];
	    n_dV = n_dW_feature[k][l3][l2]>0;
	    if (n_dV)
	      {
		gsl_vector_scale(dV, learning_rate/n_dV);
		gsl_vector_scale(V, (1-learning_rate*weight_decay));
		gsl_vector_add(V, dV);
	      }
	  }
      }
  
  for(it=W.begin(); it!=W.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  dM = dW[l1][l2];
	  n_dM = n_dW[l1][l2]>0;
	  if (n_dM)
	    {
	      gsl_matrix_scale(dM, learning_rate/n_dM);
	      gsl_matrix_scale(M, (1-learning_rate*weight_decay));
	      gsl_matrix_add(M, dM);
	    }
	}
    }

  for(it=W_RBM.begin(); it!=W_RBM.end(); it++)
    {
      l1 = it->first;
      for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	{
	  l2 = it_M->first;
	  M = it_M->second;
	  dM = dW_RBM[l1][l2];
	  n_dM = n_dW_RBM[l1][l2]>0;
	  if (n_dM)
	    {
	      gsl_matrix_scale(dM, learning_rate/n_dM);
	      gsl_matrix_scale(M, (1-learning_rate*weight_decay));
	      gsl_matrix_add(M, dM);
	    }
	}
    }

  for(k=0;k<n_connections;k++)
    for(it=W_past[k].begin(); it!=W_past[k].end(); it++)
      {
	l1 = it->first;
	for(it_M=it->second.begin(); it_M!=it->second.end(); it_M++)
	  {
	    l2 = it_M->first;
	    M = it_M->second;
	    dM = dW_past[k][l1][l2];
	    n_dM = n_dW_past[k][l1][l2]>0;
	    if (n_dM)
	      {
		gsl_matrix_scale(dM, learning_rate/n_dM);
		gsl_matrix_scale(M, (1-learning_rate*weight_decay));
		gsl_matrix_add(M, dM);
	      }
	  }
      }
}


void LearningModel::set_learning_rate(float _learning_rate)
{
  learning_rate = _learning_rate;
}

void LearningModel::set_weight_decay(float _weight_decay)
{
  weight_decay = _weight_decay;
}

void LearningModel::set_momentum(float _momentum)
{
  momentum = _momentum;
}

void sigmoid(gsl_vector* X)
{
  for(unsigned int i=0;i<X->size;i++)
    {
      double x = gsl_vector_get(X,i);
      double y;
      if (x>=-500)
	y = 1.0/(1+exp(-x));
      else
	y = 0;
      gsl_vector_set(X, i, y);
    }
}

void derivative_sigmoid(gsl_vector* sigmoidX)
{
  gsl_vector* Y = gsl_vector_alloc(sigmoidX->size);
  gsl_vector_memcpy(Y, sigmoidX);
  gsl_vector_scale(sigmoidX, -1);
  gsl_vector_add_constant(sigmoidX, 1);
  gsl_vector_mul(sigmoidX, Y);
  gsl_vector_free(Y);
}

void soft_max(gsl_vector* X)
{
  double sum = 0;
  for(unsigned int i=0;i<X->size;i++)
    {
      double x = gsl_vector_get(X,i);
      double y = exp(x);
      sum += y;
      gsl_vector_set(X, i, y);
    }
  gsl_vector_scale(X, 1.0/sum);
}

void sample(gsl_vector* X, gsl_rng * rng)
{
  for(unsigned int i=0;i<X->size;i++)
    {
      double x = gsl_vector_get(X,i);
      double r = gsl_rng_uniform(rng);
      gsl_vector_set(X, i, r<x);
    }
}

void print_vector(FILE *stream, const gsl_vector * V)
{
  fprintf(stream,"[");
  for(unsigned int i=0;i<V->size-1;i++)
    fprintf(stream,"%f,", gsl_vector_get(V,i));
  fprintf(stream, "%f]", gsl_vector_get(V,V->size-1));
}

double matrix_norm(gsl_matrix * M)
{
  double norm=0;
  for(unsigned j=0;j<M->size2;j++)
    {
      gsl_vector_view column = gsl_matrix_column (M, j);
      norm += gsl_blas_dnrm2 (&column.vector);
    }
  return norm;
}
