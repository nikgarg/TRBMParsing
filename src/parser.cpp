#include "parser.h"

ParserConfiguration::ParserConfiguration()
{
}

ParserConfiguration::ParserConfiguration(vector<Word*>& words, FeatureModel *feature_model, ConnectionModel *connection_model, LearningModel *_learning_model)
{
  learning_model = _learning_model;
  learning_instance = learning_model->get_new_instance();
  partial_parse_tree = new ParseTree();
  for(unsigned int i=0;i<words.size();i++)
    partial_parse_tree->add_node(words[i]->id, words[i], words[i]->desc);
  state = new State(feature_model, connection_model);
  for(unsigned int i=0;i<words.size();i++)
    state->queue_enqueue(words[i]->id);
  state->recalculate_features(partial_parse_tree);
  log_probability = 0;  
}

ParserConfiguration::~ParserConfiguration()
{
  if(partial_parse_tree!=NULL)
    delete partial_parse_tree;
  if(state!=NULL)
    delete state;
  if(learning_instance!=NULL)
    delete learning_instance;
}

ParserConfiguration* ParserConfiguration::clone()
{
  ParserConfiguration* new_config = new ParserConfiguration();
  new_config->learning_model = learning_model;
  new_config->learning_instance = new_config->learning_model->clone_instance(learning_instance);
  new_config->partial_parse_tree = partial_parse_tree->clone();
  new_config->state = state->clone();
  new_config->history = history;
  new_config->log_probability = log_probability;
  return new_config;
}

void ParserConfiguration::bury()
{
  delete partial_parse_tree;
  delete state;
  delete learning_instance;
  log_probability = -DBL_MAX;
}

void ParserConfiguration::make_transition(string action, double probability, int labelid=-1)
{
  state->set_previous_operation(action);
  history.push_back(make_pair(action, probability));
  log_probability += probability;

  if (action=="LA")
    {
      int stack_top = state->stack_top();
      int queue_front = state->queue_front();
      partial_parse_tree->add_edge(queue_front, stack_top, labelid);
      state->stack_pop();
    }
  else if (action=="RA")
    {
      int stack_top = state->stack_top();
      int queue_front = state->queue_front();
      partial_parse_tree->add_edge(stack_top, queue_front, labelid);
    }
  else if (action=="R")
    {
      state->stack_pop();
    }
  else if (action=="S")
    {
      int queue_front = state->queue_front();
      state->queue_dequeue();
      state->stack_push(queue_front);
    }
  state->recalculate_features(partial_parse_tree);    
}

int ParserConfiguration::possible_transitions(unordered_map<string,bool>& transitions)
{
  int n_transitions = 0;
  transitions["LA"] = false;
  transitions["RA"] = false;
  transitions["R"] = false;
  transitions["S"] = false;

  if (state->get_previous_operation()=="RA")
    {
      transitions["S"] = true;
      n_transitions += 1;
      return n_transitions;
    }

  int queue_front = state->queue_front();
  if(!state->stack_empty())
    {
      int stack_top = state->stack_top();
      transitions["R"] = true;
      n_transitions += 1;
      if(queue_front!=-1)
	{
	  if(partial_parse_tree->nodes[stack_top]->parent==NULL)
	    {
	      transitions["LA"] = true;
	      n_transitions += 1;	      
	    }
	  if(partial_parse_tree->nodes[queue_front]->parent==NULL)
	    {
	      transitions["RA"] = true;
	      n_transitions += 1;
	    }
	}
    }

  if(queue_front!=-1)
    {
      transitions["S"] = true;
      n_transitions += 1;
    }


  return n_transitions;

}


Parser::Parser(unordered_map<string,string> _params, bool train_new, string model_type)
{
  params = _params;
  beam_size = atoi(params["beam-size"].c_str());
  label_branch_size = atoi(params["label-branch-size"].c_str());
  learning_rate_init = atof(params["learning-rate-init"].c_str());
  learning_rate_red_rate = atof(params["learning-rate-red-rate"].c_str());
  learning_rate_max_red = atof(params["learning-rate-max-red"].c_str());
  weight_decay_init = atof(params["weight-decay-init"].c_str());
  weight_decay_red_rate = atof(params["weight-decay-red-rate"].c_str());
  weight_decay_max_red = atof(params["weight-decay-max-red"].c_str());
  momentum = atof(params["momemtum"].c_str());

  learning_rate = learning_rate_init;
  weight_decay = weight_decay_init;

  feature_model = new FeatureModel(params["feature-model"]);
  connection_model = new ConnectionModel(params["connection-model"]);
  data = new Data();

  cerr<<feature_model->to_string();
  cerr<<connection_model->to_string();

  decisions2id["Action"] = 0;
  decisions2id["LabelLA"] = 1;
  decisions2id["LabelRA"] = 2;
  decisions2id["POS"] = 3;
  decision_size[decisions2id["Action"]] = 4;  //LA, RA, S, R
  action2id["LA"] = 0;
  action2id["RA"] = 1;
  action2id["S"] = 2;
  action2id["R"] = 3;

  if(train_new)
    {
      data->build_lexicon(params["train-set"]);
      data->save_lexicon(params["lexicon"]);
      decision_size[decisions2id["LabelLA"]] = data->id2label.size();
      decision_size[decisions2id["LabelRA"]] = data->id2label.size();
      decision_size[decisions2id["POS"]] = data->id2pos.size();
      char buf[10];
      for(unsigned i=0;i<data->id2pos.size();i++)
	{
	  sprintf(buf, "LEX-%d",i);
          int decid = decisions2id.size();
	  decisions2id[string(buf)] = decid;
	  decision_size[decisions2id[string(buf)]] = data->id2lex[i].size();
	}
      learning_model = new LearningModel(model_type, atoi(params["hidden-variables"].c_str()), feature_model->features.size(), connection_model->connections.size(), &decision_size);
    }
  else
    {
      read_from_file();
      data->read_lexicon(params["lexicon"]);
      decision_size[decisions2id["LabelLA"]] = data->id2label.size();
      decision_size[decisions2id["LabelRA"]] = data->id2label.size();
      decision_size[decisions2id["POS"]] = data->id2pos.size();
      char buf[10];
      for(unsigned i=0;i<data->id2pos.size();i++)
	{
	  sprintf(buf, "LEX-%d",i);
          int decid = decisions2id.size();
	  decisions2id[string(buf)] = decid;
	  decision_size[decisions2id[string(buf)]] = data->id2lex[i].size();
	}
      learning_model = new LearningModel(params["learning-model"]);
    }

  unordered_set<int> ignore_tokens_with_labelid;
  ignore_tokens_with_labelid.insert(data->label2id["P"]);
  evaluator = new Evaluator(ignore_tokens_with_labelid);
}

Parser::~Parser()
{
  delete feature_model;
  delete connection_model;
  delete learning_model;
  delete evaluator;
  delete data;
}

void Parser::train()
{
  cerr<<"Starting training on file: "<<params["train-set"]<<"\n";
  cout<<"Starting training on file: "<<params["train-set"]<<"\n";

  data->open_file("train", params["train-set"]);
  data->open_file("development", params["dev-set"]);

  int max_iter = atoi(params["max-train-iter"].c_str());
  int iter = 0;
  int n_sentences = 0;
  int n_sentences_used = 0;
  int n_dev_sentences = 0;
  int n_dev_sentences_used = 0;
  double previous_likelihood = INT_MIN;
  double total_likelihood = INT_MIN;
  double dev_previous_likelihood = INT_MIN;
  double dev_total_likelihood = INT_MIN;
  double dev_highest_likelihood = INT_MIN;
  int updates_since_last_save = 0;
  int n_small_changes = 0;
  int n_likelihood_decreases = 0;
  int n_likelihood_decreases_wrt_highest = 0;
  int iter_since_learning_rate_update = 0;
  int iter_since_weight_decay_update = 0;
  int weight_decay_change_interval = 10;

  double change;
  Sentence *sentence;

  vector< vector<double>* > word_repr;

  string fl_highest_likelihood_weights = params["tmp-dir"] + "/highest_likelihood_weights.dat";
  string fl_highest_likelihood_weights_info = params["tmp-dir"] + "/highest_likelihood_weights.info";

  cerr<<"Setting learning rate: "<<learning_rate<<"\n";
  cout<<"Setting learning rate: "<<learning_rate<<"\n";
  learning_model->set_learning_rate(learning_rate);
  cerr<<"Setting weight decay: "<<weight_decay<<"\n";
  cout<<"Setting weight decay: "<<weight_decay<<"\n";
  learning_model->set_weight_decay(weight_decay);
  cerr<<"Setting momentum: "<<momentum<<"\n";
  cout<<"Setting momentum: "<<momentum<<"\n";
  learning_model->set_momentum(momentum);

  while(iter<max_iter)
    {
      previous_likelihood = total_likelihood;
      iter += 1;
      iter_since_learning_rate_update += 1;
      iter_since_weight_decay_update += 1;
      n_sentences = 0;
      n_sentences_used = 0;
      total_likelihood = 0;
      data->reset("train");
      while((sentence=data->read_sentence("train")))
	{
	  n_sentences += 1;
	  cerr<<"iter#"<<iter<<", training sentence#"<<n_sentences<<"\n";
	  cout<<"iter#"<<iter<<", training sentence#"<<n_sentences<<"\n";
	  //cout<<"gold_parse_tree:\n"<<sentence->gold_parse_tree->to_string()<<"\n";
	  if (sentence->gold_parse_tree->is_well_formed())
	    {
	      double likelihood = simulate_sentence(sentence->words, sentence->gold_parse_tree, true, false, word_repr, iter);
	      cout<<"likelihood: "<<likelihood<<"\n";
	      if (likelihood!=0)
		{
		  n_sentences_used += 1;
		  total_likelihood += likelihood;
		  updates_since_last_save += 1;
		}
	    }
	  else
	    {
	      cerr<<"...not well formed. Ignored!\n";
	      cout<<"...not well formed. Ignored!\n";
	    }
	  delete sentence;
	}

      n_dev_sentences = 0;
      n_dev_sentences_used = 0;
      dev_previous_likelihood = dev_total_likelihood;
      dev_total_likelihood = 0;
      data->reset("development");
      while((sentence=data->read_sentence("development")))
	{
	  n_dev_sentences += 1;
	  cerr<<"iter#"<<iter<<", dev sentence#"<<n_dev_sentences<<"\n";
	  cout<<"iter#"<<iter<<", dev sentence#"<<n_dev_sentences<<"\n";
	  //cout<<"gold_parse_tree:\n"<<sentence->gold_parse_tree->to_string()<<"\n";
	  if (sentence->gold_parse_tree->is_well_formed())
	    {
	      double likelihood = simulate_sentence(sentence->words, sentence->gold_parse_tree, false, false, word_repr, iter);
	      cout<<"likelihood: "<<likelihood<<"\n";
	      if (likelihood!=0)
		{
		  n_dev_sentences_used += 1;
		  dev_total_likelihood += likelihood;
		}
	    }
	  else
	    {
	      cerr<<"...not well formed. Ignored!\n";
	      cout<<"...not well formed. Ignored!\n";
	    }
	  delete sentence;
	}      

      cerr<<"Iter#"<<iter<<", Likelihoods: [Train] "<<total_likelihood<<"\t[Dev] "<<dev_total_likelihood<<"\n";
      cout<<"#Sentences: [Train] "<<n_sentences_used<<" ("<<n_sentences<<")\t[Dev] "<<n_dev_sentences_used<<" ("<<n_dev_sentences<<")\n";
      cout<<"Iter#"<<iter<<", Likelihoods: [Train] "<<total_likelihood<<"\t[Dev] "<<dev_total_likelihood<<"\n";

      /*
      if (updates_since_last_save>=1000)
	{
	  save();
	  updates_since_last_save = 0;
	}
      */

      if (dev_previous_likelihood)
	{
	  change = fabs((dev_total_likelihood-dev_previous_likelihood)/dev_previous_likelihood);
	  if (change<1e-5)
	    {
	      cerr<<"Very small change in dev likelihood: "<<change<<"\n";
	      cout<<"Very small change in dev likelihood: "<<change<<"\n";
	      n_small_changes += 1;
	      if (n_small_changes>=3)
		{
		  cerr<<n_small_changes<<" very small changes in a row. Terminating...\n";
		  cout<<n_small_changes<<" very small changes in a row. Terminating...\n";
		  break;
		}
	    }
	  else
	    {
	      n_small_changes = 0;
	    }
	}

      bool reduce_learning_rate = false;

      if(dev_total_likelihood>dev_highest_likelihood)
	{
	  n_likelihood_decreases_wrt_highest = 0;
	  dev_highest_likelihood = dev_total_likelihood;
	  save();
	  updates_since_last_save = 0;
	  learning_model->save_weights_binary(fl_highest_likelihood_weights_info, fl_highest_likelihood_weights);
	}
      else
	{
	  n_likelihood_decreases_wrt_highest += 1;
	  if(n_likelihood_decreases_wrt_highest>=3)
	    {
	      cerr<<"Dev Likelihood lower than highest for "<<n_likelihood_decreases_wrt_highest<<" consecutive iterations.\n";
	      cout<<"Dev Likelihood lower than highest for "<<n_likelihood_decreases_wrt_highest<<" consecutive iterations.\n";
	      reduce_learning_rate = true;
	      if(n_likelihood_decreases_wrt_highest>=10)
		{
		  cerr<<"Dev Likelihood lower than highest for "<<n_likelihood_decreases_wrt_highest<<" consecutive iterations. Terminating...\n";
		  cout<<"Dev Likelihood lower than highest for "<<n_likelihood_decreases_wrt_highest<<" consecutive iterations. Terminating...\n";
		  break;
		}
	    }
	}


      if(dev_previous_likelihood>dev_total_likelihood)
	{
	  cerr<<"Dev likelihood decreased!\n";
	  cout<<"Dev likelihood decreased!\n";
	  n_likelihood_decreases += 1;
	  cerr<<"clearing momentum\n";
          cout<<"clearing momentum\n";
          learning_model->clear_momentum();
	  //cerr<<"Setting momentum: 0.1"<<"\n";
	  //cout<<"Setting momentum: 0.1"<<"\n";
	  //learning_model->set_momentum(0.1);
	  if(n_likelihood_decreases>=2)
	    {
	      cerr<<"Dev Likelihood decreased for "<<n_likelihood_decreases<<" consecutive iterations.\n";
	      cout<<"Dev Likelihood decreased for "<<n_likelihood_decreases<<" consecutive iterations.\n";
	      reduce_learning_rate = true;
	    }
	}
      else
	{
	  n_likelihood_decreases = 0;
	}

      if(iter_since_weight_decay_update>=weight_decay_change_interval && iter_since_learning_rate_update>=2)
	{
	  float new_weight_decay = weight_decay * weight_decay_red_rate;
	  if(new_weight_decay/weight_decay_init>weight_decay_max_red)
	    {
	      weight_decay = new_weight_decay;
	      cerr<<"iter#"<<iter<<" Reducing weight decay to: "<<weight_decay<<"\n";
	      cout<<"iter#"<<iter<<" Reducing weight decay to: "<<weight_decay<<"\n";
	      learning_model->set_weight_decay(weight_decay);
	      iter_since_weight_decay_update = 0;
	    }
	  weight_decay_change_interval *= 0.9;
	  if(weight_decay_change_interval<3)
	    weight_decay_change_interval = 3;
	}

      if(reduce_learning_rate && iter_since_learning_rate_update>=2 && iter_since_weight_decay_update>=1)
	{
	  float new_learning_rate = learning_rate * learning_rate_red_rate;
	  if(new_learning_rate/learning_rate_init>learning_rate_max_red)
	    {
	      learning_rate = new_learning_rate;
	      cerr<<"iter#"<<iter<<" Reducing learning rate to: "<<learning_rate<<"\n";
	      cout<<"iter#"<<iter<<" Reducing learning rate to: "<<learning_rate<<"\n";
	      learning_model->set_learning_rate(learning_rate);
	      iter_since_learning_rate_update = 0;
	    }
	  learning_model->read_weights_binary(fl_highest_likelihood_weights_info, fl_highest_likelihood_weights);
	}

    }

  learning_model->read_weights_binary(fl_highest_likelihood_weights_info, fl_highest_likelihood_weights);

  data->close_file("train");
  data->close_file("development");
}

void Parser::test()
{
  cerr<<"Starting testing on file: "<<params["test-set"]<<"\n";
  cout<<"Starting testing on file: "<<params["test-set"]<<"\n";
  data->open_file("test", params["test-set"]);

  ofstream fd_output;
  fd_output.open(params["parser-output"].c_str());

  int n_sentences = 0;
  int n_sentences_used = 0;
  Sentence *sentence;

  while((sentence=data->read_sentence("test")))
    {
      n_sentences += 1;
      cerr<<"test sentence#"<<n_sentences<<"\n";
      cout<<"test sentence#"<<n_sentences<<"\n";
      cout<<"gold_parse_tree:\n"<<sentence->gold_parse_tree->to_string()<<"\n";
      //if (sentence->gold_parse_tree->is_well_formed())
      //{
	  n_sentences_used += 1;
	  //change beam size
	  /*
	  int temp = beam_size;
	  if (sentence->words.size()>40 && beam_size>1)
	    {
	      cerr<<"Long sentence with "<<sentence->words.size()<<" words. ";
	      float f = ((float)sentence->words.size())/40.0;
	      int new_beam_size = (int)(beam_size/f);
	      if (new_beam_size>1)
		beam_size = new_beam_size;
	      cerr<<"Using beam size of "<<beam_size<<". (Default:"<<temp<<")\n";
	    }
	  */
	  sentence->predicted_parse_tree = predict_parse_tree(sentence->words);
	  evaluator->update(sentence->gold_parse_tree, sentence->predicted_parse_tree);
	  fd_output<<data->write_output(sentence);
	  //beam_size = temp;
	  if (sentence->predicted_parse_tree!=NULL)
	    {
	      cout<<"predicted_parse_tree:\n"<<sentence->predicted_parse_tree->to_string()<<"\n";
	    }
	  else
	    {
	      cout<<"predicted_parse_tree:\nNone\n";
	    }
	  //}
	  //else
	  //{
	  //cerr<<"...not well formed. Ignored!\n";
	  //cout<<"...not well formed. Ignored!\n";
	  //}   
      delete sentence;

      if(n_sentences%50==0)
	{
	  cout<<"Results:\n"<<get_results()<<"\n";
	}
    }
  cerr<<"Testing: total sentences: "<<n_sentences<<"\n"
      <<"Testing: used sentences:  "<<n_sentences_used<<"\n";
  cout<<"Testing: total sentences: "<<n_sentences<<"\n"
      <<"Testing: used sentences:  "<<n_sentences_used<<"\n";
  data->close_file("test");
  fd_output.close();
}

void Parser::print_word_representations(string data_file, string repr_file)
{
  cerr<<"Printing word representations for file: "<<data_file.c_str()<<"\n";
  cout<<"Printing word representations for file: "<<data_file.c_str()<<"\n";

  data->open_file("repr", data_file);

  ofstream fd_repr;
  fd_repr.open(repr_file.c_str());

  int n_sentences = 0;
  int n_sentences_used = 0;
  double total_likelihood = 0;
  Sentence *sentence;
  vector< vector<double>* > repr;
  int repr_size;

  n_sentences = 0;
  n_sentences_used = 0;
  while((sentence=data->read_sentence("repr")))
    {
      n_sentences += 1;
      cerr<<"Representation for sentence#"<<n_sentences<<"\n";
      cout<<"Representation for sentence#"<<n_sentences<<"\n";

      if (sentence->gold_parse_tree->is_well_formed())
	{
	  double likelihood = simulate_sentence(sentence->words, sentence->gold_parse_tree, false, true, repr, 1);
	  repr_size = repr[0]->size();
	  cout<<"likelihood: "<<likelihood<<"\n";
	  if (likelihood!=0)
	    {
	      n_sentences_used += 1;
	      total_likelihood += likelihood;
	      for(unsigned int i=0;i<sentence->words.size()-1;i++)
		{
		  fd_repr<<sentence->words[i]->input_line<<"\t";
		  for (int j=0;j<repr_size-1;j++)
		    fd_repr<<(*repr[i])[j]<<",";
		  fd_repr<<(*repr[i])[repr_size-1]<<"\n";
		}
	      fd_repr<<"\n";
	    }
	  else
	    {
	      for(unsigned int i=0;i<sentence->words.size()-1;i++)
		fd_repr<<sentence->words[i]->input_line<<"\t\n";
	      fd_repr<<"\n";	 	      
	    }
	  for(unsigned i=0;i<repr.size();i++)
	    delete repr[i];
	  repr.clear();
	}
      else
	{
	  cerr<<"...not well formed. Ignored!\n";
	  cout<<"...not well formed. Ignored!\n";
	  for(unsigned int i=0;i<sentence->words.size()-1;i++)
	    fd_repr<<sentence->words[i]->input_line<<"\t\n";	  
	  fd_repr<<"\n";
	}

      delete sentence;
    }      

  cerr<<"Likelihood: "<<total_likelihood<<"\n";
  cerr<<"#Sentences: "<<n_sentences_used<<" ("<<n_sentences<<")\n";
  cout<<"Likelihood: "<<total_likelihood<<"\n";
  cout<<"#Sentences: "<<n_sentences_used<<" ("<<n_sentences<<")\n";

  data->close_file("repr");
  fd_repr.close();
}


string Parser::get_results()
{
  return evaluator->get_results();
}

void Parser::read_from_file()
{
  const char *fl_parsing_params = params["parser-params"].c_str();
  cerr<<"reading Parsing Parameters from file: "<<fl_parsing_params<<"...";
  ifstream fd;
  fd.open(fl_parsing_params);
  try
    {
      xmlpp::TextReader reader(fl_parsing_params);
      while(reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="LearningRate")
		{
		  reader.move_to_element();
		  reader.read();
		  learning_rate = atof(reader.get_value().c_str());
		}
	      else if(reader.get_name()=="WeightDecay")
		{
		  reader.move_to_element();
		  reader.read();
		  weight_decay = atof(reader.get_value().c_str());
		}
	    }
	}
    }
  catch(const std::exception& e)
    {
      cerr << "Exception caught: " << e.what() << endl;
    }  

  fd.close();
  cerr<<"done!\n";
}

void Parser::save()
{
  const char *fl_parsing_params = params["parser-params"].c_str();
  cerr<<"\nsaving Parsing Parameters in file: "<<fl_parsing_params<<"...";
  cout<<"saving Parsing Parameters in file: "<<fl_parsing_params<<"...";
  char bkp_file[1000];
  strcpy(bkp_file, fl_parsing_params);
  strcat(bkp_file, ".bkp");
  rename(fl_parsing_params, bkp_file);
  ofstream fd;  
  fd.open(fl_parsing_params);
  fd<<"<ParsingParams>\n"
    <<"\t<LearningRate>"<<learning_rate<<"</LearningRate>\n"
    <<"\t<WeightDecay>"<<weight_decay<<"</WeightDecay>\n"
    <<"</ParsingParams>\n";
  fd.close();
  remove(bkp_file);
  cerr<<"done!\n";
  cout<<"done!\n";

  cerr<<"saving Learning Model in file: "<<params["learning-model"]<<"...";
  cout<<"saving Learning Model in file: "<<params["learning-model"]<<"...";
  learning_model->save(params["learning-model"]);
  cerr<<"done!\n\n";
  cout<<"done!\n\n";  
}


double Parser::simulate_sentence(vector<Word*>& words, ParseTree *parsetree, bool learn, bool get_repr, vector< vector<double>* >& repr, int iter)
{
  ParserConfiguration *config = new ParserConfiguration(words, feature_model, connection_model, learning_model);
  unordered_map<string,bool> possible_transitions;
  unordered_set<int> stack_nodes;
  //cout<<"gold parse tree:\n"<<parsetree->to_string()<<"\n";
  while (config->possible_transitions(possible_transitions))
    {
      //cout<<"----------------\n";
      //cout<<"state: "<<config->state->to_string()<<"\n";
      //cout<<"possible_transitions:"<<possible_transitions["LA"]<<" "<<possible_transitions["RA"]<<" "<<possible_transitions["R"]<<" "<<possible_transitions["S"]<<"\n";
      //cout<<"partial parse tree:\n"<<config->partial_parse_tree->to_string();
      
      if (possible_transitions["LA"])
	{
	  unordered_map<string,string> args;
	  int stack_top = config->state->stack_top();
	  int queue_front = config->state->queue_front();
	  if (parsetree->nodes[stack_top]->parent!=NULL && parsetree->nodes[stack_top]->parent->id==queue_front)
	    {
	      int labelid = parsetree->nodes[stack_top]->labelid;
	      learning_model->update(config->learning_instance, config->state, decisions2id["Action"], action2id["LA"], true);
	      learning_model->update(config->learning_instance, config->state, decisions2id["LabelLA"], labelid, false);
	      learning_model->process_last_step(config->learning_instance, true);
	      config->make_transition("LA", 0, labelid);
	      stack_nodes.erase(stack_top);
	      continue;
	    }
	}
      if (possible_transitions["RA"])
	{
	  unordered_map<string,string> args;
	  int stack_top = config->state->stack_top();
	  int queue_front = config->state->queue_front();
	  if (parsetree->nodes[queue_front]->parent!=NULL && parsetree->nodes[queue_front]->parent->id==stack_top)
	    {
	      int labelid = parsetree->nodes[queue_front]->labelid;
	      learning_model->update(config->learning_instance, config->state, decisions2id["Action"], action2id["RA"], true);
	      learning_model->update(config->learning_instance, config->state, decisions2id["LabelRA"], labelid, false);
	      learning_model->process_last_step(config->learning_instance, true);
	      config->make_transition("RA", 0, labelid);
	      continue;
	    }
	}
      if (possible_transitions["S"])
	{
	  unordered_map<string,string> args;
	  int queue_front = config->state->queue_front();
	  bool go_ahead = true;
	  if (config->partial_parse_tree->nodes[queue_front]->parent==NULL && parsetree->nodes[queue_front]->parent!=NULL && stack_nodes.find(parsetree->nodes[queue_front]->parent->id)!=stack_nodes.end())
	    {
	      //cout<<"can't shift. parent in the stack\n";
	      go_ahead = false;
	    }
	  if(go_ahead)
	    {
	      for(vector<ParseNode*>::iterator it=parsetree->nodes[queue_front]->children.begin(); it!=parsetree->nodes[queue_front]->children.end(); it++)
		{
		  ParseNode* child = *it;
		  int cid = child->id;
		  if(config->partial_parse_tree->nodes[cid]->parent==NULL && parsetree->nodes[cid]->parent!=NULL && parsetree->nodes[cid]->parent->id==queue_front && stack_nodes.find(cid)!=stack_nodes.end())
		    {
		      //cout<<"can't shift. child "<<cid<<" in the stack\n";
		      go_ahead = false;
		      break;
		    }
		}
	    }
	  
	  if(go_ahead)
	    {
	      Word *word = (Word*)parsetree->nodes[config->state->queue_element(1)]->data;
	      args["pos"] = word->pos;
	      args["lex"] = word->lex;
	      learning_model->update(config->learning_instance, config->state, decisions2id["Action"], action2id["S"], true);
	      learning_model->update(config->learning_instance, config->state, decisions2id["POS"], word->posid, false);
	      char buf[10];
	      sprintf(buf, "LEX-%d",word->posid);
	      learning_model->update(config->learning_instance, config->state, decisions2id[string(buf)], word->lexid, false);
	      learning_model->process_last_step(config->learning_instance, true);
	      if(get_repr)
		repr.push_back(config->learning_instance->get_hidden_vector_of_last_step());
	      config->make_transition("S", 0);
	      stack_nodes.insert(queue_front);
	      continue;
	    }
	}
      if (possible_transitions["R"])  //may be include the attachment hinderance concept
	{
	  unordered_map<string,string> args;
	  int stack_top = config->state->stack_top();
	  if (config->partial_parse_tree->nodes[stack_top]->parent!=NULL || parsetree->nodes[stack_top]->parent==NULL)
	    {	  
	      if (config->partial_parse_tree->nodes[stack_top]->children.size()==parsetree->nodes[stack_top]->children.size())
		{
		  learning_model->update(config->learning_instance, config->state, decisions2id["Action"], action2id["R"], true);
		  learning_model->process_last_step(config->learning_instance, true);
		  config->make_transition("R", 0);
		  stack_nodes.erase(stack_top);
		  continue;
		}
	    }
	}
      cerr <<"Error: Could not execute any possible transition! Ignoring this sentence!\n";
      return 0;
    }
  
  if(learn)
    learning_model->learn(config->learning_instance, iter);

  double likelihood = config->learning_instance->get_probability();

  delete config;
  return likelihood;
}

ParseTree* Parser::predict_parse_tree(vector<Word*>& words)
{
  priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>* candidate_configs = new priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>;
  ParserConfiguration* config = new ParserConfiguration(words, feature_model, connection_model, learning_model);
  priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, LessParserConfiguration>* PQ = new priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, LessParserConfiguration>;
  //priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>* reversePQ = new priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>;
  priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>* PQ_next = new priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, GreaterParserConfiguration>;
  PQ_next->push(config);

  unordered_map<string,bool> possible_transitions;
  double min_log_probability = -DBL_MAX;
  int n = 1;
  
  while (!PQ_next->empty())
    {      
      while(!PQ->empty())
	{
	  config = PQ->top();
	  PQ->pop();
	  delete config;
	}
      while (((int)PQ_next->size())>beam_size)
	{
	  config = PQ_next->top();
	  PQ_next->pop();
	  delete config;
	}
      while (!PQ_next->empty())
	{
	  config = PQ_next->top();
	  PQ_next->pop();
	  PQ->push(config);
	}
      min_log_probability = -DBL_MAX;
      
      while(!PQ->empty())
	{
	  config = PQ->top();
	  PQ->pop();
	  if(config->log_probability<min_log_probability)
	    {
	      delete config;
	      break;
	    }

	  if (config->possible_transitions(possible_transitions))
	    {
	      if (possible_transitions["LA"])
		{
		  //cout<<"LA\n";
		  ParserConfiguration* config1 = config->clone();
		  learning_model->update(config1->learning_instance, config1->state, decisions2id["Action"], action2id["LA"], true);
		  learning_model->update(config1->learning_instance, config1->state, decisions2id["LabelLA"], -1, false);
		  learning_model->process_last_step(config1->learning_instance, false);
		  double action_prob = config1->learning_instance->get_probability_of_last_decision(decisions2id["Action"]);
		  vector<double>* label_prob = config1->learning_instance->get_probability_vector_of_last_decision(decisions2id["LabelLA"]);
		  vector< pair<int,double> > label_prob1(label_prob->size(), make_pair(-1,0.0));
	          for(unsigned i=0;i<label_prob->size();i++)
		    label_prob1[i] = make_pair(i,(*label_prob)[i]);
		  delete label_prob;
		  for (int i=0;i<label_branch_size;i++)
		    {
		      vector< pair<int,double> >::iterator top_label = max_element(label_prob1.begin(), label_prob1.end(), LessLabelProbability());
		      double step_prob = action_prob + top_label->second;
		      if(config1->log_probability+step_prob>=min_log_probability)
			{
			  ParserConfiguration* new_config = config1->clone();
			  learning_model->set_target_of_last_decision(new_config->learning_instance, decisions2id["LabelLA"], top_label->first);
			  new_config->make_transition("LA", step_prob, top_label->first);
			  PQ->push(new_config);
			  label_prob1.erase(top_label);
			}
		      else
			break;		      
		    }

		  delete config1;
		}
	      if (possible_transitions["RA"])
		{
		  //cout<<"RA\n";
		  ParserConfiguration* config1 = config->clone();
		  learning_model->update(config1->learning_instance, config1->state, decisions2id["Action"], action2id["RA"], true);
		  learning_model->update(config1->learning_instance, config1->state, decisions2id["LabelRA"], -1, false);
		  learning_model->process_last_step(config1->learning_instance, false);
		  double action_prob = config1->learning_instance->get_probability_of_last_decision(decisions2id["Action"]);
		  vector<double>* label_prob = config1->learning_instance->get_probability_vector_of_last_decision(decisions2id["LabelRA"]);
		  vector< pair<int,double> > label_prob1(label_prob->size(), make_pair(-1,0.0));
	          for(unsigned i=0;i<label_prob->size();i++)
		    label_prob1[i] = make_pair(i,(*label_prob)[i]);
		  delete label_prob;
		  for (int i=0;i<label_branch_size;i++)
		    {
		      vector< pair<int,double> >::iterator top_label = max_element(label_prob1.begin(), label_prob1.end(), LessLabelProbability());
		      double step_prob = action_prob + top_label->second;
		      if(config1->log_probability+step_prob>=min_log_probability)
			{
			  ParserConfiguration* new_config = config1->clone();
			  learning_model->set_target_of_last_decision(new_config->learning_instance, decisions2id["LabelRA"], top_label->first);
			  new_config->make_transition("RA", step_prob, top_label->first);
			  PQ->push(new_config);
			  label_prob1.erase(top_label);
			}
		      else
			break;		      
		    }

		  delete config1;
		}
	      if (possible_transitions["R"])
		{
		  //cout<<"R\n";
		  ParserConfiguration* new_config = config->clone();
		  learning_model->update(new_config->learning_instance, new_config->state, decisions2id["Action"], action2id["R"], true);
		  learning_model->process_last_step(new_config->learning_instance, false);
		  double step_prob = new_config->learning_instance->get_probability_of_last_step();
		  if(new_config->log_probability+step_prob>=min_log_probability)
		    {
		      new_config->make_transition("R", step_prob);
		      PQ->push(new_config);
		    }
		  else
		    delete new_config;
		}
	      if (possible_transitions["S"])
		{
		  //cout<<"S\n";
		  ParserConfiguration* new_config = config->clone();
		  Word *word = (Word*)new_config->partial_parse_tree->nodes[new_config->state->queue_element(1)]->data;
		  learning_model->update(new_config->learning_instance, new_config->state, decisions2id["Action"], action2id["S"], true);
		  learning_model->update(new_config->learning_instance, new_config->state, decisions2id["POS"], word->posid, false);
		  char buf[10];
		  sprintf(buf, "LEX-%d",word->posid);
		  learning_model->update(new_config->learning_instance, new_config->state, decisions2id[string(buf)], word->lexid, false);
		  learning_model->process_last_step(new_config->learning_instance, false);
		  double step_prob = new_config->learning_instance->get_probability_of_last_step();
		  if(new_config->log_probability+step_prob>=min_log_probability)
		    {		      
		      new_config->make_transition("S", step_prob);
		      PQ_next->push(new_config);
		      if(((int)PQ_next->size())>=beam_size)
			{
			  while (((int)PQ_next->size())>beam_size)
			    {
			      ParserConfiguration* c = PQ_next->top();
			      PQ_next->pop();
			      delete c;
			    }
			  if(PQ_next->top()->log_probability>min_log_probability)
			    {
			      min_log_probability = PQ_next->top()->log_probability;
			      priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, LessParserConfiguration>* newPQ = new priority_queue<ParserConfiguration*, vector<ParserConfiguration*>, LessParserConfiguration>;
			      double mp;
			      while(!PQ->empty())
				{
				  ParserConfiguration* c = PQ->top();
				  PQ->pop();
				  if(c->log_probability>=min_log_probability)
				    {
				      newPQ->push(c);
				      mp = c->log_probability;
				    }
				  else
				    {
				      delete c;
				      break;
				    }
				}
			      while(!PQ->empty())
				{
				  ParserConfiguration* c = PQ->top();
				  delete c;
				  PQ->pop();
				}		  
			      delete PQ;
			      PQ = newPQ;
			    }
			}
		    }
		  else
		    delete new_config;
		}
	      delete config;
            }
	  else
	    {
	      candidate_configs->push(config);
	      while (((int)candidate_configs->size())>n)
		{
		  ParserConfiguration* c = candidate_configs->top();
		  candidate_configs->pop();
		  delete c;
		}
	    }
         }
    }

  while(!PQ->empty())
    {
      config = PQ->top();
      PQ->pop();
      delete config;
    }

  delete PQ;
  delete PQ_next;

  if (!candidate_configs->empty())
    {
      ParserConfiguration* c = candidate_configs->top();
      candidate_configs->pop();
      while (!candidate_configs->empty())
	{
	  delete c;
	  c = candidate_configs->top();
	  candidate_configs->pop();
	}
      ParseTree *T = c->partial_parse_tree->clone();
      delete c;
      delete candidate_configs;
      return T;
    }

  delete candidate_configs;
  return NULL;
}

