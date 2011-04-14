#include "state.h"

State::State(FeatureModel* _feature_model, ConnectionModel* _connection_model)
{
  feature_model = _feature_model;
  connection_model = _connection_model;
  previous_operation = "";
  features.reserve(feature_model->features.size());
  for(unsigned int i=0;i<feature_model->features.size();i++)
    features.push_back("#NONE#");
  connection_features.reserve(connection_model->connections.size());
  for(unsigned int i=0;i<connection_model->connections.size();i++)
    connection_features.push_back(make_pair("#NONE#","#NONE#"));
}

State::~State()
{
}

State* State::clone()
{
  State* new_state = new State(feature_model, connection_model);
  new_state->S = S;
  new_state->Q = Q;
  new_state->previous_operation = previous_operation;
  new_state->features = features;
  new_state->connection_features = connection_features;
  return new_state;  
}

string State::to_string()
{
  unsigned int i;
  ostringstream	result;
  result << "S:[";
  for(i=0;i<S.size();i++)
    result << S[i] <<",";
  result << "] Q:[";
  for(i=0;i<Q.size();i++)
    result << Q[i] <<",";
  result << "]";
  result << " prev_op:" << previous_operation;
  result << " features:[";
  for(i=0;i<features.size();i++)
    result<<features[i]<<",";
  result << "]";
  result << " connection_features:[";
  for(i=0;i<connection_features.size();i++)
    result<<"("<<connection_features[i].first<<":"<<connection_features[i].second<<"),";
  result << "]";
  return result.str();
}

void State::stack_push(int val)
{
  S.push_back(val);
}

int State::stack_top()
{
  return S.back();
}

bool State::stack_empty()
{
  return S.empty();
}

void State::stack_pop()
{
  S.pop_back();
}

void State::queue_enqueue(int val)
{
  Q.push_back(val);
}

int State::queue_front()
{
  return Q.front();
}

int State::queue_element(int position)
{
  return Q[position];
}

bool State::queue_empty()
{
  return Q.empty();
}

int State::queue_size()
{
  return Q.size();
}

void State::queue_dequeue()
{
  Q.pop_front();
}

string State::get_previous_operation()
{
  return previous_operation;
}

void State::set_previous_operation(string op)
{
  previous_operation = op;
}

vector<string> State::get_features()
{
  return features;
}

void State::recalculate_features(ParseTree* parsetree)
{
  for(unsigned int i=0;i<feature_model->features.size();i++)
    {
      features[i] = feature_model->features[i].get_value(S, Q, parsetree);
    }

  for (unsigned int i=0;i<connection_model->connections.size();i++)
    {
      string fv_present = connection_model->connections[i].present_feature.get_value(S, Q, parsetree);
      string fv_past = connection_model->connections[i].past_feature.get_value(S, Q, parsetree);
      connection_features[i] = make_pair(fv_past, fv_present);
    }
}

vector<int> State::get_compatible_connections(vector<State*> past_states)
{
  vector<int> compatible_states(connection_model->connections.size());
  for (unsigned int i=0;i<connection_model->connections.size();i++)
    {
      int compatible_state = -1;
      string present_value = connection_features[i].second;
      if(present_value!="#NONE#")
	{
	  if (connection_model->connections[i].offset==-1)
	    {
	      for(int j=past_states.size()-1;j>=0;j--)
		{
		  string past_value = past_states[j]->connection_features[i].first;
		  if (present_value==past_value)
		    {
		      compatible_state = j;
		      break;
		    } 
		}
	    }
	  else if((int)past_states.size()>=connection_model->connections[i].offset)
	    {
	      int j = past_states.size() - connection_model->connections[i].offset;
	      string past_value = past_states[j]->connection_features[i].first;
	      if (present_value==past_value)
		{
		  compatible_state = j;
		}
	    }
	}
      compatible_states[i] = compatible_state;
    }
  return compatible_states;
}

Feature::Feature()
{
  field_name = "";
  dependency_function = "";
  data_structure = "";
  offset = 0;
}

void Feature::from_string(string s)
{
  int i;
  i = s.find('(');
  s = s.substr(i+1, s.size()-i-2);
  i = s.find(',');
  field_name = s.substr(0, i);
  s = s.substr(i+1, s.size()-i-1);
  i = s.find('(');
  if (i!=-1)
    {
      dependency_function = s.substr(0,i);
      s = s.substr(i+1,s.size()-1-2);
    }
  i = s.find('[');
  data_structure = s.substr(0,i);
  s = s.substr(i+1,s.size()-i-2);
  offset = atoi(s.c_str());
}

string Feature::to_string()
{
  ostringstream result;
  result << "(" << field_name << ", " << dependency_function << "(" << data_structure << "[" << offset << "]))";
  return result.str();
}


string Feature::get_value(deque<int>& S, deque<int>& Q, ParseTree* parsetree)
{
  string feature_value = "#NONE#";

  ParseNode* node = NULL;
  if (data_structure=="Stack")
    {
      if ((int)S.size()>=offset+1)
	node = parsetree->nodes[S[S.size() - 1 - offset]];
    }
  else if (data_structure=="Input")
    {
      if ((int)Q.size()>=offset+1)
	node = parsetree->nodes[Q[offset]];
    }

  if (node==NULL)
    {
      return feature_value;
    }

  if (dependency_function!="")
    {
      ParseNode* node1 = NULL;
      if (dependency_function=="head")
	node1 = node->parent;
      else if(dependency_function=="ldep")
	{
	  if(!node->children.empty())
	    {
	      ParseNode* leftmost_child = node->children[0];
	      for (unsigned int i=1;i<node->children.size();i++)
		if(node->children[i]->id<leftmost_child->id)
		  leftmost_child = node->children[i];
	      node1 = leftmost_child;
	    }	  
	}
      else if(dependency_function=="lldep")
	{
	  if(!node->children.empty())
	    {
	      ParseNode* leftmost_child = node->children[0];
	      for (unsigned int i=1;i<node->children.size();i++)
		if(node->children[i]->id<leftmost_child->id)
		  leftmost_child = node->children[i];
	      if (leftmost_child->id<node->id)
		node1 = leftmost_child;
	    }
	}
      else if(dependency_function=="rdep")
	{
	  if(!node->children.empty())
	    {
	      ParseNode* rightmost_child = node->children[0];
	      for (unsigned int i=1;i<node->children.size();i++)
		if(node->children[i]->id>rightmost_child->id)
		  rightmost_child = node->children[i];
	      node1 = rightmost_child;
	    }
	}
      else if(dependency_function=="rrdep")
	{
	  if(!node->children.empty())
	    {
	      ParseNode* rightmost_child = node->children[0];
	      for (unsigned int i=1;i<node->children.size();i++)
		if(node->children[i]->id>rightmost_child->id)
		  rightmost_child = node->children[i];
	      if (rightmost_child->id>node->id)
		node1 = rightmost_child;
	    }
	}
      node = node1;
    }

  if (node!=NULL)
    {
      if(field_name=="DEPREL")
	{
	  if(node->parent!=NULL)
	    {
	      char buf[10];
	      sprintf(buf, "%d", node->labelid);
	      feature_value = string(buf);
	    }
	}
      else if(field_name=="FORM" || field_name=="LEMMA" || field_name=="PLEMMA")
	{
	  Word* word = (Word*)node->data;
	  feature_value = word->pos + ":-" + word->corpus_fields[field_name];
	}
      else
	{
	  Word* word = (Word*)node->data;
	  feature_value = word->corpus_fields[field_name];
	}
    }
  return feature_value;
}


FeatureModel::FeatureModel(string filename)
{
  cerr<<"reading FeatureModel from file: "<<filename.c_str()<<"...";
  try
    { 
      xmlpp::TextReader reader(filename);
      while (reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="feature")
		{
		  reader.move_to_element();
		  reader.read();
		  Feature feature;
		  feature.from_string(reader.get_value());
		  features.push_back(feature);
		}
	    }
	}
    }
  catch(const std::exception& e)
    {
      cerr << "Exception caught: " << e.what() << endl;
    }
  cerr<<"done!\n";
}

string FeatureModel::to_string()
{
  ostringstream	result;
  result << "Feature Model:\n";
  for (unsigned int i=0;i<features.size();i++)
    result << "\tFeature#" << i << " : " << features[i].to_string() << "\n";
  return result.str();  
}

Connection::Connection(Feature _present_feature, Feature _past_feature, int _offset)
{
  present_feature = _present_feature;
  past_feature = _past_feature;
  offset = _offset;
}

string Connection::to_string()
{
  ostringstream	result;
  result<< past_feature.to_string() << "-->" << present_feature.to_string() << " (offset:" << offset << ")";
  return result.str();
}


ConnectionModel::ConnectionModel(string filename)
{
  cerr<<"reading ConnectionModel from file: "<<filename.c_str()<<"...";
  try
    { 
      xmlpp::TextReader reader(filename);
      while (reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="connection")
		{
		  reader.move_to_element();
		  Feature present_feature, past_feature;
		  int offset = -1;
		  bool end_connection = false;
		  while (!end_connection)
		    {
		      reader.read();
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::EndElement)
			{
			  if(reader.get_name()=="connection")
			    end_connection = true;
			}
		      else if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
			{
			  if(reader.get_name()=="feature")
			    {
			      reader.move_to_attribute("location");
			      if (reader.get_value()=="present")
				{
				  reader.move_to_element();
				  reader.read();
				  present_feature.from_string(reader.get_value());
				}
			      else if (reader.get_value()=="past")
				{
				  reader.move_to_element();
				  reader.read();
				  past_feature.from_string(reader.get_value());
				}
			    }
			  else if(reader.get_name()=="offset")
			    {
			      reader.move_to_element();
			      reader.read();
			      offset = atoi(reader.get_value().c_str());
			    }
			}
		    }

		  Connection connection(present_feature, past_feature, offset);
		  connections.push_back(connection);
		}
	    }
	}
    }
  catch(const std::exception& e)
    {
      cout << "Exception caught: " << e.what() << endl;
    }  
  cerr<<"done!\n";
}

string ConnectionModel::to_string()
{
  ostringstream	result;
  result << "ConnectionModel:\n";
  for (unsigned int i=0;i<connections.size();i++)
    result << "\tConnection#" << i << " : " << connections[i].to_string() << "\n";
  return result.str();
}
