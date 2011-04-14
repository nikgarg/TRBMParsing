#include "data.h"

Data::Data()
{
  columns.reserve(14);
  columns.push_back("ID");
  columns.push_back("FORM");
  columns.push_back("LEMMA");
  columns.push_back("PLEMMA");
  columns.push_back("POS");
  columns.push_back("PPOS");
  columns.push_back("FEAT");
  columns.push_back("PFEAT");
  columns.push_back("HEAD");
  columns.push_back("PHEAD");
  columns.push_back("DEPREL");
  columns.push_back("PDEPREL");
  columns.push_back("FILLPRED");
  columns.push_back("PRED");
  char buf[10];
  for(unsigned int i=1;i<=10;i++)
    {
      sprintf(buf, "APRED%d",i);
      columns.push_back(string(buf));
    }
  for (unsigned int i=0;i<columns.size();i++)
    columnid[columns[i]] = i;
  cutoff_freq["FORM"] = 20;  //in combination with POS
  cutoff_freq["PLEMMA"] = 20;
}

Data::~Data()
{
  for(unordered_map<string, ifstream*>::iterator it=fd.begin(); it!=fd.end(); it++)
    delete it->second;
}

void Data::build_lexicon(string filename)
{
  cerr<<"building lexicon from file: "<<filename<<"...";
  ifstream f;
  f.open(filename.c_str());
  unordered_map<string, unordered_map<string, unordered_map<string,int> > > freq_pos_field;
  char line[1024];
  for(unordered_map<string, int>::iterator it=cutoff_freq.begin(); it!=cutoff_freq.end(); it++)
    {
      string col = it->first;
      unordered_map<string, unordered_map<string,int> > pos_values;
      freq_pos_field[col] = pos_values;
    }

  while (f.getline(line, 1024))
    {
      if(line[0]=='\0')
	continue;
      vector<char*> fields(columns.size());

      char *token;
      token = strtok(line, "\t");
      unsigned int i = 0;
      while(token!=NULL && i<columns.size())
	{
	  fields[i] = token;
	  token = strtok(NULL, "\t");
	  i++;
	}

      string label(fields[columnid["DEPREL"]]);
      if(label2id.find(label)==label2id.end())
	{
	  label2id[label] = id2label.size();
	  id2label.push_back(label);
	}

      string pos(fields[columnid["PPOS"]]);

      for(unordered_map<string, int>::iterator it=cutoff_freq.begin(); it!=cutoff_freq.end(); it++)
	{
	  string col = it->first;
	  string field_value = fields[columnid[col]];
	  if(freq_pos_field[col].find(pos)==freq_pos_field[col].end())
	    {
	      unordered_map<string,int> values;
	      freq_pos_field[col][pos] = values;
	    }
	  if(freq_pos_field[col][pos].find(field_value)!=freq_pos_field[col][pos].end())
	    freq_pos_field[col][pos][field_value] += 1;
	  else
	    freq_pos_field[col][pos][field_value] = 1;
	}
    }

  for(unordered_map<string, unordered_map<string, unordered_map<string,int> > >::iterator it1=freq_pos_field.begin(); it1!=freq_pos_field.end(); it1++)
    {
      string field = it1->first;
      for(unordered_map<string, unordered_map<string,int> >::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++)
	{
	  string pos = it2->first;
	  for(unordered_map<string,int>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++)
	    {
	      string field_value = it3->first;
	      int freq = it3->second;
	      if(freq>=cutoff_freq[field])
		valid_values[field][pos].insert(field_value);
	    }
	  valid_values[field][pos].insert("#"+pos+"-UNKNOWN#");
	}
    }
  valid_values["FORM"]["#END_POS#"].insert("#END_WORD#");
  valid_values["PLEMMA"]["#END_POS#"].insert("#END_WORD#");
  
  for(unordered_map<string, unordered_set<string> >::iterator it=valid_values["FORM"].begin(); it!=valid_values["FORM"].end(); it++)
    {
      pos2id[it->first] = id2pos.size();
      id2pos.push_back(it->first);
      vector<string> _id2lex;
      unordered_map<string,int> _lex2id;
      for(unordered_set<string>::iterator it_lex = it->second.begin(); it_lex!=it->second.end(); it_lex++)
	{
	  _lex2id[*it_lex] = _id2lex.size();
	  _id2lex.push_back(*it_lex);
	}
      lex2id.push_back(_lex2id);
      id2lex.push_back(_id2lex);
    }

  label2id["#END_DEPREL#"] = id2label.size();
  id2label.push_back("#END_DEPREL#");
  
  f.close();
  cerr<<"done!\n";
}


void Data::reset(string desc)
{
  fd[desc]->clear();
  fd[desc]->seekg(0, ios::beg);
}

void Data::read_lexicon(string filename)
{
  cerr<<"reading lexicon from file: "<<filename<<"...";
  ifstream f;
  f.open(filename.c_str());
  try
    {
      xmlpp::TextReader reader(filename);
      while(reader.read())
	{
	  if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
	    {
	      if(reader.get_name()=="LABELS")
		{
		  reader.move_to_attribute("n");
		  int n = atoi(reader.get_value().c_str());
		  for(int i=0;i<n;i++)
		    id2label.push_back("");
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="LABELS")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="label")
			{
			  reader.move_to_attribute("id");
			  int labelid = atoi(reader.get_value().c_str());
			  reader.move_to_element();
			  reader.read();
			  string l = reader.get_value();
			  id2label[labelid] = l;
			  label2id[l] = labelid;
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="POS_LEX")
		{
		  reader.move_to_attribute("n");
		  int n = atoi(reader.get_value().c_str());
		  for(int i=0;i<n;i++)
		    {
		      id2pos.push_back("");
		      vector<string> v;
		      id2lex.push_back(v);
		      unordered_map<string,int> m;
		      lex2id.push_back(m);
		    }
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="POS_LEX")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="pos")
			{
			  reader.move_to_attribute("id");
			  int posid = atoi(reader.get_value().c_str());
			  reader.move_to_attribute("n");
			  int n1 = atoi(reader.get_value().c_str());
			  for(int j=0;j<n1;j++)
			    id2lex[posid].push_back("");
			  reader.move_to_element();
			  reader.read();
			  string pos = "";
			  while(reader.get_name()!="pos")
			    {
			      if(reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
				{
				  if(reader.get_name()=="pos_name")
				    {
				      reader.move_to_element();
				      reader.read();
				      pos = reader.get_value();
				      id2pos[posid] = pos;
				      pos2id[pos] = posid;
				    }
				  else if(reader.get_name()=="lex")
				    {
				      reader.move_to_attribute("id");
				      int lexid = atoi(reader.get_value().c_str());
				      reader.move_to_element();
				      reader.read();
				      string lex = reader.get_value();
				      id2lex[posid][lexid] = lex;
				      lex2id[posid][lex] = lexid;
				    }
				}
			      reader.read();
			    }
			}
		      reader.read();
		    }
		}
	      else if(reader.get_name()=="FIELD_POS_VALUE")
		{
		  reader.move_to_attribute("n");
		  //int n = atoi(reader.get_value().c_str());
		  reader.move_to_element();
		  reader.read();
		  while(reader.get_name()!="FIELD_POS_VALUE")
		    {
		      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="field")
			{
			  reader.move_to_attribute("name");
			  string field = reader.get_value();
			  reader.move_to_attribute("cutoff");
			  int c = atoi(reader.get_value().c_str());
			  cutoff_freq[field] = c;
			  reader.move_to_element();
			  reader.read();
			  while(reader.get_name()!="field")
			    {
			      if (reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element && reader.get_name()=="pos")
				{
				  reader.move_to_attribute("n");
				  //int n1 = atoi(reader.get_value().c_str());
				  reader.move_to_element();
				  reader.read();
				  string pos = "";
				  while(reader.get_name()!="pos")
				    {
				      if(reader.get_node_type()==xmlpp::TextReader::xmlNodeType::Element)
					{
					  if(reader.get_name()=="pos_name")
					    {
					      reader.move_to_element();
					      reader.read();
					      pos = reader.get_value();
					    }
					  else if(reader.get_name()=="value")
					    {
					      reader.move_to_element();
					      reader.read();
					      string value = reader.get_value();
					      valid_values[field][pos].insert(value);
					    }
					}
				      reader.read();
				    }
				}
			      reader.read();
			    }
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

  f.close();
  cerr<<"done!\n";
}

void Data::save_lexicon(string filename)
{
  cerr<<"saving lexicon in file: "<<filename<<"...";
  ofstream f;
  f.open(filename.c_str());
  f<<"<Lexicon>\n";

  f<<"\t<LABELS n=\""<<id2label.size()<<"\">\n";
  for(unsigned i=0;i<id2label.size();i++)
    f<<"\t\t<label id=\""<<i<<"\"><![CDATA["<<id2label[i]<<"]]></label>\n";
  f<<"\t</LABELS>\n";

  f<<"\t<POS_LEX n=\""<<id2pos.size()<<"\">\n";
  for(unsigned i=0;i<id2pos.size();i++)
    {
      f<<"\t\t<pos id =\""<<i<<"\" n=\""<<id2lex[i].size()<<"\">\n";
      f<<"\t\t\t<pos_name><![CDATA["<<id2pos[i]<<"]]></pos_name>\n";
      for(unsigned j=0;j<id2lex[i].size();j++)
	f<<"\t\t\t<lex id=\""<<j<<"\"><![CDATA["<<id2lex[i][j]<<"]]></lex>\n";
      f<<"\t\t</pos>\n";
    }
  f<<"\t</POS_LEX>\n";

  f<<"\t<FIELD_POS_VALUE n=\""<< valid_values.size() <<"\">\n";
  for(unordered_map<string, unordered_map<string, unordered_set<string> > >::iterator it1=valid_values.begin(); it1!=valid_values.end(); it1++)
    {
      f<<"\t\t<field name=\""<< (it1->first) <<"\" cutoff=\""<< cutoff_freq[it1->first] <<"\" >\n";
      for(unordered_map<string, unordered_set<string> >::iterator it2=it1->second.begin(); it2!=it1->second.end(); it2++)
	{
	  f<<"\t\t\t<pos id=\""<<pos2id[it2->first]<<"\" n=\""<< it2->second.size() <<"\">\n";
	  f<<"\t\t\t\t<pos_name><![CDATA["<<(it2->first)<<"]]></pos_name>\n";
	  for(unordered_set<string>::iterator it3=it2->second.begin(); it3!=it2->second.end(); it3++)
	    f<<"\t\t\t\t<value><![CDATA["<<(*it3)<<"]]></value>\n";
	  f<<"\t\t\t</pos>\n";
	}
      f<<"\t\t</field>\n";
    }
  f<<"\t</FIELD_POS_VALUE>\n";

  f<<"</Lexicon>\n";

  f.close();
  cerr<<"done!\n";
}


void Data::open_file(string desc, string filename)
{
  ifstream* f = new ifstream;
  f->open(filename.c_str(), ios::in);
  fd[desc] = f;
}

void Data::close_file(string desc)
{
  fd[desc]->close();
  fd[desc]->clear();
}

Sentence* Data::read_sentence(string desc)
{
  char line[1024];
  ifstream* f = fd[desc];
  f->getline(line, 1024);
  if (line[0]=='\0')
    return NULL;

  unsigned int i;

  Sentence* sentence = new Sentence();
  while (line[0]!='\0')
    {
      Word *word = new Word();
      char *token;
      word->input_line = string(line);
      token = strtok(line, "\t");
      i = 0;
      while(token!=NULL && i<columns.size())
	{
	  word->corpus_fields[columns[i]] = string(token);
	  token = strtok(NULL, "\t");
	  i++;
	}
      string pos = word->corpus_fields["PPOS"];
      for(unordered_map<string, int>::iterator it=cutoff_freq.begin(); it!=cutoff_freq.end(); it++)
	{
	  string col = it->first;
	  string col_value = word->corpus_fields[col];
	  word->corpus_fields["ORIGINAL-"+col] = col_value;
	  if (valid_values[col][pos].find(col_value)==valid_values[col][pos].end())
	    word->corpus_fields[col] = "#"+pos+"-UNKNOWN#";
	    
	}

      word->id = atoi(word->corpus_fields["ID"].c_str());
      word->lex = word->corpus_fields["FORM"];
      word->pos = pos;
      word->posid = pos2id[pos];
      word->lexid = lex2id[word->posid][word->corpus_fields["FORM"]];

      ostringstream desc;
      desc << word->corpus_fields["ORIGINAL-FORM"] << "[" << word->id << "]";
      word->desc = desc.str();
      sentence->words.push_back(word);
      
      f->getline(line, 1024);
    }

  
  Word *end_word = new Word();
  end_word->id = -1;
  end_word->lex = "#END_WORD#";
  end_word->pos = "#END_POS";
  end_word->posid = pos2id["#END_POS"];
  end_word->lexid = lex2id[end_word->posid]["#END_WORD#"];
  end_word->desc = "#END_WORD#[-1]";
  end_word->corpus_fields["ID"] = "-1";
  end_word->corpus_fields["FORM"] = "#END_WORD#";
  end_word->corpus_fields["LEMMA"] = "#END_WORD#";
  end_word->corpus_fields["PLEMMA"] = "#END_WORD#";
  end_word->corpus_fields["POS"] = "#END_POS#";
  end_word->corpus_fields["PPOS"] = "#END_POS#";
  end_word->corpus_fields["DEPREL"] = "#END_DEPREL#";
  end_word->corpus_fields["HEAD"] = "0";
  sentence->words.push_back(end_word);
  

  sentence->gold_parse_tree = new ParseTree();
  for(i=0;i<sentence->words.size();i++)
    sentence->gold_parse_tree->add_node(sentence->words[i]->id, sentence->words[i], sentence->words[i]->desc);

  for(i=0;i<sentence->words.size();i++)
    if(sentence->words[i]->corpus_fields["HEAD"]!="" && sentence->words[i]->corpus_fields["HEAD"]!="0")
      {
	string label = sentence->words[i]->corpus_fields["DEPREL"];
	int labelid = label2id[sentence->words[i]->corpus_fields["DEPREL"]];
	int head = atoi(sentence->words[i]->corpus_fields["HEAD"].c_str());
	sentence->gold_parse_tree->add_edge(head, sentence->words[i]->id, labelid);	  
      }
  
  return sentence;
}


string Data::write_output(Sentence* sentence)
{
  ostringstream output;

  for(unsigned int i=0;i<sentence->words.size()-1;i++)
    {
      istringstream iss(sentence->words[i]->input_line, istringstream::in);
      string tok;
      iss>>tok;
      output<<tok; //assume that 1st token is always ID
      unsigned int j=1;
      while(iss>>tok)
	{
	  if(j<columns.size())
	    {
	      if(columns[j]=="HEAD")
		{
		  if(sentence->predicted_parse_tree!=NULL)
		    {
		      int id = sentence->words[i]->id;
		      ParseNode* node = sentence->predicted_parse_tree->nodes.find(id)->second;
		      if(node->parent==NULL)
			output<<"\t0";
		      else
			output<<"\t"<<node->parent->id;
		    }
		  else
		    output<<"\t_";
		}
	      else if(columns[j]=="DEPREL")
		{
		  if(sentence->predicted_parse_tree!=NULL)
		    {
		      int id = sentence->words[i]->id;
		      ParseNode* node = sentence->predicted_parse_tree->nodes.find(id)->second;
		      if(node->parent==NULL)
			output<<"\tROOT";
		      else
			output<<"\t"<<id2label[node->labelid];
		    }
		  else
		    output<<"\t_";
		}
	      else
		{
		  output<<"\t"<<tok;
		}
	    }
	  else
	    {
	      output<<"\t"<<tok;
	    }
	  j += 1;
	}
      output<<"\n";
    }
  output<<"\n";
  return output.str();
}
