#ifndef __DATA_H
#define __DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <string.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <libxml++/libxml++.h>
#include <libxml++/parsers/textreader.h>
#include "sentence.h"

using namespace std;

class Data
{
  unordered_map<string, ifstream*> fd;
  vector<string> columns;
  unordered_map<string, int> columnid;
  unordered_map<string, int> cutoff_freq;
  unordered_map<string, unordered_map<string, unordered_set<string> > > valid_values;

 public:
  vector<string> id2label;
  unordered_map<string,int> label2id;
  vector<string> id2pos;
  unordered_map<string,int> pos2id;
  vector< vector<string> > id2lex;
  vector< unordered_map<string,int> > lex2id;

  Data();
  ~Data();
  void build_lexicon(string filename);
  void save_lexicon(string filename);
  void read_lexicon(string filename);
  void open_file(string desc, string filename);
  void close_file(string desc);
  void reset(string desc);
  Sentence* read_sentence(string desc);
  string write_output(Sentence* sentence);
};

#endif
