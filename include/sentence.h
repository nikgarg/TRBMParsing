#ifndef __SENTENCE_H
#define __SENTENCE_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include "parsetree.h"

using namespace std;

class Word
{
 public:
  int id;
  string lex;
  string pos;

  int lexid;
  int posid;

  unordered_map<string,string> corpus_fields;
  string desc;
  string input_line;

  Word();
  ~Word();
};

class Sentence
{
 public:
  vector<Word*> words;
  ParseTree *gold_parse_tree;
  ParseTree *predicted_parse_tree;

  Sentence();
  ~Sentence();
};

#endif
