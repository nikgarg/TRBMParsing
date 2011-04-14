#ifndef __EVALUATOR_H
#define __EVALUATOR_H

#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>
#include "parsetree.h"

using namespace std;

class Evaluator
{
  int running_sum_AS_per_token;
  int running_sum_LAS_per_token;
  double running_sum_AS_per_sentence;
  double running_sum_LAS_per_sentence;
  
 public:
  int n_sentences;
  int n_no_result;
  int n_correct_sentences;
  int n_Lcorrect_sentences;
  int n_tokens;
  int n_tokens_ignored;
  double AS_per_token;
  double LAS_per_token;
  double AS_per_sentence;
  double LAS_per_sentence;
  unordered_set<int> ignore_tokens_with_labelid;

  Evaluator(unordered_set<int> _ignore_tokens_with_label);
  ~Evaluator();
  void update(ParseTree* gold_parse_tree, ParseTree* predicted_parse_tree);
  string get_results();
};

#endif
