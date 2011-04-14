#include "evaluator.h"

Evaluator::Evaluator(unordered_set<int> _ignore_tokens_with_labelid)
{
  n_sentences = 0;
  n_no_result = 0;
  n_correct_sentences = 0;
  n_Lcorrect_sentences = 0;
  n_tokens = 0;
  AS_per_token = 0;
  LAS_per_token = 0;
  AS_per_sentence = 0;
  LAS_per_sentence = 0;
  n_tokens_ignored = 0;
  running_sum_AS_per_token = 0;
  running_sum_LAS_per_token = 0;
  running_sum_AS_per_sentence = 0;
  running_sum_LAS_per_sentence = 0;
  ignore_tokens_with_labelid = _ignore_tokens_with_labelid;
}

Evaluator::~Evaluator()
{
}

void Evaluator::update(ParseTree* gold_parse_tree, ParseTree* predicted_parse_tree)
{
  if (gold_parse_tree==NULL)
    return;

  n_sentences += 1;
  int _n_nodes = 0;
  int _n_nodes_total = 0;
  for(map<int,ParseNode*>::iterator it=gold_parse_tree->nodes.begin(); it!=gold_parse_tree->nodes.end(); it++)
    {
      if(it->first!=-1)
	{
	  _n_nodes_total += 1;
	  if(ignore_tokens_with_labelid.find(it->second->labelid)==ignore_tokens_with_labelid.end())
	    _n_nodes += 1;
	}
    }

  n_tokens += _n_nodes;
  n_tokens_ignored += _n_nodes_total - _n_nodes;

  if (predicted_parse_tree==NULL)
    {
      n_no_result += 1;
      return;
    }

  int relevant = 0;
  int Lrelevant = 0;
  
  for(map<int,ParseNode*>::iterator it=gold_parse_tree->nodes.begin(); it!=gold_parse_tree->nodes.end(); it++)
    {
      int id = it->first;
      if(id==-1)
	continue;
      ParseNode* gnode = it->second;
      if(ignore_tokens_with_labelid.find(gnode->labelid)==ignore_tokens_with_labelid.end())
	{
	  map<int,ParseNode*>::iterator it1 = predicted_parse_tree->nodes.find(id);
	  if(it1!=predicted_parse_tree->nodes.end())
	    {
	      ParseNode* pnode = it1->second;
	      if(gnode->parent==NULL)
		{
		  if(pnode->parent==NULL)
		    {
		      relevant += 1;
		      Lrelevant += 1;
		    }
		}
	      else if(pnode->parent!=NULL && gnode->parent->id==pnode->parent->id)
		{
		  relevant += 1;
		  if (gnode->labelid==pnode->labelid)
		    {
		      Lrelevant += 1;
		    }
		}
	    }
	}
    }

  if (relevant==_n_nodes)
    {
      n_correct_sentences += 1;
      if (Lrelevant==_n_nodes)
	n_Lcorrect_sentences += 1;
    }

  running_sum_AS_per_token += relevant;
  running_sum_LAS_per_token += Lrelevant;
  running_sum_AS_per_sentence += ((float)relevant)/_n_nodes;
  running_sum_LAS_per_sentence += ((float)Lrelevant)/_n_nodes;
}

string Evaluator::get_results()
{
  if (n_tokens>0)
    {
      AS_per_token = (100.0*running_sum_AS_per_token)/((float)n_tokens);
      LAS_per_token = (100.0*running_sum_LAS_per_token)/((float)n_tokens);
    }
  else
    {
      AS_per_token = 0;
      LAS_per_token = 0;
    }

  float percentage_correct_sentences=0, percentage_Lcorrect_sentences=0;
  if (n_sentences>0)
    {
      percentage_correct_sentences = (100.0*n_correct_sentences)/((float)n_sentences);
      percentage_Lcorrect_sentences = (100.0*n_Lcorrect_sentences)/((float)n_sentences);
      AS_per_sentence = (100.0*running_sum_AS_per_sentence)/((float)n_sentences);
      LAS_per_sentence = (100.0*running_sum_LAS_per_sentence)/((float)n_sentences);
    }
  else
    {
      percentage_correct_sentences = 0;
      percentage_Lcorrect_sentences = 0;
      AS_per_sentence = 0;
      LAS_per_sentence = 0;
      
    }

  ostringstream result;
  result<<"# Sentences:                  "<<n_sentences<<"\n"
	<<"# No result:                  "<<n_no_result<<"\n"
	<<"# Correct sentences:          "<<n_correct_sentences<<" ("<<percentage_correct_sentences<<"%)\n"
	<<"# Labeled Correct sentences:  "<<n_Lcorrect_sentences<<" ("<<percentage_Lcorrect_sentences<<"%)\n"
	<<"AS per sentence:              "<<AS_per_sentence<<"%\n"
	<<"Labeled AS per sentence:      "<<LAS_per_sentence<<"%\n"
	<<"# Tokens:                     "<<n_tokens<<"\n"
	<<"# Tokens ignored (out of "<<(n_tokens+n_tokens_ignored)<<"):"<<n_tokens_ignored<<" ("<<((100.0*n_tokens_ignored)/((float)n_tokens + n_tokens_ignored))<<"%)\n"
	<<"AS per token:                 "<<AS_per_token<<"%\n"
	<<"Labeled AS per token:         "<<LAS_per_token<<"%\n";
  return result.str();
}


