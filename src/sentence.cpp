#include "sentence.h"

Word::Word()
{
  id = -1;
  lex = "";
  pos = "";
  //label = "";
  desc = "";
  lexid = -1;
  posid = -1;
}

Word::~Word()
{
}

Sentence::Sentence()
{
  gold_parse_tree = NULL;
  predicted_parse_tree = NULL;
}

Sentence::~Sentence()
{
  if(gold_parse_tree!=NULL)
    delete gold_parse_tree;
  if(predicted_parse_tree!=NULL)
    delete predicted_parse_tree;
  for(unsigned int i=0;i<words.size();i++)
    delete words[i];
}
