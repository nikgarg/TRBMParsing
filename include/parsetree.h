#ifndef __PARSETREE_H
#define __PARSETREE_H

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;


class ParseNode
{
 public:
  int id;
  void *data;
  ParseNode* parent;
  //string label;
  int labelid;
  vector<ParseNode*> children;
  string desc;

  ParseNode();
  ~ParseNode();
  vector<ParseNode*> get_descendants();
  string to_string();
};


class ParseTree
{
  bool is_connected();
  bool is_projective();
  vector< pair<ParseNode*,int> > preorder_traversal(ParseNode* r, int depth);

 public:
  map<int, ParseNode*> nodes;
  //int root;
  ParseTree();
  ~ParseTree();
  void add_node(int id, void *data, string desc);
  void add_edge(int source, int destination, int labelid);
  ParseTree* clone();
  bool is_well_formed();
  string to_string();
};

#endif
