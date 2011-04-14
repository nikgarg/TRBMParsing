#include "parsetree.h"

ParseNode::ParseNode()
{
  id = -1;
  data = NULL;
  parent = NULL;
  labelid = -1;
  desc = "";
}

ParseNode::~ParseNode()
{

}

string ParseNode::to_string()
{
  ostringstream result;
  result << desc;
  if(parent!=NULL)
    result << "(" << parent->id << "," << labelid << ")";
  return result.str();
  //return ((Word*)data)->to_string();
  /*
  ostringstream	result;
  result << (*data)["ORIGINAL-FORM"] << "[" << id << "]";
  if(parent!=NULL)
    result << "(" << parent->id << "," << label << ")";
  return result.str();
  */
}

vector<ParseNode*> ParseNode::get_descendants()
{
  vector<ParseNode*> descendants;
  for(unsigned int i=0;i<children.size();i++)
    {
      descendants.push_back(children[i]);
      vector<ParseNode*> D = children[i]->get_descendants();
      descendants.insert(descendants.end(), D.begin(), D.end());
    }
  return descendants;
}

ParseTree::ParseTree()
{

}

ParseTree::~ParseTree()
{
  map<int, ParseNode*>::iterator it;
  for (it=nodes.begin(); it!=nodes.end(); it++)
    {
      ParseNode *node = it->second;
      delete node;
    }  
}

//void ParseTree::add_node(int id, unordered_map<string,string> *data)
void ParseTree::add_node(int id, void *data, string desc)
{
  ParseNode* node = new ParseNode();
  node->id = id;
  node->data = data;
  node->desc = desc;
  nodes[id] = node;
}

void ParseTree::add_edge(int source, int destination, int labelid)
{
  ParseNode* S = nodes[source];
  ParseNode* D = nodes[destination];
  D->parent = S;
  D->labelid = labelid;
  S->children.push_back(D);
}

ParseTree* ParseTree::clone()
{
  ParseTree* cloneTree = new ParseTree();
  map<int, ParseNode*>::iterator it;
  for (it=nodes.begin(); it!=nodes.end(); it++)
    {
      ParseNode *node = it->second;
      ParseNode *cloneNode = new ParseNode();
      cloneNode->id = node->id;
      cloneNode->data = node->data;
      cloneNode->desc = node->desc;
      cloneTree->nodes[it->first] = cloneNode;
      cloneNode->children.reserve(node->children.size());
    }
  for (it=nodes.begin(); it!=nodes.end(); it++)
    {
      ParseNode *node = it->second;
      if(node->parent!=NULL)
	cloneTree->add_edge(node->parent->id, node->id, node->labelid);
    }
  return cloneTree;
}

bool ParseTree::is_well_formed()
{
  if (is_connected() && is_projective())
    return true;
  return false;
}

bool ParseTree::is_connected()
{
  int count = 0;
  map<int, ParseNode*>::iterator it;
  for (it=nodes.begin(); it!=nodes.end(); it++)
    if (it->first!=-1 && it->second->parent==NULL)
      count++;
  if(count>1)
    return false;
  return true;
}

bool ParseTree::is_projective()
{
  map<int, ParseNode*>::iterator it;
  for (it=nodes.begin(); it!=nodes.end(); it++)
    {
      if(it->first!=-1)
	{
	  ParseNode* node = it->second;
	  vector<ParseNode*> descendants = node->get_descendants();
	  descendants.push_back(node);
	  int leftmost_descendant = descendants[0]->id;
	  int rightmost_descendant = descendants[0]->id;
	  for(unsigned int i=0;i<descendants.size();i++)
	    if(descendants[i]->id<leftmost_descendant)
	      leftmost_descendant = descendants[i]->id;
	    else if(descendants[i]->id>rightmost_descendant)
	      rightmost_descendant = descendants[i]->id;
	  if (rightmost_descendant-leftmost_descendant+1!=(int)descendants.size())
	    return false;	  
	}
    }
  return true;
}

vector< pair<ParseNode*,int> > ParseTree::preorder_traversal(ParseNode* r, int depth)
{
  vector< pair<ParseNode*,int> > L;
  L.push_back(make_pair(r,depth));
  for(unsigned int i=0;i<r->children.size();i++)
    {
      vector< pair<ParseNode*,int> > L1 = preorder_traversal(r->children[i], depth+1);
      L.insert(L.end(), L1.begin(), L1.end());
    }
  return L;
}

string ParseTree::to_string()
{
  string result;
  map<int, ParseNode*>::iterator it;
  for (it=nodes.begin(); it!=nodes.end(); it++)
    {
      if(it->first==-1)
	continue;
      ParseNode* node = it->second;
      if (node->parent==NULL)
	{
	  vector< pair<ParseNode*,int> > ordered_nodes = preorder_traversal(node, 0);
	  vector<int> depth_active(nodes.size()+1, 0);
	  for(unsigned int i=0;i<ordered_nodes.size();i++)
	    {
	      int depth = ordered_nodes[i].second;
	      if (depth>0)
		{
		  for(int j=0;j<depth-1;j++)
		    if (depth_active[j]>0)
		      result += "|   ";
		    else
		      result += "    ";
		  result += "|___";
		  depth_active[depth-1] -= 1;
		}
	      depth_active[depth] += ordered_nodes[i].first->children.size();
	      result += ordered_nodes[i].first->to_string() + '\n';
	    }
	}
    }
  return result;
}
