#include "main.h"


void print_usage(char* command)
{
  cerr<<"Usage: "<<command<<" <config_file> <train/test/train,test> <(train new?)yes <FF/TRBM>/no>\n";
  cerr<<"Example 1: "<<command<<" config/config train,test yes TRBM\n"
      <<"Example 2: "<<command<<" config/config test no\n";
}

string trim(string s)
{
  unsigned int p = s.find_first_not_of(" \t\n");
  if(p==string::npos)
    return "";
  string result = s.substr(p);
  p = result.find_last_not_of(" \t\n");
  result.erase(p+1);
  return result;
}

int main(int argc, char* argv[])
{
  if(argc<4)
    {
      print_usage(argv[0]);
      return 1;
    }

  char* config_file = argv[1];
  bool train = true;
  bool test = true;
  if(!strcmp(argv[2], "train"))
    test = false;
  else if(!strcmp(argv[2], "test"))
    train = false;
  else if(!strcmp(argv[2], "train,test"))
    ;
  else
    {
      print_usage(argv[0]);
      return 1;
    }

  bool train_new = false;
  string model_type = "";
  if(!strcmp(argv[3],"yes"))
    {
      train_new = true;
      if(!train)
	{
	  cerr<<"Incompatible options! You need to train the model first to use it!\n";
	  print_usage(argv[0]);
	  return 1;
	}
      if(argc<5)
	{
	  cerr<<"Please specify model_type = FF/TRBM\n";
	  print_usage(argv[0]);
	  return 1;	  
	}
      if(!strcmp(argv[4],"FF"))
	model_type = "FF";
      else if(!strcmp(argv[4],"TRBM"))
	model_type = "TRBM";
      else
	{
	  cerr<<"Incorrect model_type. Possible options are: FF/TRBM\n";
	  print_usage(argv[0]);
	  return 1;	  
	}
    }
  else if(!strcmp(argv[3],"no"))
    ;
  else
    {
      print_usage(argv[0]);
      return 1;
    }

  cerr<<"Reading configuration from file: "<<config_file<<"...";
  unordered_map<string,string> params;
  ifstream fconfig(config_file);
  string line;
  while(getline(fconfig, line))
    {
      unsigned p = line.find('#');
      line = trim(line.substr(0,p));
      p = line.find('\t');
      if(p!=string::npos)
	{
	  string key = trim(line.substr(0,p));
	  string value = trim(line.substr(p+1));
	  if(!key.empty())
	  params[key] = value;
	}
    }
  fconfig.close();
  cerr<<"done!\n";

  Parser parser(params, train_new, model_type);

  if(train)
    parser.train();

  if(test)
    {
      parser.test(); 
      string results = parser.get_results();
      cerr<<"Results:\n"<<results<<"\n";
      cout<<"Results:\n"<<results<<"\n";
    }
  //parser.print_word_representations(params["dev-set"], params["word-repr"]);
  
  return 0;
}
