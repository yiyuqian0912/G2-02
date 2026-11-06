#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <valgrind/callgrind.h>

template<class T>void read(std::ifstream&in,T&value){in.read(reinterpret_cast<char*>(&value),sizeof(T));if(!in)throw std::runtime_error("Truncated profile input");}
int main(int argc,char**argv){
  if(argc!=3)return 2;std::ifstream in(argv[1],std::ios::binary);std::uint32_t m,n;read(in,m);read(in,n);
  if(!m||m>1024||!n||n>4096)return 3;b2::Inputs input;
  for(unsigned i=0;i<m;i++){b2::P center;b2::Matrix matrix;double weight;for(auto&v:center)read(in,v);for(auto&v:matrix)read(in,v);read(in,weight);input.centers.push_back(center);input.matrices.push_back(matrix);input.weights.push_back(weight);}
  double symbolic;read(in,input.threshold);read(in,symbolic);input.exponential=symbolic!=0;b2::P source;for(auto&v:source)read(in,v);
  std::vector<b2::P> points(n);for(auto&p:points)for(auto&v:p)read(in,v);
  b2::Options options;options.query_seconds=1000; // profiling overhead is not a physical timeout
  b2::Stats stats;int hit=0,miss=0,unknown=0;
  auto run=[&](auto&engine){
    CALLGRIND_START_INSTRUMENTATION;CALLGRIND_ZERO_STATS;
    for(auto&p:points){auto d=engine.ray(source,p,options,stats);if(d.status==1)hit++;else if(d.status==0)miss++;else unknown++;}
    CALLGRIND_DUMP_STATS;CALLGRIND_STOP_INSTRUMENTATION;
  };
  if(std::string(argv[2])=="reference"){b2::Independent engine(input);run(engine);}else{b2::Primary engine(input);run(engine);}
  std::cout<<"HIT "<<hit<<" MISS "<<miss<<" UNKNOWN "<<unknown<<"\n";
  for(auto&[name,value]:stats.n)std::cout<<name<<" "<<value<<"\n";
  return unknown?4:0;
}
