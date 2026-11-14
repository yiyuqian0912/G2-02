#include "functional.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace r3 {
namespace {
struct I {double lo,hi;explicit I(double v):lo(v),hi(v){}I(double a,double b):lo(a),hi(b){}};
double down(double x){return std::nextafter(x,-INFINITY);}
double up(double x){return std::nextafter(x,INFINITY);}
I whole(){return {-INFINITY,INFINITY};}
I add(I a,I b){double l=a.lo+b.lo,h=a.hi+b.hi;return std::isnan(l)||std::isnan(h)?whole():I(down(l),up(h));}
I sub(I a,I b){double l=a.lo-b.hi,h=a.hi-b.lo;return std::isnan(l)||std::isnan(h)?whole():I(down(l),up(h));}
I mul(I a,I b){double q[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};for(double v:q)if(std::isnan(v))return whole();return {down(*std::min_element(q,q+4)),up(*std::max_element(q,q+4))};}
I div(I a,I b){if(b.lo<=0&&b.hi>=0)return whole();return mul(a,I(down(1/b.hi),up(1/b.lo)));}
void validate(const Point&p){for(double v:p)if(!std::isfinite(v)||v<0||v>1)throw std::invalid_argument("Point outside normalized world");}
struct Node {std::int32_t index;Point lo,hi;};
Node child(const Node& p,std::int32_t index,int k){
 Node c{index,p.lo,p.hi};for(int a=0;a<3;++a){double mid=(p.lo[a]+p.hi[a])/2;if(k&(1<<a))c.lo[a]=mid;else c.hi[a]=mid;}return c;
}
// Interval slab clipping. MISS is a proof of disjointness; HIT is a proof
// that an exact closed segment meets the closed certified-solid box. Parallel
// or nearly parallel uncertainty cannot silently become either decision.
int clip(const Point&s,const Point&q,const Node&box){
 I enter(0),exit(1);bool certain=true;
 for(int a=0;a<3;++a){
  if(std::max(s[a],q[a])<box.lo[a]||std::min(s[a],q[a])>box.hi[a])return 0;
  if(s[a]==q[a]){if(s[a]<box.lo[a]||s[a]>box.hi[a])return 0;continue;}
  I d=sub(I(q[a]),I(s[a]));if(d.lo<=0&&d.hi>=0){certain=false;continue;}
  I e=div(sub(I(box.lo[a]),I(s[a])),d),f=div(sub(I(box.hi[a]),I(s[a])),d);if(d.hi<0)std::swap(e,f);
  enter={std::max(enter.lo,e.lo),std::max(enter.hi,e.hi)};exit={std::min(exit.lo,f.lo),std::min(exit.hi,f.hi)};
  if(enter.lo>exit.hi)return 0;
 }
 return certain&&enter.hi<exit.lo?1:-1;
}
// Is intersection POSSIBLE for any receiver in a box? Ignoring an uncertain
// direction axis only enlarges this set. Used to prove all rays FREE, never HIT.
bool bundle_possible(const Point&s,const Point&lo,const Point&hi,const Node&box){
 I enter(0),exit(1);
 for(int a=0;a<3;++a){
  if(std::max(s[a],hi[a])<box.lo[a]||std::min(s[a],lo[a])>box.hi[a])return false;
  I d=sub(I(lo[a],hi[a]),I(s[a]));if(d.lo<=0&&d.hi>=0)continue;
  I e=div(sub(I(box.lo[a]),I(s[a])),d),f=div(sub(I(box.hi[a]),I(s[a])),d);if(d.hi<0)std::swap(e,f);
  enter={std::max(enter.lo,e.lo),std::max(enter.hi,e.hi)};exit={std::min(exit.lo,f.lo),std::min(exit.hi,f.hi)};
  if(enter.lo>exit.hi)return false;
 }
 return true;
}
void push(std::vector<Node>&stack,const Node&node,std::int32_t first){for(int k=7;k>=0;--k)stack.push_back(child(node,first+k,k));}
}

CertifiedSpace::CertifiedSpace(std::vector<std::int8_t> labels,std::vector<std::int32_t> children):labels_(std::move(labels)),children_(std::move(children)){
 if(labels_.empty()||labels_.size()!=children_.size())throw std::invalid_argument("Invalid certified space arrays");
 std::vector<unsigned char> owners(labels_.size(),0);owners[0]=1;std::vector<unsigned> depths(labels_.size(),0);
 for(std::size_t i=0;i<labels_.size();++i){
  if(!owners[i])throw std::invalid_argument("Unreachable certificate node");int label=labels_[i];auto first=children_[i];
  if(label!=2){if((label!=-1&&label!=0&&label!=1)||first!=-1)throw std::invalid_argument("Invalid certificate leaf");continue;}
  if(first<=static_cast<std::int32_t>(i)||static_cast<std::size_t>(first)+8>labels_.size()||depths[i]>=12)throw std::invalid_argument("Invalid certificate children/depth");
  for(int k=0;k<8;++k){if(owners[first+k]++)throw std::invalid_argument("Shared certificate children");depths[first+k]=depths[i]+1;}
 }
}
int CertifiedSpace::contains(const Point&p)const{
 validate(p);Node node{0,{0,0,0},{1,1,1}};
 while(labels_[node.index]==2){int k=0;for(int a=0;a<3;++a)if(p[a]>=(node.lo[a]+node.hi[a])/2)k|=1<<a;node=child(node,children_[node.index]+k,k);}
 return labels_[node.index];
}
int CertifiedSpace::visibility(const Point&s,const Point&p,std::uint64_t&tests)const{
 validate(s);validate(p);std::vector<Node>stack{{0,{0,0,0},{1,1,1}}};bool unknown=false;
 while(!stack.empty()){
  Node node=stack.back();stack.pop_back();int label=labels_[node.index];if(label==0)continue;
  ++tests;int hit=clip(s,p,node);if(hit==0)continue;
  if(label==1&&hit==1)return 0;
  if(label==2)push(stack,node,children_[node.index]);else unknown=true;
 }
 return unknown?-1:1;
}
int CertifiedSpace::certificate(const Point&s,const Point&lo,const Point&hi,std::uint64_t&tests)const{
 validate(s);validate(lo);validate(hi);for(int a=0;a<3;++a)if(lo[a]>hi[a])throw std::invalid_argument("Reversed receiver box");
 // Search only solid leaves reachable by the first corner. All EIGHT corners
 // must hit the SAME convex interior box; hits on different leaves do not prove
 // a shadow-volume fill. Convexity then covers every receiver in the box.
 std::vector<Node>stack{{0,{0,0,0},{1,1,1}}};
 while(!stack.empty()){
  Node node=stack.back();stack.pop_back();int label=labels_[node.index];if(label==0||label==-1)continue;
  ++tests;int hit=clip(s,lo,node);if(hit==0)continue;
  if(label==2){push(stack,node,children_[node.index]);continue;}
  bool all=hit==1;
  for(int k=1;k<8&&all;++k){Point p;for(int a=0;a<3;++a)p[a]=(k&(1<<a))?hi[a]:lo[a];++tests;all=clip(s,p,node)==1;}
  if(all)return 0;
 }
 // Every possible source-to-box segment must be covered exclusively by FREE
 // closed cells. Touching any unknown/solid cell yields no visibility proof.
 stack.push_back({0,{0,0,0},{1,1,1}});
 while(!stack.empty()){
  Node node=stack.back();stack.pop_back();int label=labels_[node.index];if(label==0)continue;
  ++tests;if(!bundle_possible(s,lo,hi,node))continue;
  if(label!=2)return -1;push(stack,node,children_[node.index]);
 }
 return 1;
}

CertifiedGrid::CertifiedGrid(unsigned n,std::vector<double>coefficients,std::vector<double>basis,std::array<double,2>bias,int constant):n_(n),coefficients_(std::move(coefficients)),basis_(std::move(basis)),bias_(bias),constant_(constant){
 if(!n||n>1024||(n&(n-1))||coefficients_.size()%2||basis_.size()!=coefficients_.size()/2*3*n*4||constant<-1||constant>1)throw std::invalid_argument("Invalid certified grid shape");
 auto check=[](double lo,double hi){if(!std::isfinite(lo)||!std::isfinite(hi)||lo>hi)throw std::invalid_argument("Invalid finite enclosure");};
 for(std::size_t i=0;i<coefficients_.size();i+=2)check(coefficients_[i],coefficients_[i+1]);
 for(std::size_t i=0;i<basis_.size();i+=2)check(basis_[i],basis_[i+1]);check(bias_[0],bias_[1]);
}
int CertifiedGrid::contains(const Point&p)const{
 validate(p);std::array<unsigned,3>index;
 for(int a=0;a<3;++a){double v=p[a]*n_-.5;if(v<0||v>=n_||v!=std::floor(v))return -1;index[a]=static_cast<unsigned>(v);}
 if(constant_>=0)return constant_;
 I score(bias_[0],bias_[1]);
 for(std::size_t term=0;term<coefficients_.size()/2;++term){
  I c(1),s(0);
  for(int a=0;a<3;++a){const double* b=&basis_[((term*3+a)*n_+index[a])*4];I ca(b[0],b[1]),sa(b[2],b[3]);
   I next_c=sub(mul(c,ca),mul(s,sa)),next_s=add(mul(s,ca),mul(c,sa));c=next_c;s=next_s;}
  score=add(score,mul(I(coefficients_[2*term],coefficients_[2*term+1]),c));
 }
 return score.lo>0?1:score.hi<0?0:-1;
}
}
