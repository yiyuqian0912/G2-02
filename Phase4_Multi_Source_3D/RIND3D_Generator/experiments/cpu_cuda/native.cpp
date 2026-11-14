#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <xmmintrin.h>
namespace py=pybind11;
using Q=mpq_class;
using P=std::array<double,3>;
using Rows=std::vector<std::vector<std::vector<std::string>>>;
struct I { double lo,hi; I(double x=0):lo(x),hi(x){} I(double a,double b):lo(a),hi(b){} };
// Equivalent to nextafter(x,+/-infinity), including signed zero/subnormals.
double step(double x,bool positive){
 if(std::isnan(x)||x==(positive?INFINITY:-INFINITY))return x;
 if(x==0)return positive?std::numeric_limits<double>::denorm_min():-std::numeric_limits<double>::denorm_min();
 auto b=std::bit_cast<std::uint64_t>(x); b+=(positive==(x>0))?1:std::uint64_t(-1);
 return std::bit_cast<double>(b);
}
I whole(){return {-INFINITY,INFINITY};}
I add(I a,I b){double l=a.lo+b.lo,h=a.hi+b.hi;return std::isnan(l)||std::isnan(h)?whole():I(step(l,false),step(h,true));}
I sub(I a,I b){return add(a,{-b.hi,-b.lo});}
I mul(I a,I b){
 if((a.lo==0&&a.hi==0)||(b.lo==0&&b.hi==0))return I(0);
 double v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
 for(double x:v)if(std::isnan(x))return whole();
 return {step(*std::min_element(v,v+4),false),step(*std::max_element(v,v+4),true)};
}
I divi(I a,I b){if(b.lo<=0&&b.hi>=0)return whole();return mul(a,{step(1/b.hi,false),step(1/b.lo,true)});}
I enclosure(const Q&q){double x=q.get_d();if(!std::isfinite(x))throw std::invalid_argument("Nonfinite coefficient");double a=x,b=x;while(Q(a)>q)a=step(a,false);while(Q(b)<q)b=step(b,true);return {a,b};}
struct Plane {
 std::array<Q,4>q;std::array<I,4>v;
 Plane(const std::vector<std::string>&r){if(r.size()!=4)throw std::invalid_argument("Four coefficients required");for(int i=0;i<4;++i){q[i]=Q(r[i]);q[i].canonicalize();v[i]=enclosure(q[i]);}}
 I range(P lo,P hi)const{I s=v[3];for(int a=0;a<3;++a)if(q[a]!=0)s=add(s,mul(v[a],{lo[a],hi[a]}));return s;}
 Q exact(P p)const{Q s=q[3];for(int a=0;a<3;++a)s+=q[a]*Q(p[a]);return s;}
};
using Poly=std::vector<Plane>;
std::vector<Poly> parse(const Rows&r){std::vector<Poly>p;for(auto&b:r){if(b.empty())throw std::invalid_argument("Empty convex body");p.emplace_back();for(auto&v:b)p.back().emplace_back(v);}return p;}
using VQ=std::array<Q,3>;
VQ cross(const Plane&a,const Plane&b){return {a.q[1]*b.q[2]-a.q[2]*b.q[1],a.q[2]*b.q[0]-a.q[0]*b.q[2],a.q[0]*b.q[1]-a.q[1]*b.q[0]};}
Q dot(const Plane&a,const VQ&v){Q s;for(int k=0;k<3;++k)s+=a.q[k]*v[k];return s;}
// Exact recession-cone boundedness test, same mathematical criterion as the
// canonical halfspace backend. Python also supplies a strict interior witness.
void bounded(const Poly&p){bool rank3=false;for(std::size_t i=0;i<p.size();++i)for(std::size_t j=0;j<i;++j){auto v=cross(p[i],p[j]);if(v[0]==0&&v[1]==0&&v[2]==0)continue;bool pos=true,neg=true;for(auto&a:p){Q d=dot(a,v);pos&=d<=0;neg&=d>=0;rank3|=d!=0;}if(pos||neg)throw std::invalid_argument("Unbounded recession cone");}if(!rank3)throw std::invalid_argument("Unbounded rank-deficient halfspaces");}
struct Geometry {
 std::vector<Poly>bodies;
 explicit Geometry(const Rows&r):bodies(parse(r)){
  if(std::fegetround()!=FE_TONEAREST||(_mm_getcsr()&((1<<15)|(1<<6))))throw std::runtime_error("Requires round-to-nearest, gradual underflow, no FTZ/DAZ");
  for(auto&p:bodies)bounded(p);
 }
};
struct Stats {std::uint64_t nodes=0,covered=0,rays=0,points=0,clips=0,exact=0;};
py::dict report(const Stats&s){py::dict d;d["nodes"]=s.nodes;d["certificate_covered_voxels"]=s.covered;d["point_rays"]=s.rays;d["solid_points"]=s.points;d["interval_clips"]=s.clips;d["exact_fallbacks"]=s.exact;return d;}
int classify(const Poly&p,P lo,P hi){bool in=true;for(auto&f:p){I v=f.range(lo,hi);if(v.lo>0)return 0;in&=v.hi<=0;}return in?1:-1;}
int united(const std::vector<Poly>&p,P lo,P hi){bool unknown=false;for(auto&b:p){int v=classify(b,lo,hi);if(v==1)return 1;unknown|=v<0;}return unknown?-1:0;}
class Model {
public:
 std::shared_ptr<Geometry>g;P source;std::vector<std::vector<I>>score;std::vector<std::vector<Q>>qscore;std::vector<Poly>shadow;
 Model(std::shared_ptr<Geometry>v,P s,const Rows&rows):g(std::move(v)),source(s),shadow(parse(rows)){
  for(double x:s)if(!std::isfinite(x)||x<=0||x>=1)throw std::invalid_argument("Source outside open world");
  for(auto&p:g->bodies){std::vector<I>a;std::vector<Q>b;bool outside=false;for(auto&f:p){Q x=f.exact(s);a.push_back(enclosure(x));b.push_back(x);outside|=x>0;}if(!outside)throw std::invalid_argument("Source in closed obstacle");score.push_back(a);qscore.push_back(b);}
 }
 int clip(std::size_t k,P lo,P hi,Stats&st)const{
  ++st.clips;I enter(0),exit(1);auto&p=g->bodies[k];bool uncertain=false;
  for(std::size_t i=0;i<p.size();++i){I a=score[k][i],b=p[i].range(lo,hi),d=sub(b,a);
   if(a.lo>0&&b.lo>0)return 0;
   if(d.lo<=0&&d.hi>=0){uncertain=true;continue;} // ignore constraint => conservative superset
   I t=divi({-a.hi,-a.lo},d);
   if(d.lo>0)exit={std::min(exit.lo,t.lo),std::min(exit.hi,t.hi)};
   else enter={std::max(enter.lo,t.lo),std::max(enter.hi,t.hi)};
   if(enter.lo>exit.hi)return 0;
  }
  return !uncertain&&enter.hi<exit.lo?1:-1;
 }
 bool exact_clip(std::size_t k,P x,Stats&st)const{
  ++st.exact;Q enter(0),exit(1);auto&p=g->bodies[k];
  for(std::size_t i=0;i<p.size();++i){Q a=qscore[k][i],d=p[i].exact(x)-a;if(d==0){if(a>0)return false;continue;}Q t=-a/d;if(d>0){if(t<exit)exit=t;}else if(t>enter)enter=t;if(enter>exit)return false;}return true;
 }
 int solid(P x,Stats&st)const{int v=united(g->bodies,x,x);if(v>=0)return v;++st.exact;for(auto&p:g->bodies){bool in=true;for(auto&f:p)if(f.exact(x)>0){in=false;break;}if(in)return 1;}return 0;}
 int visible(P x,Stats&st)const{for(std::size_t k=0;k<g->bodies.size();++k){int v=clip(k,x,x,st);if(v==1||(v<0&&exact_clip(k,x,st)))return 0;}return 1;}
 std::array<int,2> cert(P lo,P hi,Stats&st,bool projected=false,bool skip_mask=false)const{
  int m=skip_mask?0:united(g->bodies,lo,hi);if(m==1)return {0,1};
  if(projected){int v=united(shadow,lo,hi);return {v<0?-1:1-v,m};}
  bool all_miss=true;
  for(std::size_t k=0;k<g->bodies.size();++k){int v=clip(k,lo,hi,st);if(v==0)continue;all_miss=false;
   bool corners=true;for(int j=0;j<8&&corners;++j){P x;for(int a=0;a<3;++a)x[a]=(j&(1<<a))?hi[a]:lo[a];corners=clip(k,x,x,st)==1;}
   if(corners)return {0,m};
  }return {all_miss?1:-1,m};
 }
 py::tuple certificate(P lo,P hi){Stats s;auto v=cert(lo,hi,s);return py::make_tuple(v[0],v[1]);}
 py::tuple points(py::array_t<double,py::array::c_style>p){if(p.ndim()!=2||p.shape(1)!=3)throw std::invalid_argument("[N,3] required");py::array_t<std::int8_t>y(p.shape(0)),m(p.shape(0));Stats s;{py::gil_scoped_release r;for(py::ssize_t i=0;i<p.shape(0);++i){P x{p.data()[3*i],p.data()[3*i+1],p.data()[3*i+2]};for(double a:x)if(!std::isfinite(a)||a<0||a>1)throw std::invalid_argument("Invalid point");m.mutable_data()[i]=solid(x,s);y.mutable_data()[i]=visible(x,s);}}return py::make_tuple(y,m,report(s));}
 py::tuple raster(std::array<int,3>origin,int side,int n,int leaf=8,bool projected=false,bool skip_mask=false){
  if(side<1||side>64||n<1||n>1024||(n&(n-1))||leaf<1)throw std::invalid_argument("Invalid raster extent");for(int x:origin)if(x<0||x+side>n)throw std::invalid_argument("Invalid origin");
  py::array_t<std::int8_t>yy({side,side,side}),mm({side,side,side});auto*y=yy.mutable_data();auto*m=mm.mutable_data();Stats st;
  struct Node{std::array<int,3>o,l;};std::vector<Node>pending{{origin,{side,side,side}}};{py::gil_scoped_release r;
  while(!pending.empty()){auto node=pending.back();pending.pop_back();P lo,hi;for(int a=0;a<3;++a){lo[a]=(node.o[a]+.5)/n;hi[a]=(node.o[a]+node.l[a]-.5)/n;}auto c=cert(lo,hi,st,projected,skip_mask);++st.nodes;bool fill=c[0]>=0&&c[1]>=0;if(fill)st.covered+=node.l[0]*node.l[1]*node.l[2];
   if(fill||*std::max_element(node.l.begin(),node.l.end())<=leaf){for(int z=0;z<node.l[2];++z)for(int v=0;v<node.l[1];++v)for(int x=0;x<node.l[0];++x){P p{(node.o[0]+x+.5)/n,(node.o[1]+v+.5)/n,(node.o[2]+z+.5)/n};std::size_t k=((node.o[2]+z-origin[2])*side+node.o[1]+v-origin[1])*side+node.o[0]+x-origin[0];int a=c[0],b=c[1];if(a<0){++st.rays;a=projected?united(shadow,p,p):-1;a=a<0?visible(p,st):1-a;}if(b<0){++st.points;b=solid(p,st);}y[k]=a;m[k]=b;}}
   else for(int z=1;z>=0;--z)for(int v=1;v>=0;--v)for(int x=1;x>=0;--x){int bits[]={x,v,z};Node child=node;for(int a=0;a<3;++a){int cut=node.l[a]/2;if(node.l[a]<=leaf){if(bits[a]){child.l[0]=0;break;}}else{child.o[a]+=bits[a]*cut;child.l[a]=bits[a]?node.l[a]-cut:cut;}}if(child.l[0])pending.push_back(child);}
  }}return py::make_tuple(yy,mm,report(st));
 }
};
PYBIND11_MODULE(_efficient_native,m){m.def("step",&step);m.def("interval",[](double a,double b,int op){I v=op==0?add(I(a),I(b)):op==1?mul(I(a),I(b)):divi(I(a),I(b));return py::make_tuple(v.lo,v.hi);});py::class_<Geometry,std::shared_ptr<Geometry>>(m,"Geometry",py::module_local()).def(py::init<const Rows&>());py::class_<Model>(m,"Model",py::module_local()).def(py::init<std::shared_ptr<Geometry>,P,const Rows&>()).def("points",&Model::points).def("certificate",&Model::certificate).def("raster",&Model::raster,py::arg("origin"),py::arg("side"),py::arg("n"),py::arg("leaf")=8,py::arg("projected")=false,py::arg("skip_mask")=false);}
