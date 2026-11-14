#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <gmpxx.h>
#include "../../native/filter.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
namespace py=pybind11;
using P=std::array<double,3>;
using Q=mpq_class;
struct I {double lo,hi;I(double x):lo(x),hi(x){} I(double a,double b):lo(a),hi(b){}};
double down(double x){return std::nextafter(x,-INFINITY);}
double up(double x){return std::nextafter(x,INFINITY);}
I whole(){return {-INFINITY,INFINITY};}
I add(I a,I b){double x=a.lo+b.lo,y=a.hi+b.hi;return std::isnan(x)||std::isnan(y)?whole():I(down(x),up(y));}
I sub(I a,I b){return add(a,I(-b.hi,-b.lo));}
I mul(I a,I b){double v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};for(double x:v)if(std::isnan(x))return whole();return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};}
I divide(I a,I b){if(b.lo<=0&&b.hi>=0)return whole();return mul(a,I(down(1/b.hi),up(1/b.lo)));}
I enclosure(const Q&q){double x=q.get_d();if(!std::isfinite(x))throw std::invalid_argument("Nonfinite coefficient");double a=x,b=x;while(Q(a)>q)a=down(a);while(Q(b)<q)b=up(b);return {a,b};}
struct Plane {
 std::array<Q,4> q;std::array<I,4> v{I(0),I(0),I(0),I(0)};
 explicit Plane(const std::vector<std::string>&row){if(row.size()!=4)throw std::invalid_argument("Four rational coefficients required");for(int a=0;a<4;++a){q[a]=Q(row[a]);q[a].canonicalize();v[a]=enclosure(q[a]);}}
 I range(const P&lo,const P&hi)const{I s=v[3];for(int a=0;a<3;++a)s=add(s,mul(v[a],I(lo[a],hi[a])));return s;}
 Q exact(const P&p)const{Q r=q[3];for(int a=0;a<3;++a)r+=q[a]*Q(p[a]);return r;}
};
using Poly=std::vector<Plane>;
using Rows=std::vector<std::vector<std::vector<std::string>>>;
std::vector<Poly> polys(const Rows&rows){std::vector<Poly> p;for(auto&body:rows){if(body.empty())throw std::invalid_argument("Empty plane set");p.emplace_back();for(auto&row:body)p.back().emplace_back(row);}return p;}
int classify(const Poly&p,const P&lo,const P&hi){bool inside=true;for(auto&plane:p){I s=plane.range(lo,hi);if(s.lo>0)return 0;inside&=s.hi<=0;}return inside?1:-1;}
int united(const std::vector<Poly>&p,const P&lo,const P&hi){bool unknown=false;for(auto&body:p){int v=classify(body,lo,hi);if(v==1)return 1;unknown|=v<0;}return unknown?-1:0;}
// Independent parametric clipping, NOT the projected shadow halfspaces.
bool clip_exact(const Poly&p,const P&s,const P&x){Q lo(0),hi(1);for(auto&f:p){Q a=f.exact(s),b=f.exact(x),d=b-a;if(d==0){if(a>0)return false;continue;}Q t=-a/d;if(d>0){if(t<hi)hi=t;}else if(t>lo)lo=t;if(lo>hi)return false;}return true;}
int clip_interval(const Poly&p,const P&s,const P&x){I enter(0),exit(1);for(auto&f:p){I a=f.range(s,s),b=f.range(x,x),d=sub(b,a);if(d.lo<=0&&d.hi>=0){if(a.lo>0&&b.lo>0)return 0;return -1;}I t=divide(I(-a.hi,-a.lo),d);if(d.lo>0)exit=I(std::min(exit.lo,t.lo),std::min(exit.hi,t.hi));else enter=I(std::max(enter.lo,t.lo),std::max(enter.hi,t.hi));if(enter.lo>exit.hi)return 0;}return enter.hi<exit.lo?1:-1;}
struct Stats {std::uint64_t nodes=0,covered=0,rays=0,points=0,unknown_rays=0,unknown_points=0,exact_clips=0;};
py::dict report(const Stats&s){py::dict r;r["nodes"]=s.nodes;r["certificate_covered_voxels"]=s.covered;r["point_rays"]=s.rays;r["solid_points"]=s.points;r["unknown_rays"]=s.unknown_rays;r["unknown_points"]=s.unknown_points;r["reference_exact_clips"]=s.exact_clips;return r;}
struct Geometry {
 std::vector<Poly> bodies;std::vector<r3::Primitive> old;
 Geometry(const Rows&g,const std::vector<std::vector<double>>&primitive):bodies(polys(g)){for(auto&r:primitive){if(r.size()!=7)throw std::invalid_argument("Primitive needs seven values");old.push_back({0,{r[1],r[2],r[3]},{r[4],r[5],r[6]}});}}
};
class Model {
public:
 std::shared_ptr<Geometry> owner;const std::vector<Poly>&geometry;const std::vector<r3::Primitive>&old;std::vector<Poly>shadow;P source;
 Model(std::shared_ptr<Geometry>g,const Rows&v,P s):owner(std::move(g)),geometry(owner->bodies),old(owner->old),shadow(polys(v)),source(s){}
 std::array<int,2> certificate(P lo,P hi)const{int m=united(geometry,lo,hi),blocked=united(shadow,lo,hi);return {m==1||blocked==1?0:blocked==0?1:-1,m};}
 py::tuple points(py::array_t<double,py::array::c_style>p,bool reference){if(p.ndim()!=2||p.shape(1)!=3)throw std::invalid_argument("[N,3] points required");py::array_t<std::int8_t>y(p.shape(0)),m(p.shape(0));Stats stats;{
 py::gil_scoped_release release;for(py::ssize_t k=0;k<p.shape(0);++k){P x{p.data()[3*k],p.data()[3*k+1],p.data()[3*k+2]};for(double a:x)if(!std::isfinite(a)||a<0||a>1)throw std::invalid_argument("Invalid point");int solid=united(geometry,x,x);if(reference&&solid<0){solid=0;for(auto&body:geometry){bool in=true;for(auto&f:body)if(f.exact(x)>0){in=false;break;}if(in){solid=1;break;}}}int blocked;
 if(reference){blocked=0;for(auto&body:geometry){int c=clip_interval(body,source,x);if(c<0){c=clip_exact(body,source,x);++stats.exact_clips;}if(c==1){blocked=1;break;}}}else blocked=united(shadow,x,x);
 y.mutable_data()[k]=blocked<0?-1:1-blocked;m.mutable_data()[k]=solid;
 }}return py::make_tuple(y,m,report(stats));}
 py::tuple raster(std::array<int,3>origin,int side,int n,int leaf,bool old_mode,bool reuse_mask=false){
 if(side<1||side>64||n<1||n>1024||leaf<1)throw std::invalid_argument("Invalid raster extent");for(int a:origin)if(a<0||a+side>n)throw std::invalid_argument("Invalid origin");if(old_mode&&old.size()!=geometry.size())throw std::invalid_argument("Original native control only supports diagonal boxes");
 py::array_t<std::int8_t>yy({side,side,side}),mm({side,side,side});auto*y=yy.mutable_data();auto*m=mm.mutable_data();Stats stats;r3::Filter control(old);
 struct Node{std::array<int,3>o,l;};std::vector<Node>pending{{origin,{side,side,side}}};{
 py::gil_scoped_release release;
 while(!pending.empty()){Node node=pending.back();pending.pop_back();P lo,hi;for(int a=0;a<3;++a){lo[a]=(node.o[a]+.5)/n;hi[a]=(node.o[a]+node.l[a]-.5)/n;}std::array<int,2> c;
 if(reuse_mask){int blocked=united(shadow,lo,hi);c={blocked<0?-1:1-blocked,0};}else c=old_mode?control.certificate(source,lo,hi):certificate(lo,hi);
 ++stats.nodes;bool fill=c[0]>=0&&c[1]>=0;if(fill)stats.covered+=node.l[0]*node.l[1]*node.l[2];
 if(fill||*std::max_element(node.l.begin(),node.l.end())<=leaf){for(int z=0;z<node.l[2];++z)for(int v=0;v<node.l[1];++v)for(int x=0;x<node.l[0];++x){P p{(node.o[0]+x+.5)/n,(node.o[1]+v+.5)/n,(node.o[2]+z+.5)/n};std::size_t k=((node.o[2]+z-origin[2])*side+node.o[1]+v-origin[1])*side+node.o[0]+x-origin[0];int a=c[0],b=c[1];if(a<0){++stats.rays;a=old_mode?control.visibility(source,p):united(shadow,p,p);if(!old_mode&&a>=0)a=1-a;}if(b<0){++stats.points;b=old_mode?control.contains(p):united(geometry,p,p);}y[k]=a;m[k]=b;stats.unknown_rays+=a<0;stats.unknown_points+=b<0;}}
 else{for(int z=1;z>=0;--z)for(int v=1;v>=0;--v)for(int x=1;x>=0;--x){int bits[]={x,v,z};Node child=node;for(int a=0;a<3;++a){int cut=node.l[a]/2;if(node.l[a]<=leaf){if(bits[a]){child.l[0]=0;break;}}else{child.o[a]+=bits[a]*cut;child.l[a]=bits[a]?node.l[a]-cut:cut;}}if(child.l[0])pending.push_back(child);}}
 }
 }return py::make_tuple(yy,mm,report(stats));}
};
PYBIND11_MODULE(_shadow_native,m){py::class_<Geometry,std::shared_ptr<Geometry>>(m,"Geometry").def(py::init<const Rows&,const std::vector<std::vector<double>>&>());py::class_<Model>(m,"Model").def(py::init<std::shared_ptr<Geometry>,const Rows&,P>()).def("certificate",&Model::certificate).def("points",&Model::points).def("raster",&Model::raster,py::arg("origin"),py::arg("side"),py::arg("n"),py::arg("leaf"),py::arg("old_mode"),py::arg("reuse_mask")=false);}
