#include "filter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace r3 {
// Every arithmetic operation encloses the corresponding real operation on the
// exact input binary64 numbers. No fast-math or contracted operations are used.
struct I {double lo,hi;I(double v):lo(v),hi(v){}I(double a,double b):lo(a),hi(b){}};
static double down(double v){return std::nextafter(v,-std::numeric_limits<double>::infinity());}
static double up(double v){return std::nextafter(v,std::numeric_limits<double>::infinity());}
static I operator+(I a,I b){return {down(a.lo+b.lo),up(a.hi+b.hi)};}
static I operator-(I a,I b){return {down(a.lo-b.hi),up(a.hi-b.lo)};}
static I operator*(I a,I b){double v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};}
static I operator/(I a,I b){if(b.lo<=0&&b.hi>=0)return {-INFINITY,INFINITY};return a*I(down(1/b.hi),up(1/b.lo));}
static I sq(I a){double low=(a.lo<=0&&a.hi>=0)?0:down(std::min(a.lo*a.lo,a.hi*a.hi));return {low,up(std::max(a.lo*a.lo,a.hi*a.hi))};}
using V=std::array<I,3>;
static V exact_v(const std::array<double,3>& a){return {I(a[0]),I(a[1]),I(a[2])};}
static int occupancy(const Primitive& p,const V& q){
 if(p.kind==0){bool inside=true;for(int a=0;a<3;++a){auto lo=I(p.center[a])-I(p.half[a]),hi=I(p.center[a])+I(p.half[a]);if(q[a].hi<lo.lo||q[a].lo>hi.hi)return 0;inside&=q[a].lo>lo.hi&&q[a].hi<hi.lo;}return inside?1:-1;}
 I value(0);for(int a=0;a<3;++a)value=value+sq((q[a]-I(p.center[a]))/I(p.half[a]));
 if(value.hi<1)return 1;if(value.lo>1)return 0;return -1;
}
static int intersects(const Primitive& p,const V& source,const V& q){
 // Cheap, conservative enclosure of the entire source-to-receiver hull.
 for(int a=0;a<3;++a){auto lo=I(p.center[a])-I(p.half[a]),hi=I(p.center[a])+I(p.half[a]);
  if(std::max(source[a].hi,q[a].hi)<lo.lo||std::min(source[a].lo,q[a].lo)>hi.hi)return 0;}
 if(p.kind==0){
  I enter(0),exit(1);
  for(int a=0;a<3;++a){
   auto d=q[a]-source[a],lo=I(p.center[a])-I(p.half[a]),hi=I(p.center[a])+I(p.half[a]);
   if(d.lo<=0&&d.hi>=0){if(q[a].lo==source[a].lo&&q[a].hi==source[a].hi&&q[a].lo==q[a].hi){if(source[a].hi<lo.lo||source[a].lo>hi.hi)return 0;continue;}return -1;}
   I e=(lo-source[a])/d,f=(hi-source[a])/d;if(d.hi<0)std::swap(e,f);
   enter={std::max(enter.lo,e.lo),std::max(enter.hi,e.hi)};exit={std::min(exit.lo,f.lo),std::min(exit.hi,f.hi)};
  }
  if(enter.lo>exit.hi)return 0;if(enter.hi<exit.lo)return 1;return -1;
 }
 I a(0),b(0),c(-1);
 for(int axis=0;axis<3;++axis){auto d=(q[axis]-source[axis])/I(p.half[axis]),o=(source[axis]-I(p.center[axis]))/I(p.half[axis]);a=a+sq(d);b=b+o*d;c=c+sq(o);}
 if(!(a.lo>0))return -1;
 I t=(I(0)-b)/a;t={std::clamp(t.lo,0.,1.),std::clamp(t.hi,0.,1.)};
 auto minimum=c+I(2)*b*t+a*sq(t);
 if(minimum.hi<0)return 1;if(minimum.lo>0)return 0;return -1;
}
int Filter::contains(const std::array<double,3>& p)const{
 bool unknown=false;for(auto& item:pieces_){auto d=occupancy(item,exact_v(p));if(d==1)return 1;unknown|=d<0;}return unknown?-1:0;
}
int Filter::visibility(const std::array<double,3>& source,const std::array<double,3>& p)const{
 bool unknown=false;for(auto& item:pieces_){auto d=intersects(item,exact_v(source),exact_v(p));if(d==1)return 0;unknown|=d<0;}return unknown?-1:1;
}
std::array<int,2> Filter::certificate(const std::array<double,3>& source,const std::array<double,3>& lo,const std::array<double,3>& hi)const{
 V region={I(lo[0],hi[0]),I(lo[1],hi[1]),I(lo[2],hi[2])};bool free=true,visible=true,solid=false,blocked=false;
 for(auto& item:pieces_){auto o=occupancy(item,region);free&=o==0;solid|=o==1;
  auto s=intersects(item,exact_v(source),region);visible&=s==0;
  // For one convex solid, its closed shadow from an exterior source is convex:
  // {s+t*(q-s): q in solid, t>=1}. All eight corners prove the whole box.
  bool corners=true;for(int k=0;k<8&&corners;++k){std::array<double,3> p;for(int a=0;a<3;++a)p[a]=(k&(1<<a))?hi[a]:lo[a];corners&=intersects(item,exact_v(source),exact_v(p))==1;}
  blocked|=corners;
 }
 return {solid||blocked?0:visible?1:-1,solid?1:free?0:-1};
}
}
