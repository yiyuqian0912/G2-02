#include "max.hpp"
#include "interval.hpp"
#include <numeric>
#include <queue>
#include <stdexcept>

namespace b2max {
void Settings::validate() const {
  if(method<1||method>3||max_nodes<0||max_nodes>1000000||max_depth<0||max_depth>50||!std::isfinite(seconds)||seconds<=0)
    throw std::invalid_argument("Invalid maximum-proof budget/method");
}
namespace {
using Inv=std::array<I,9>;
Inv inverse(const Matrix &m) {
  Q rows[3][6];
  for(int i=0;i<3;i++)for(int j=0;j<6;j++)rows[i][j]=j<3?Q(m[3*i+j]):Q(i==j-3);
  for(int j=0;j<3;j++) {
    int k=j;while(k<3&&rows[k][j]==0)k++;
    if(k==3)throw std::invalid_argument("Singular canonical Gaussian matrix");
    for(int z=0;z<6;z++)std::swap(rows[j][z],rows[k][z]);
    Q d=rows[j][j];for(int z=0;z<6;z++)rows[j][z]/=d;
    for(int i=0;i<3;i++)if(i!=j){d=rows[i][j];for(int z=0;z<6;z++)rows[i][z]-=d*rows[j][z];}
  }
  Inv v;for(int i=0;i<3;i++)for(int j=0;j<3;j++)v[3*i+j]=enclose(rows[i][j+3]);return v;
}
struct Kernel { P center; Inv inv; double weight; };
struct Poly { I a,b,c; };
I at(const Poly &q,I t){return add(mul(add(mul(q.a,t),q.b),t),q.c);}
I qrange(const Poly &p,double lo,double hi) {
  I l=at(p,I(lo)),h=at(p,I(hi));double low=std::min(l.lo,h.lo);
  if(p.a.lo>0) {
    I root=divi(neg(p.b),mul(I(2),p.a));double a=std::max(lo,root.lo),b=std::min(hi,root.hi);
    if(a<=b)low=std::min(low,at(p,I(a,b)).lo);
  } else low=std::min(low,at(p,I(lo,hi)).lo);
  // Positive definiteness follows from nonsingular exact canonical M.
  return {std::max(0.,low),std::max(0.,std::max(l.hi,h.hi))};
}
struct Eval { I value,first,second;std::vector<I> terms;bool derived=false; };
struct Ray {
  const std::vector<Kernel> &kernels;std::vector<Poly> q;Stats &stats;Exp exp;
  std::map<double,Eval> points;
  Ray(const std::vector<Kernel> &k,P s,P x,Stats &st):kernels(k),stats(st) {
    q.reserve(k.size());
    for(const auto &g:k) {
      I a,b,c;
      for(int row=0;row<3;row++) {
        I u,v;for(int d=0;d<3;d++) {
          u=add(u,mul(g.inv[3*row+d],sub(I(s[d]),I(g.center[d]))));
          v=add(v,mul(g.inv[3*row+d],sub(I(x[d]),I(s[d]))));
        }
        a=add(a,square(v));b=add(b,mul(I(2),mul(u,v)));c=add(c,square(u));
      }
      q.push_back({a,b,c});stats.add("coefficient_builds");
    }
  }
  Eval evaluate(double lo,double hi) {
    auto start=Clock::now();Eval e;e.terms.reserve(q.size());
    for(std::size_t i=0;i<q.size();i++) {
      I g=mul(I(kernels[i].weight),exp.value(qrange(q[i],lo,hi),stats));
      e.value=add(e.value,g);e.terms.push_back(g);
    }
    stats.add("function_bounds");stats.add("function_bound_ns",std::uint64_t(elapsed(start)*1e9));return e;
  }
  void derive(Eval &e,double lo,double hi) {
    if(e.derived)return;
    auto start=Clock::now();
    for(std::size_t i=0;i<q.size();i++) {
      I dot=add(mul(q[i].a,I(lo,hi)),mul(I(.5),q[i].b));
      e.first=add(e.first,mul(neg(dot),e.terms[i]));
      e.second=add(e.second,mul(sub(square(dot),q[i].a),e.terms[i]));
    }
    e.derived=true;stats.add("derivative_bounds");stats.add("derivative_bound_ns",std::uint64_t(elapsed(start)*1e9));
  }
  Eval &point(double t) {
    auto it=points.find(t);if(it!=points.end()){stats.add("point_cache_hits");return it->second;}
    auto v=evaluate(t,t);stats.add("point_evaluations");return points.emplace(t,std::move(v)).first->second;
  }
  double taylor(double lo,double hi,Eval &range) {
    double mid=lo+(hi-lo)/2;
    double half=std::max(sub(I(mid),I(lo)).hi,sub(I(hi),I(mid)).hi);
    auto &m=point(mid);derive(m,mid,mid);derive(range,lo,hi);
    double d=std::max(std::abs(m.first.lo),std::abs(m.first.hi)),k=range.second.hi;
    if(!std::isfinite(d)||!std::isfinite(k))return INFINITY;
    I error=add(mul(I(d),I(half)),mul(mul(I(.5),I(std::max(0.,k))),square(I(half))));
    double upper=add(m.value,error).hi;
    if(k<0) {
      I correction=divi(square(I(d)),mul(I(-2),I(k)));
      upper=std::min(upper,add(m.value,correction).hi);stats.add("concave_taylor_tests");
    }
    stats.add("local_taylor_tests");return upper;
  }
};
struct Node { double lo,hi,priority;int depth;std::uint64_t id; };
struct Order {
  bool operator()(const Node &a,const Node &b)const {
    if(a.priority!=b.priority)return a.priority<b.priority;
    if(a.hi-a.lo!=b.hi-b.lo)return a.hi-a.lo<b.hi-b.lo;
    return a.id>b.id;
  }
};
} // namespace
struct Solver::Impl {
  Inputs input;std::vector<Kernel> kernels;I tau;Primary fallback;
  explicit Impl(const Inputs &in):input(in),tau(threshold(in)),fallback(in) {
    // The unchanged constructor validates finite inputs, positive weights and M.
    for(std::size_t i=0;i<in.centers.size();i++)kernels.push_back({in.centers[i],inverse(in.matrices[i]),in.weights[i]});
  }
};
Solver::Solver(const Inputs &i):p(std::make_shared<Impl>(i)){}
Result Solver::solve(P source,P receiver,const Settings &o,bool proof) const {
  o.validate();require_environment();Result result;auto began=Clock::now();auto &st=result.stats;
  auto finish=[&](int status,const std::string &reason,int nodes=0,int depth=0) {
    result.decision={status,reason,nodes,depth};result.seconds=elapsed(began);
    if(status!=0)result.leaves.clear();
    return result;
  };
  if(!finite(source)||!finite(receiver))return finish(-2,"nonfinite_segment");
  Ray ray(p->kernels,source,receiver,st);result.preparation_seconds=elapsed(began);
  auto hit=[&](Eval &e){return e.value.lo>=p->tau.hi;};
  std::priority_queue<Node,std::vector<Node>,Order> queue;queue.push({0,1,INFINITY,0,0});
  std::uint64_t next=1;int nodes=0,depth=0;std::string unresolved;
  auto retain=[&](Leaf leaf){st.add("proof_leaves");if(proof)result.leaves.push_back(leaf);};
  while(!queue.empty()) {
    if(proof&&result.leaves.size()>=8192){unresolved="proof_partition_budget";break;}
    if(nodes>=o.max_nodes||elapsed(began)>o.seconds){unresolved=nodes>=o.max_nodes?"node_budget":"time_budget";break;}
    Node node=queue.top();queue.pop();nodes++;depth=std::max(depth,node.depth);st.add("nodes");st.maximum("max_depth",depth);
    double lo=node.lo,hi=node.hi,mid=lo+(hi-lo)/2;Eval range=ray.evaluate(lo,hi);Leaf leaf;leaf.lo=lo;leaf.hi=hi;leaf.upper=range.value.hi;
    if(range.value.hi<p->tau.lo){retain(leaf);st.add("natural_free");continue;}
    if(hit(range))return finish(1,"whole_interval_solid",nodes,depth);
    // Point proposals are heuristics; only their enclosed f values may label HIT.
    auto &middle=ray.point(mid);
    if(hit(middle))return finish(1,"certified_midpoint_witness",nodes,depth);
    if(nodes==1) {
      if(o.endpoint_probes&&(hit(ray.point(0))||hit(ray.point(1))))return finish(1,"closed_endpoint_witness",nodes,depth);
      std::vector<std::size_t> rank(ray.q.size());std::iota(rank.begin(),rank.end(),0);
      std::stable_sort(rank.begin(),rank.end(),[&](auto a,auto b){return range.terms[a].hi>range.terms[b].hi;});
      for(std::size_t j=0;j<std::min<std::size_t>(2,rank.size());j++) {
        auto q=ray.q[rank[j]];double a=q.a.lo+(q.a.hi-q.a.lo)/2,b=q.b.lo+(q.b.hi-q.b.lo)/2;
        double t=a>0&&std::isfinite(a)&&std::isfinite(b)?std::clamp(-b/(2*a),0.,1.):.5;
        if(hit(ray.point(t)))return finish(1,"certified_kernel_peak_proposal",nodes,depth);
      }
    }
    ray.derive(range,lo,hi);
    if(range.first.lo>0||range.first.hi<0) {
      bool positive=range.first.lo>0;auto &endpoint=ray.point(positive?hi:lo);
      if(hit(endpoint))return finish(1,"monotone_endpoint_hit",nodes,depth);
      if(endpoint.value.hi<p->tau.lo) {
        leaf.op=positive?1:2;leaf.upper=endpoint.value.hi;retain(leaf);st.add("monotone_free");continue;
      }
    }
    if(o.method==3) {
      double upper=ray.taylor(lo,hi,range);
      if(upper<p->tau.lo){leaf.op=3;leaf.upper=upper;retain(leaf);st.add("local_taylor_free");continue;}
    }
    if(o.method>=2&&(range.second.lo>0||range.second.hi<0)) {
      st.add("stationary_tests");ray.derive(middle,mid,mid);
      I n=sub(I(mid),divi(middle.first,range.second));double jl=std::max(lo,n.lo),jh=std::min(hi,n.hi);
      bool empty=jl>jh;
      if(empty||jh-jl<.75*(hi-lo)) {
        st.add("stationary_contractions");double upper=std::max(ray.point(lo).value.hi,ray.point(hi).value.hi);int inner_op=0;
        if(!empty) {
          Eval inner=ray.evaluate(jl,jh);double inside=inner.value.hi;
          if(o.method==3) {double t=ray.taylor(jl,jh,inner);if(t<inside){inside=t;inner_op=3;}}
          upper=std::max(upper,inside);
          if(hit(ray.point(jl+(jh-jl)/2)))return finish(1,"certified_stationary_region_witness",nodes,depth);
        }
        if(upper<p->tau.lo) {
          leaf.op=4;leaf.upper=upper;leaf.empty=empty;leaf.jlo=empty?lo:jl;leaf.jhi=empty?lo:jh;leaf.inner_op=inner_op;
          retain(leaf);st.add("stationary_free");continue;
        }
      }
    }
    if(node.depth>=o.max_depth||mid<=lo||mid>=hi){unresolved="depth_or_binary64_enclosure_budget";break;}
    // Upper-bound ordering is scheduling only, never proof of a physical label.
    double priority=std::isnan(range.value.hi)?INFINITY:range.value.hi;
    queue.push({lo,mid,priority,node.depth+1,next++});queue.push({mid,hi,priority,node.depth+1,next++});
    st.add("subdivisions");st.maximum("active_nodes",queue.size());
  }
  if(unresolved.empty())return finish(0,"all_closed_intervals_strictly_below_threshold",nodes,depth);
  if(o.fallback) {
    result.fallback_used=true;Options old;old.max_nodes=o.max_nodes;old.max_depth=o.max_depth;old.query_seconds=o.seconds;
    auto t=Clock::now();auto d=p->fallback.ray(source,receiver,old,st);st.add("fallback_nanoseconds",std::uint64_t(elapsed(t)*1e9));
    result.leaves.clear();return finish(d.status,"old_primary_fallback:"+d.reason,nodes+d.nodes,std::max(depth,d.depth));
  }
  return finish(-1,unresolved,nodes,depth);
}
std::map<std::string,std::vector<double>> Solver::bounds(P s,P x,double lo,double hi)const {
  require_environment();if(!finite(s)||!finite(x)||!std::isfinite(lo)||!std::isfinite(hi)||lo<0||hi>1||lo>hi)throw std::invalid_argument("Bad diagnostic interval");
  Stats stats;Ray ray(p->kernels,s,x,stats);auto e=ray.evaluate(lo,hi);ray.derive(e,lo,hi);
  return {{"f",{e.value.lo,e.value.hi}},{"first",{e.first.lo,e.first.hi}},{"second",{e.second.lo,e.second.hi}},{"local_upper",{ray.taylor(lo,hi,e)}}};
}
} // namespace b2max
