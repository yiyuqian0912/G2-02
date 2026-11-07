// A proof checker, NOT an independent physical renderer. No primary bound,
// search routine, interval class or geometry inverse is used here.
#include "max.hpp"
#include <stdexcept>

namespace b2max {
namespace {
thread_local int precision=128;
struct R {
  MP l,h;
  R():l(precision),h(precision){}
  explicit R(double d):R(){mpfr_set_d(l.v,d,MPFR_RNDN);mpfr_set_d(h.v,d,MPFR_RNDN);}
  explicit R(const Q &q):R(){mpfr_set_q(l.v,q.get_mpq_t(),MPFR_RNDD);mpfr_set_q(h.v,q.get_mpq_t(),MPFR_RNDU);}
  R(double a,double b):R(){mpfr_set_d(l.v,a,MPFR_RNDN);mpfr_set_d(h.v,b,MPFR_RNDN);}
};
R plus(const R&a,const R&b){R r;mpfr_add(r.l.v,a.l.v,b.l.v,MPFR_RNDD);mpfr_add(r.h.v,a.h.v,b.h.v,MPFR_RNDU);return r;}
R minus(const R&a,const R&b){R r;mpfr_sub(r.l.v,a.l.v,b.h.v,MPFR_RNDD);mpfr_sub(r.h.v,a.h.v,b.l.v,MPFR_RNDU);return r;}
R negative(const R&a){R r;mpfr_neg(r.l.v,a.h.v,MPFR_RNDN);mpfr_neg(r.h.v,a.l.v,MPFR_RNDN);return r;}
R times(const R&a,const R&b) {
  R r;MP v(precision);bool first=true;
  for(mpfr_srcptr x:{a.l.v,a.h.v})for(mpfr_srcptr y:{b.l.v,b.h.v}) {
    mpfr_mul(v.v,x,y,MPFR_RNDD);if(first||mpfr_cmp(v.v,r.l.v)<0)mpfr_set(r.l.v,v.v,MPFR_RNDD);
    mpfr_mul(v.v,x,y,MPFR_RNDU);if(first||mpfr_cmp(v.v,r.h.v)>0)mpfr_set(r.h.v,v.v,MPFR_RNDU);first=false;
  }
  return r;
}
R squared(const R&a) {
  R r;
  if(mpfr_sgn(a.l.v)<=0&&mpfr_sgn(a.h.v)>=0)mpfr_set_zero(r.l.v,0);
  else mpfr_sqr(r.l.v,mpfr_cmpabs(a.l.v,a.h.v)<0?a.l.v:a.h.v,MPFR_RNDD);
  mpfr_sqr(r.h.v,mpfr_cmpabs(a.l.v,a.h.v)>0?a.l.v:a.h.v,MPFR_RNDU);return r;
}
R quotient(const R&a,const R&b) {
  if(mpfr_sgn(b.l.v)<=0&&mpfr_sgn(b.h.v)>=0)throw std::runtime_error("zero denominator in proof");
  R reciprocal;mpfr_ui_div(reciprocal.l.v,1,b.h.v,MPFR_RNDD);mpfr_ui_div(reciprocal.h.v,1,b.l.v,MPFR_RNDU);return times(a,reciprocal);
}
R exponential(const R&q,Stats &st) {
  R r;MP v(precision);
  mpfr_neg(v.v,q.h.v,MPFR_RNDN);mpfr_div_2ui(v.v,v.v,1,MPFR_RNDD);mpfr_exp(r.l.v,v.v,MPFR_RNDD);
  mpfr_neg(v.v,q.l.v,MPFR_RNDN);mpfr_div_2ui(v.v,v.v,1,MPFR_RNDU);mpfr_exp(r.h.v,v.v,MPFR_RNDU);
  st.add("checker_exp_endpoints",2);return r;
}
std::array<Q,9> cofactors(const Matrix&m) {
  Q a(m[0]),b(m[1]),c(m[2]),d(m[3]),e(m[4]),f(m[5]),g(m[6]),h(m[7]),i(m[8]);
  Q det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  if(det==0)throw std::invalid_argument("Checker singular canonical matrix");
  return {Q((e*i-f*h)/det),Q((c*h-b*i)/det),Q((b*f-c*e)/det),Q((f*g-d*i)/det),Q((a*i-c*g)/det),Q((c*d-a*f)/det),Q((d*h-e*g)/det),Q((b*g-a*h)/det),Q((a*e-b*d)/det)};
}
struct Canonical { P center;std::array<Q,9> inverse;double weight; };
struct Coeff { R a,mu,minimum,constant;bool stationary=false; };
struct Values {R f,first,second;};
struct CheckRay {
  const std::vector<Canonical>&kernels;std::vector<Coeff> coefficients;Stats &stats;
  CheckRay(const std::vector<Canonical>&k,P source,P receiver,Stats &st):kernels(k),stats(st) {
    for(const auto &g:k) {
      Q a,b,c;
      for(int row=0;row<3;row++) {
        Q u,v;
        for(int j=0;j<3;j++) {u+=g.inverse[3*row+j]*(Q(source[j])-Q(g.center[j]));v+=g.inverse[3*row+j]*(Q(receiver[j])-Q(source[j]));}
        a+=v*v;b+=2*u*v;c+=u*u;
      }
      Coeff p;p.a=R(a);p.constant=R(c);p.stationary=a!=0;
      if(a!=0) {Q m=-b/(2*a),minimum=c-b*b/(4*a);if(minimum<0)throw std::runtime_error("Invalid exact quadratic");p.mu=R(m);p.minimum=R(minimum);}
      coefficients.push_back(std::move(p));stats.add("checker_exact_coefficient_builds");
    }
  }
  Values evaluate(double lo,double hi,bool derivatives) {
    Values out;R t(lo,hi);
    for(std::size_t j=0;j<coefficients.size();j++) {
      const auto &p=coefficients[j];R dot,q=p.constant;
      if(p.stationary){R delta=minus(t,p.mu);q=plus(times(p.a,squared(delta)),p.minimum);dot=times(p.a,delta);}
      R g=times(R(kernels[j].weight),exponential(q,stats));out.f=plus(out.f,g);
      if(derivatives) {
        out.first=plus(out.first,times(negative(dot),g));
        out.second=plus(out.second,times(minus(squared(dot),p.a),g));
      }
    }
    stats.add("checker_function_bounds");if(derivatives)stats.add("checker_derivative_bounds");return out;
  }
  MP local_upper(double lo,double hi,const Values &range) {
    double mid=lo+(hi-lo)/2;auto point=evaluate(mid,mid,true);
    R left=minus(R(mid),R(lo)),right=minus(R(hi),R(mid));MP h(precision),d(precision),k(precision),u(precision),v(precision),error(precision);
    mpfr_set(h.v,mpfr_cmp(left.h.v,right.h.v)>0?left.h.v:right.h.v,MPFR_RNDU);
    mpfr_abs(d.v,mpfr_cmpabs(point.first.l.v,point.first.h.v)>0?point.first.l.v:point.first.h.v,MPFR_RNDU);
    mpfr_set(k.v,range.second.h.v,MPFR_RNDU);if(mpfr_sgn(k.v)<0)mpfr_set_zero(k.v,0);
    mpfr_sqr(v.v,h.v,MPFR_RNDU);mpfr_mul(v.v,v.v,k.v,MPFR_RNDU);mpfr_div_2ui(v.v,v.v,1,MPFR_RNDU);
    mpfr_mul(error.v,d.v,h.v,MPFR_RNDU);mpfr_add(error.v,error.v,v.v,MPFR_RNDU);mpfr_add(u.v,point.f.h.v,error.v,MPFR_RNDU);
    if(mpfr_sgn(range.second.h.v)<0) {
      mpfr_neg(k.v,range.second.h.v,MPFR_RNDN);mpfr_mul_2ui(k.v,k.v,1,MPFR_RNDD);
      mpfr_sqr(v.v,d.v,MPFR_RNDU);mpfr_div(v.v,v.v,k.v,MPFR_RNDU);mpfr_add(v.v,point.f.h.v,v.v,MPFR_RNDU);
      if(mpfr_cmp(v.v,u.v)<0)mpfr_set(u.v,v.v,MPFR_RNDU);
    }
    stats.add("checker_local_taylor_bounds");return u;
  }
};
bool sign_excluded(const R&r){return mpfr_sgn(r.l.v)>0||mpfr_sgn(r.h.v)<0;}
} // namespace
struct Checker::Impl {
  Inputs input;std::vector<Canonical> kernels;
  explicit Impl(const Inputs &in):input(in) {
    if(in.centers.empty()||in.centers.size()>1024||in.centers.size()!=in.matrices.size()||in.centers.size()!=in.weights.size()||!std::isfinite(in.threshold)||(!in.exponential&&in.threshold<=0))throw std::invalid_argument("Invalid checker canonical inputs");
    for(std::size_t j=0;j<in.centers.size();j++) {
      if(!finite(in.centers[j])||!std::isfinite(in.weights[j])||in.weights[j]<=0||!std::all_of(in.matrices[j].begin(),in.matrices[j].end(),[](double d){return std::isfinite(d);}))throw std::invalid_argument("Invalid checker kernel");
      kernels.push_back({in.centers[j],cofactors(in.matrices[j]),in.weights[j]});
    }
  }
};
Checker::Checker(const Inputs &in):p(std::make_shared<Impl>(in)){}
CheckResult Checker::check(P source,P receiver,const std::vector<Leaf>&leaves,int bits,double seconds)const {
  CheckResult out;auto start=Clock::now();auto reject=[&](const std::string &reason){out.reason=reason;out.seconds=elapsed(start);return out;};
  require_environment();if(bits<80||bits>1024||!std::isfinite(seconds)||seconds<=0)return reject("unsupported_checker_budget");
  if(!finite(source)||!finite(receiver))return reject("nonfinite_segment");
  if(leaves.empty()||leaves.size()>8192)return reject("missing_or_excessive_partition");
  std::vector<Leaf> sorted=leaves;
  for(const auto &v:sorted) {
    if(!std::isfinite(v.lo)||!std::isfinite(v.hi)||!std::isfinite(v.upper)||v.lo<0||v.hi>1||v.lo>=v.hi||v.upper<0)return reject("invalid_interval_or_bound");
    if(v.op<0||v.op>4)return reject("unsupported_proof_operation");
  }
  std::sort(sorted.begin(),sorted.end(),[](auto a,auto b){return a.lo==b.lo?a.hi<b.hi:a.lo<b.lo;});
  double cursor=0;
  for(const auto &v:sorted){if(v.lo>cursor)return reject("uncovered_parameter_gap");cursor=std::max(cursor,v.hi);}
  if(sorted.front().lo!=0||cursor!=1)return reject("missing_closed_endpoint_coverage");
  precision=bits;R tau(p->input.threshold);
  if(p->input.exponential){mpfr_exp(tau.l.v,tau.l.v,MPFR_RNDD);mpfr_exp(tau.h.v,tau.h.v,MPFR_RNDU);}
  CheckRay ray(p->kernels,source,receiver,out.stats);out.preparation_seconds=elapsed(start);
  try {
    for(const auto &v:sorted) {
      if(elapsed(start)>seconds)return reject("checker_time_budget");
      MP upper(bits);auto full=ray.evaluate(v.lo,v.hi,v.op!=0);
      if(v.op==0)mpfr_set(upper.v,full.f.h.v,MPFR_RNDU);
      else if(v.op==1||v.op==2) {
        bool positive=v.op==1;
        if(positive?mpfr_sgn(full.first.l.v)<=0:mpfr_sgn(full.first.h.v)>=0)return reject("unproved_monotonicity");
        auto endpoint=ray.evaluate(positive?v.hi:v.lo,positive?v.hi:v.lo,false);mpfr_set(upper.v,endpoint.f.h.v,MPFR_RNDU);
      } else if(v.op==3)upper=ray.local_upper(v.lo,v.hi,full);
      else {
        if(!sign_excluded(full.second))return reject("unproved_stationary_exclusion");
        double mid=v.lo+(v.hi-v.lo)/2;auto point=ray.evaluate(mid,mid,true);R n=minus(R(mid),quotient(point.first,full.second));
        MP nl(bits),nh(bits);mpfr_set(nl.v,n.l.v,MPFR_RNDD);mpfr_set(nh.v,n.h.v,MPFR_RNDU);
        if(mpfr_cmp_d(nl.v,v.lo)<0)mpfr_set_d(nl.v,v.lo,MPFR_RNDN);
        if(mpfr_cmp_d(nh.v,v.hi)>0)mpfr_set_d(nh.v,v.hi,MPFR_RNDN);
        bool empty=mpfr_cmp(nl.v,nh.v)>0;
        if(v.empty!=empty)return reject("incorrect_stationary_empty_claim");
        auto left=ray.evaluate(v.lo,v.lo,false),right=ray.evaluate(v.hi,v.hi,false);
        mpfr_set(upper.v,mpfr_cmp(left.f.h.v,right.f.h.v)>0?left.f.h.v:right.f.h.v,MPFR_RNDU);
        if(!empty) {
          if(!std::isfinite(v.jlo)||!std::isfinite(v.jhi)||v.jlo<v.lo||v.jhi>v.hi||v.jlo>v.jhi||mpfr_cmp_d(nl.v,v.jlo)<0||mpfr_cmp_d(nh.v,v.jhi)>0)return reject("stationary_region_not_enclosed");
          if(v.inner_op!=0&&v.inner_op!=3)return reject("unsupported_inner_operation");
          auto inner=ray.evaluate(v.jlo,v.jhi,v.inner_op==3);MP inside(bits);
          if(v.inner_op==0)mpfr_set(inside.v,inner.f.h.v,MPFR_RNDU);else inside=ray.local_upper(v.jlo,v.jhi,inner);
          if(mpfr_cmp(inside.v,upper.v)>0)mpfr_set(upper.v,inside.v,MPFR_RNDU);
        }
      }
      if(!mpfr_number_p(upper.v)||mpfr_cmp_d(upper.v,v.upper)>0)return reject("unverified_serialized_upper_bound");
      // Both the reconstructed enclosure and the claimed bound must be strict.
      if(mpfr_cmp(upper.v,tau.l.v)>=0||mpfr_cmp_d(tau.l.v,v.upper)<=0)return reject("not_strictly_below_threshold");
      out.stats.add("checked_intervals");
    }
  }catch(const std::exception&e){return reject(std::string("unresolved_checker_operation:")+e.what());}
  out.accepted=true;out.reason="complete_closed_cover_independently_checked";out.seconds=elapsed(start);return out;
}
} // namespace b2max
