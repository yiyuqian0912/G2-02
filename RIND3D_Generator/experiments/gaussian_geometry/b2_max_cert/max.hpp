#pragma once
#include "common.hpp"

namespace b2max {
using namespace b2;
struct Settings {
  int method=3, max_nodes=4096, max_depth=48;
  double seconds=2.;
  bool fallback=false,endpoint_probes=true;
  void validate() const;
};
// Closed intervals with binary64 (therefore lossless dyadic) endpoints.
// op: 0 natural, 1 increasing, 2 decreasing, 3 local Taylor, 4 stationary.
struct Leaf {
  double lo=0,hi=1,upper=INFINITY,jlo=0,jhi=0;
  int op=0,inner_op=0;
  bool empty=false;
};
struct Result {
  Decision decision;
  std::vector<Leaf> leaves;
  Stats stats;
  double seconds=0,preparation_seconds=0;
  bool fallback_used=false;
};
class Solver {
  struct Impl;
  std::shared_ptr<const Impl> p;
public:
  explicit Solver(const Inputs &);
  Result solve(P source,P receiver,const Settings &,bool proof) const;
  std::map<std::string,std::vector<double>> bounds(P,P,double,double) const;
};
struct CheckResult {
  bool accepted=false;
  std::string reason;
  Stats stats;
  double seconds=0,preparation_seconds=0;
};
class Checker {
  struct Impl;
  std::shared_ptr<const Impl> p;
public:
  explicit Checker(const Inputs &);
  CheckResult check(P,P,const std::vector<Leaf>&,int precision=128,double seconds=2.) const;
};
} // namespace b2max
