#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <map>
#include <memory>
#include <mpfr.h>
#include <string>
#include <vector>

namespace b2 {
using Q = mpq_class;
using P = std::array<double, 3>;
using Matrix = std::array<double, 9>;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}
struct MP {
  mpfr_t v;
  explicit MP(int bits = 192) {
    mpfr_init2(v, bits);
    mpfr_set_zero(v, 0);
  }
  MP(const MP &o) {
    mpfr_init2(v, mpfr_get_prec(o.v));
    mpfr_set(v, o.v, MPFR_RNDN);
  }
  MP &operator=(const MP &o) {
    if (this != &o) {
      mpfr_set_prec(v, mpfr_get_prec(o.v));
      mpfr_set(v, o.v, MPFR_RNDN);
    }
    return *this;
  }
  ~MP() { mpfr_clear(v); }
};
struct Inputs {
  std::vector<P> centers;
  std::vector<Matrix> matrices;
  std::vector<double> weights;
  double threshold;
  bool exponential;
};
struct Options {
  bool coefficients = true, filters = true, groups = false, regions = true,
       fallback = true, taylor = true, tail_bounds = true,
       inherited_regions = true;
  int max_nodes = 4096, max_depth = 40, precision = 192;
  int exact_depth = 20, taylor_depth = 4;
  double query_seconds = 2.;
  void validate() const;
};
struct Stats {
  std::map<std::string, std::uint64_t> n;
  void add(const std::string &s, std::uint64_t count = 1) { n[s] += count; }
  void maximum(const std::string &s, std::uint64_t value) {
    n[s] = std::max(n[s], value);
  }
};
// 0=MISS/FREE, 1=HIT/SOLID, -1=UNKNOWN, -2=INVALID. Reason is never a physical
// label.
struct Decision {
  int status = -1;
  std::string reason = "unresolved";
  int nodes = 0, depth = 0;
};
struct Field {
  std::vector<std::uint8_t> response, solid, valid;
  int resolution;
  bool complete = false;
  std::string reason;
  double seconds = 0;
  Stats stats;
};
class Primary {
  struct Impl;
  std::shared_ptr<const Impl> impl;

public:
  explicit Primary(const Inputs &);
  Decision point(P, const Options &, Stats &) const;
  Decision ray(P, P, const Options &, Stats &) const;
  Decision box(P, P, const Options &, Stats &) const;
  std::pair<int, int> certificate(P, P, P, const Options &, Stats &,
                                  int known_y = -1, int known_m = -1) const;
  Field field(P, int, double, const Options &, bool scalar = false) const;
  std::map<std::string, double> info() const;
};
class Independent {
  struct Impl;
  std::shared_ptr<const Impl> impl;

public:
  explicit Independent(const Inputs &);
  Decision point(P, const Options &, Stats &) const;
  Decision ray(P, P, const Options &, Stats &) const;
  Decision box(P, P, const Options &, Stats &) const;
  std::pair<int, int> certificate(P, P, P, const Options &, Stats &,
                                  int known_y = -1, int known_m = -1) const;
  Field field(P, int, double, const Options &, bool scalar = false) const;
};
bool finite(P);
void require_environment();
} // namespace b2
