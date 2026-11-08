#pragma once
#include "interval.hpp"

namespace b2 {
struct HitWitness {
  const char *kind = "none";
  double t = NAN, lower = 0, margin = 0;

  void record(const char *reason, double parameter, double bound, I tau) {
    kind = reason;
    t = parameter;
    lower = std::isfinite(bound) ? bound : 0.;
    if (t >= 0 && t <= 1 && std::isfinite(tau.hi) && lower > tau.hi)
      margin = std::max(0., sub(I(lower), I(tau.hi)).lo);
  }
};

struct HitTrial {
  int direction = 0, steps = 0;
  bool gradient = false, direct = false, controls = false;
  bool cover = false, proxy = false, section = false;
  double gradient_seconds = 0, direct_seconds = 0, cover_seconds = 0,
         proxy_seconds = 0, section_seconds = 0;
  std::uint64_t gradient_exp = 0, direct_exp = 0;
};

struct HitAudit {
  Decision decision;
  HitWitness witness;
  double scalar_seconds = 0, capture_seconds = 0, tau_upper = 0;
  std::vector<HitTrial> trials;
};
} // namespace b2
