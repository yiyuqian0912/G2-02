#pragma once
#include "common.hpp"
#include <bit>
#include <cfenv>
#include <limits>
#include <xmmintrin.h>

namespace b2 {
// Outward binary64 algebra follows the unchanged production interval policy.
// No libm exp/log/sqrt is used for a physical bound.
struct I {
  double lo, hi;
  I(double x = 0) : lo(x), hi(x) {}
  I(double a, double b) : lo(a), hi(b) {}
};
inline double step(double x, bool up) {
  if (std::isnan(x) || x == (up ? INFINITY : -INFINITY))
    return x;
  if (x == 0)
    return up ? std::numeric_limits<double>::denorm_min()
              : -std::numeric_limits<double>::denorm_min();
  auto v = std::bit_cast<std::uint64_t>(x);
  v += (up == (x > 0)) ? 1 : std::uint64_t(-1);
  return std::bit_cast<double>(v);
}
inline I whole() { return {-INFINITY, INFINITY}; }
inline I add(I a, I b) {
  double l = a.lo + b.lo, h = a.hi + b.hi;
  return std::isnan(l) || std::isnan(h) ? whole()
                                        : I(step(l, false), step(h, true));
}
inline I neg(I a) { return {-a.hi, -a.lo}; }
inline I sub(I a, I b) { return add(a, neg(b)); }
inline I mul(I a, I b) {
  if ((a.lo == 0 && a.hi == 0) || (b.lo == 0 && b.hi == 0))
    return I(0);
  double v[] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  for (double x : v)
    if (std::isnan(x))
      return whole();
  return {step(*std::min_element(v, v + 4), false),
          step(*std::max_element(v, v + 4), true)};
}
inline I divi(I a, I b) {
  if (b.lo <= 0 && b.hi >= 0)
    return whole();
  return mul(a, {step(1 / b.hi, false), step(1 / b.lo, true)});
}
inline I square(I a) {
  double h = std::max(std::abs(a.lo), std::abs(a.hi));
  double l =
      a.lo <= 0 && a.hi >= 0 ? 0 : std::min(std::abs(a.lo), std::abs(a.hi));
  return {l == 0 ? 0 : std::max(0., step(l * l, false)), step(h * h, true)};
}
inline I enclose(const Q &q) {
  double x = q.get_d();
  if (!std::isfinite(x))
    return whole();
  double l = x, h = x;
  while (Q(l) > q)
    l = step(l, false);
  while (Q(h) < q)
    h = step(h, true);
  return {l, h};
}
struct Exp {
  MP x;
  explicit Exp(int bits = 53) : x(bits) {}
  I value(I q, Stats &s) {
    // True q is nonnegative. ±infinity endpoints give conservative limits.
    double lo = std::max(0., q.lo), hi = std::max(lo, q.hi);
    s.add("exp_endpoints", 2);
    mpfr_set_d(x.v, hi, MPFR_RNDN);
    mpfr_div_2ui(x.v, x.v, 1, MPFR_RNDN);
    mpfr_neg(x.v, x.v, MPFR_RNDN);
    mpfr_exp(x.v, x.v, MPFR_RNDD);
    double a = mpfr_get_d(x.v, MPFR_RNDD);
    mpfr_set_d(x.v, lo, MPFR_RNDN);
    mpfr_div_2ui(x.v, x.v, 1, MPFR_RNDN);
    mpfr_neg(x.v, x.v, MPFR_RNDN);
    mpfr_exp(x.v, x.v, MPFR_RNDU);
    double b = mpfr_get_d(x.v, MPFR_RNDU);
    if (std::isnan(a) || std::isnan(b))
      return {0, INFINITY};
    return {a, b};
  }
};
inline I threshold(const Inputs &i, int bits = 192) {
  if (!i.exponential)
    return I(i.threshold);
  MP l(bits), h(bits);
  mpfr_set_d(l.v, i.threshold, MPFR_RNDN);
  mpfr_set(h.v, l.v, MPFR_RNDN);
  mpfr_exp(l.v, l.v, MPFR_RNDD);
  mpfr_exp(h.v, h.v, MPFR_RNDU);
  return {mpfr_get_d(l.v, MPFR_RNDD), mpfr_get_d(h.v, MPFR_RNDU)};
}
inline int compare(I value, I tau) {
  if (value.lo >= tau.hi)
    return 1;
  if (value.hi < tau.lo)
    return 0;
  return -1;
}
} // namespace b2
