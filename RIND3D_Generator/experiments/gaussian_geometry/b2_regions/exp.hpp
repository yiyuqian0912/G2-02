#pragma once
#include "interval.hpp"
namespace b2r {
using namespace b2;
// Only outward arithmetic and an analytic remainder are used. No libm exp.
inline I negative_half_exp(double q, Stats &s) {
  if (q == INFINITY)
    return I(0);
  if (q == 0)
    return I(1);
  if (!(q >= 0))
    return {0, INFINITY};
  if (q >= 2148) { // e>2, so exp(-q/2)<2^-1074, with positive remainder.
    s.add("fast_exp_positive_underflow_remainders");
    return {0, std::numeric_limits<double>::denorm_min()};
  }
  double v = q * .5;
  int m = 0;
  while (v > .03125) {
    v *= .5;
    ++m;
  }
  I u = mul(I(q), I(std::ldexp(1., -m - 1)));
  u.lo = std::max(0., u.lo);
  if (u.hi > .0625)
    return {0, 1};
  I term(1), sum(1);
  for (int j = 1; j <= 12; ++j) {
    term = divi(mul(term, u), I(j));
    sum = add(sum, term);
  }
  // Positive series: remaining terms have ratio at most u/(12+2).
  I next = divi(mul(term, u), I(13));
  I tail = divi(next, sub(I(1), divi(u, I(14))));
  I positive(std::max(1., sum.lo), add(sum, I(0, tail.hi)).hi);
  I value = divi(I(1), positive);
  for (int j = 0; j < m; ++j)
    value = square(value);
  value.lo = std::max(0., value.lo);
  value.hi = std::min(1., value.hi);
  s.add("fast_exp_polynomials");
  s.add("fast_exp_squarings", m);
  return value;
}
inline I exp_q(I q, bool fast, Exp &mpfr, Stats &s) {
  q.lo = std::max(0., q.lo);
  q.hi = std::max(q.lo, q.hi);
  if (!fast)
    return mpfr.value(q, s);
  I a = negative_half_exp(q.hi, s), b = negative_half_exp(q.lo, s);
  return {a.lo, b.hi};
}
} // namespace b2r
