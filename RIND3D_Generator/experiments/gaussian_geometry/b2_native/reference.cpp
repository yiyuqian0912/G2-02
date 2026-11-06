#include "common.hpp"
#include "field.hpp"
#include <deque>
#include <numeric>
#include <stdexcept>

namespace b2 {
namespace {
constexpr int bits = 256;
struct Bound {
  MP l, h;
  Bound() : l(bits), h(bits) {}
  explicit Bound(const Q &q) : Bound() {
    mpfr_set_q(l.v, q.get_mpq_t(), MPFR_RNDD);
    mpfr_set_q(h.v, q.get_mpq_t(), MPFR_RNDU);
  }
};
using RInv = std::array<Q, 9>;
RInv cofactor(const Matrix &m) {
  Q a(m[0]), b(m[1]), c(m[2]), d(m[3]), e(m[4]), f(m[5]), g(m[6]), h(m[7]),
      i(m[8]);
  Q det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (det == 0)
    throw std::invalid_argument("Independent singular M");
  return {Q((e * i - f * h) / det), Q((c * h - b * i) / det),
          Q((b * f - c * e) / det), Q((f * g - d * i) / det),
          Q((a * i - c * g) / det), Q((c * d - a * f) / det),
          Q((d * h - e * g) / det), Q((b * g - a * h) / det),
          Q((a * e - b * d) / det)};
}
struct RefKernel {
  P c;
  RInv inv;
  double w;
  std::array<Bound, 9> bounds;
};
struct Line {
  std::array<Bound, 3> offset, slope;
  Bound derivative_offset, curvature;
};
// Only RAII storage/parameters are shared with the primary. These intervals,
// natural bounds, breadth-first ray decisions and shadow sections are separate.
struct Scratch {
  MP ql, qh, lo, hi, x, y, t, suml, sumh, dl, dh, gl, gh;
  Scratch()
      : ql(bits), qh(bits), lo(bits), hi(bits), x(bits), y(bits), t(bits),
        suml(bits), sumh(bits), dl(bits), dh(bits), gl(bits), gh(bits) {};
};
// The inverse is still derived by independent exact cofactors. Only reusable
// 256-bit coefficient enclosures are cached; no primary geometry bound is used.
void affine(const RefKernel &k, int row, P low, P high, P origin, Scratch &z) {
  mpfr_set_zero(z.lo.v, 0);
  mpfr_set_zero(z.hi.v, 0);
  for (int a = 0; a < 3; a++) {
    mpfr_set_d(z.dl.v, low[a], MPFR_RNDN);
    mpfr_sub_d(z.dl.v, z.dl.v, origin[a], MPFR_RNDD);
    mpfr_set_d(z.dh.v, high[a], MPFR_RNDN);
    mpfr_sub_d(z.dh.v, z.dh.v, origin[a], MPFR_RNDU);
    const auto &coefficient = k.bounds[row * 3 + a];
    bool negative = k.inv[row * 3 + a] < 0;
    const auto &lower_delta = negative ? z.dh : z.dl;
    const auto &upper_delta = negative ? z.dl : z.dh;
    const auto &lower_coefficient =
        mpfr_sgn(lower_delta.v) >= 0 ? coefficient.l : coefficient.h;
    const auto &upper_coefficient =
        mpfr_sgn(upper_delta.v) >= 0 ? coefficient.h : coefficient.l;
    mpfr_mul(z.x.v, lower_coefficient.v, lower_delta.v, MPFR_RNDD);
    mpfr_mul(z.y.v, upper_coefficient.v, upper_delta.v, MPFR_RNDU);
    mpfr_add(z.lo.v, z.lo.v, z.x.v, MPFR_RNDD);
    mpfr_add(z.hi.v, z.hi.v, z.y.v, MPFR_RNDU);
  }
}
void add_square(Scratch &z) {
  if (mpfr_sgn(z.lo.v) <= 0 && mpfr_sgn(z.hi.v) >= 0)
    mpfr_set_zero(z.x.v, 0);
  else if (mpfr_sgn(z.lo.v) > 0)
    mpfr_sqr(z.x.v, z.lo.v, MPFR_RNDD);
  else
    mpfr_sqr(z.x.v, z.hi.v, MPFR_RNDD);
  mpfr_add(z.ql.v, z.ql.v, z.x.v, MPFR_RNDD);
  if (mpfr_cmpabs(z.lo.v, z.hi.v) > 0)
    mpfr_sqr(z.y.v, z.lo.v, MPFR_RNDU);
  else
    mpfr_sqr(z.y.v, z.hi.v, MPFR_RNDU);
  mpfr_add(z.qh.v, z.qh.v, z.y.v, MPFR_RNDU);
}
void accumulate(Scratch &z, double weight, Stats &s, bool tails) {
  if (tails && mpfr_cmp_ui(z.ql.v, 128) >= 0) {
    // e>2: exp(-q/2) <= 2^-floor(q_lower/2). Retain a positive
    // upper remainder, even below binary64's representable range.
    unsigned long exponent = 1074;
    if (mpfr_cmp_ui(z.ql.v, 2148) < 0) {
      mpfr_div_2ui(z.x.v, z.ql.v, 1, MPFR_RNDD);
      exponent = mpfr_get_ui(z.x.v, MPFR_RNDD);
    }
    mpfr_set_zero(z.gl.v, 0);
    mpfr_set_ui(z.gh.v, 1, MPFR_RNDN);
    mpfr_div_2ui(z.gh.v, z.gh.v, exponent, MPFR_RNDU);
    mpfr_mul_d(z.gh.v, z.gh.v, weight, MPFR_RNDU);
    mpfr_add(z.sumh.v, z.sumh.v, z.gh.v, MPFR_RNDU);
    s.add("reference_positive_tail_remainders");
    s.add("reference_gaussian_terms");
    return;
  }
  mpfr_neg(z.x.v, z.qh.v, MPFR_RNDN);
  mpfr_div_2ui(z.x.v, z.x.v, 1, MPFR_RNDN);
  mpfr_exp(z.x.v, z.x.v, MPFR_RNDD);
  mpfr_mul_d(z.x.v, z.x.v, weight, MPFR_RNDD);
  mpfr_set(z.gl.v, z.x.v, MPFR_RNDD);
  mpfr_add(z.suml.v, z.suml.v, z.x.v, MPFR_RNDD);
  mpfr_neg(z.x.v, z.ql.v, MPFR_RNDN);
  mpfr_div_2ui(z.x.v, z.x.v, 1, MPFR_RNDN);
  mpfr_exp(z.x.v, z.x.v, MPFR_RNDU);
  mpfr_mul_d(z.x.v, z.x.v, weight, MPFR_RNDU);
  mpfr_set(z.gh.v, z.x.v, MPFR_RNDU);
  mpfr_add(z.sumh.v, z.sumh.v, z.x.v, MPFR_RNDU);
  s.add("reference_exp_endpoints", 2);
  s.add("reference_gaussian_terms");
}
} // namespace
struct Independent::Impl {
  Inputs input;
  std::vector<RefKernel> kernels;
  Bound tau;
  bool analytic = false;
  explicit Impl(const Inputs &in) : input(in) {
    if (in.centers.empty() || in.centers.size() > 1024 ||
        in.centers.size() != in.matrices.size() ||
        in.centers.size() != in.weights.size() ||
        !std::isfinite(in.threshold) || (!in.exponential && in.threshold <= 0))
      throw std::invalid_argument("Invalid independent parameters");
    mpfr_set_d(tau.l.v, in.threshold, MPFR_RNDN);
    mpfr_set(tau.h.v, tau.l.v, MPFR_RNDN);
    if (in.exponential) {
      mpfr_exp(tau.l.v, tau.l.v, MPFR_RNDD);
      mpfr_exp(tau.h.v, tau.h.v, MPFR_RNDU);
    }
    Q sum;
    for (std::size_t i = 0; i < in.centers.size(); i++) {
      if (!finite(in.centers[i]) || !std::isfinite(in.weights[i]) ||
          in.weights[i] <= 0 ||
          !std::all_of(in.matrices[i].begin(), in.matrices[i].end(),
                       [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("Invalid independent kernel");
      RefKernel k;
      k.c = in.centers[i];
      k.inv = cofactor(in.matrices[i]);
      k.w = in.weights[i];
      for (int j = 0; j < 9; j++)
        k.bounds[j] = Bound(k.inv[j]);
      kernels.push_back(k);
      sum += Q(k.w);
    }
    analytic = in.exponential && sum == 1;
    for (std::size_t i = 0; i < in.centers.size(); i++)
      if (in.centers[i] != in.centers[0] || in.matrices[i] != in.matrices[0])
        analytic = false;
  }
  int decide(const Scratch &s) const {
    if (mpfr_cmp(s.suml.v, tau.h.v) >= 0)
      return 1;
    if (mpfr_cmp(s.sumh.v, tau.l.v) < 0)
      return 0;
    return -1;
  }
  int spatial(P low, P high, Stats &s, bool tails) const {
    Scratch z;
    for (const auto &k : kernels) {
      mpfr_set_zero(z.ql.v, 0);
      mpfr_set_zero(z.qh.v, 0);
      for (int row = 0; row < 3; row++) {
        affine(k, row, low, high, k.c, z);
        add_square(z);
      }
      accumulate(z, k.w, s, tails);
    }
    return decide(z);
  }
  std::vector<Line> lines(P source, P receiver, Stats &s) const {
    std::vector<Line> out;
    out.reserve(kernels.size());
    Scratch z;
    for (auto &k : kernels) {
      Line line;
      for (int r = 0; r < 3; r++) {
        affine(k, r, source, source, k.c, z);
        mpfr_set(line.offset[r].l.v, z.lo.v, MPFR_RNDD);
        mpfr_set(line.offset[r].h.v, z.hi.v, MPFR_RNDU);
        affine(k, r, receiver, receiver, source, z);
        mpfr_set(line.slope[r].l.v, z.lo.v, MPFR_RNDD);
        mpfr_set(line.slope[r].h.v, z.hi.v, MPFR_RNDU);
      }
      out.push_back(line);
      s.add("reference_line_builds");
    }
    return out;
  }
  int natural(const std::vector<Line> &line, double low, double high, Stats &s,
              bool tails) const {
    Scratch z;
    for (std::size_t i = 0; i < kernels.size(); i++) {
      mpfr_set_zero(z.ql.v, 0);
      mpfr_set_zero(z.qh.v, 0);
      for (int r = 0; r < 3; r++) {
        const auto &l = line[i];
        mpfr_mul_d(z.lo.v, l.slope[r].l.v,
                   mpfr_sgn(l.slope[r].l.v) >= 0 ? low : high, MPFR_RNDD);
        mpfr_add(z.lo.v, z.lo.v, l.offset[r].l.v, MPFR_RNDD);
        mpfr_mul_d(z.hi.v, l.slope[r].h.v,
                   mpfr_sgn(l.slope[r].h.v) >= 0 ? high : low, MPFR_RNDU);
        mpfr_add(z.hi.v, z.hi.v, l.offset[r].h.v, MPFR_RNDU);
        add_square(z);
      }
      accumulate(z, kernels[i].w, s, tails);
    }
    return decide(z);
  }
  void derivatives(std::vector<Line> &line, P source, P receiver, MP &second,
                   Stats &s) const {
    mpfr_set_zero(second.v, 0);
    MP tmp(bits);
    for (std::size_t i = 0; i < kernels.size(); i++) {
      const auto &k = kernels[i];
      Q a, dot;
      for (int row = 0; row < 3; row++) {
        Q u, v;
        for (int axis = 0; axis < 3; axis++) {
          u += k.inv[row * 3 + axis] * (Q(source[axis]) - Q(k.c[axis]));
          v += k.inv[row * 3 + axis] * (Q(receiver[axis]) - Q(source[axis]));
        }
        a += v * v;
        dot += u * v;
      }
      line[i].derivative_offset = Bound(dot);
      line[i].curvature = Bound(a);
      mpfr_mul_d(tmp.v, line[i].curvature.h.v, k.w, MPFR_RNDU);
      mpfr_mul_2ui(tmp.v, tmp.v, 1, MPFR_RNDU);
      mpfr_add(second.v, second.v, tmp.v, MPFR_RNDU);
    }
    s.add("reference_taylor_preparations");
  }
  int taylor(const std::vector<Line> &line, double low, double high,
             const MP &second, Stats &s, bool tails) const {
    Scratch z;
    MP dl(bits), dh(bits), dotlo(bits), dothi(bits), tmp(bits), error(bits),
        other(bits);
    double mid = (low + high) / 2, half = (high - low) / 2;
    for (std::size_t i = 0; i < kernels.size(); i++) {
      mpfr_set_zero(z.ql.v, 0);
      mpfr_set_zero(z.qh.v, 0);
      for (int row = 0; row < 3; row++) {
        mpfr_mul_d(z.lo.v, line[i].slope[row].l.v, mid, MPFR_RNDD);
        mpfr_add(z.lo.v, z.lo.v, line[i].offset[row].l.v, MPFR_RNDD);
        mpfr_mul_d(z.hi.v, line[i].slope[row].h.v, mid, MPFR_RNDU);
        mpfr_add(z.hi.v, z.hi.v, line[i].offset[row].h.v, MPFR_RNDU);
        add_square(z);
      }
      accumulate(z, kernels[i].w, s, tails);
      mpfr_mul_d(dotlo.v, line[i].curvature.l.v, mid, MPFR_RNDD);
      mpfr_add(dotlo.v, dotlo.v, line[i].derivative_offset.l.v, MPFR_RNDD);
      mpfr_mul_d(dothi.v, line[i].curvature.h.v, mid, MPFR_RNDU);
      mpfr_add(dothi.v, dothi.v, line[i].derivative_offset.h.v, MPFR_RNDU);
      mpfr_neg(tmp.v, dothi.v, MPFR_RNDN);
      mpfr_mul(tmp.v, tmp.v, mpfr_sgn(tmp.v) >= 0 ? z.gl.v : z.gh.v, MPFR_RNDD);
      mpfr_add(dl.v, dl.v, tmp.v, MPFR_RNDD);
      mpfr_neg(tmp.v, dotlo.v, MPFR_RNDN);
      mpfr_mul(tmp.v, tmp.v, mpfr_sgn(tmp.v) >= 0 ? z.gh.v : z.gl.v, MPFR_RNDU);
      mpfr_add(dh.v, dh.v, tmp.v, MPFR_RNDU);
    }
    if (decide(z) == 1)
      return 1;
    mpfr_abs(tmp.v, dl.v, MPFR_RNDU);
    mpfr_abs(other.v, dh.v, MPFR_RNDU);
    if (mpfr_cmp(tmp.v, other.v) < 0)
      mpfr_set(tmp.v, other.v, MPFR_RNDU);
    mpfr_mul_d(error.v, tmp.v, half, MPFR_RNDU);
    mpfr_set_d(tmp.v, half, MPFR_RNDN);
    mpfr_sqr(tmp.v, tmp.v, MPFR_RNDU);
    mpfr_mul(tmp.v, tmp.v, second.v, MPFR_RNDU);
    mpfr_div_2ui(tmp.v, tmp.v, 1, MPFR_RNDU);
    mpfr_add(error.v, error.v, tmp.v, MPFR_RNDU);
    mpfr_add(tmp.v, z.sumh.v, error.v, MPFR_RNDU);
    s.add("reference_taylor_tests");
    if (mpfr_cmp(tmp.v, tau.l.v) < 0) {
      s.add("reference_taylor_interval_free");
      return 0;
    }
    return -1;
  }
  Decision analytic_ray(P source, P receiver) const {
    auto &k = kernels[0];
    Q a, b, c;
    for (int r = 0; r < 3; r++) {
      Q u, v;
      for (int j = 0; j < 3; j++) {
        u += k.inv[r * 3 + j] * (Q(source[j]) - Q(k.c[j]));
        v += k.inv[r * 3 + j] * (Q(receiver[j]) - Q(source[j]));
      }
      a += v * v;
      b += 2 * u * v;
      c += u * u;
    }
    c += 2 * Q(input.threshold);
    bool hit = c <= 0 || a + b + c <= 0;
    if (a > 0 && b < 0 && b > -2 * a && b * b - 4 * a * c >= 0)
      hit = true;
    return {hit ? 1 : 0, "independent_exact_discriminant"};
  }
};
Independent::Independent(const Inputs &i) : impl(std::make_shared<Impl>(i)) {}
Decision Independent::point(P p, const Options &o, Stats &s) const {
  s.add("reference_point_queries");
  if (!finite(p))
    return {-2, "nonfinite_point"};
  if (impl->analytic)
    return impl->analytic_ray(p, p);
  int state = impl->spatial(p, p, s, o.tail_bounds);
  if (state < 0 && o.tail_bounds)
    state = impl->spatial(p, p, s, false);
  return {state,
          state < 0 ? "reference_point_precision" : "independent_MPFR_point"};
}
Decision Independent::box(P lo, P hi, const Options &o, Stats &s) const {
  s.add("reference_region_queries");
  if (!finite(lo) || !finite(hi))
    return {-2, "nonfinite_box"};
  for (int a = 0; a < 3; a++)
    if (lo[a] > hi[a])
      return {-2, "reversed_box"};
  int result = impl->spatial(lo, hi, s, o.tail_bounds);
  if (result < 0 && lo == hi && o.tail_bounds)
    result = impl->spatial(lo, hi, s, false);
  if (result >= 0)
    s.add(result ? "reference_solid_boxes" : "reference_free_boxes");
  return {result,
          result < 0 ? "mixed_or_unresolved" : "independent_natural_MPFR_box"};
}
Decision Independent::ray(P source, P receiver, const Options &o,
                          Stats &s) const {
  s.add("reference_ray_queries");
  if (!finite(source) || !finite(receiver))
    return {-2, "nonfinite_segment"};
  if (impl->analytic)
    return impl->analytic_ray(source, receiver);
  auto line = impl->lines(source, receiver, s);
  struct Node {
    double l, h;
    int d;
  };
  std::deque<Node> queue{{0, 1, 0}};
  auto start = Clock::now();
  int nodes = 0, depth = 0;
  bool derivatives_ready = false;
  MP second(bits);
  while (!queue.empty()) {
    if (nodes >= o.max_nodes || elapsed(start) > o.query_seconds) {
      s.add("reference_unknown_rays");
      return {-1,
              nodes >= o.max_nodes ? "reference_node_budget"
                                   : "reference_time_budget",
              nodes, depth};
    }
    auto n = queue.front();
    queue.pop_front();
    nodes++;
    depth = std::max(depth, n.d);
    s.add("reference_ray_nodes");
    s.maximum("reference_max_depth", depth);
    int state = impl->natural(line, n.l, n.h, s, o.tail_bounds);
    if (state == 0)
      continue;
    if (state == 1)
      return {1, "independent_parameter_interval_solid", nodes, depth};
    double mid = (n.l + n.h) / 2;
    if (o.taylor && n.d >= o.taylor_depth) {
      if (!derivatives_ready) {
        impl->derivatives(line, source, receiver, second, s);
        derivatives_ready = true;
      }
      state = impl->taylor(line, n.l, n.h, second, s, o.tail_bounds);
      if (state == 0)
        continue;
    } else
      state = impl->natural(line, mid, mid, s, o.tail_bounds);
    if (state == 1)
      return {1, "independent_midpoint_witness", nodes, depth};
    if (state < 0 && o.tail_bounds &&
        impl->natural(line, mid, mid, s, false) == 1)
      return {1, "independent_full_exp_midpoint_witness", nodes, depth};
    if (n.d >= o.max_depth) {
      if (o.tail_bounds) {
        int tight = impl->natural(line, n.l, n.h, s, false);
        if (tight == 0)
          continue;
        if (tight == 1)
          return {1, "independent_full_exp_interval", nodes, depth};
      }
      s.add("reference_unknown_rays");
      return {-1, "reference_depth_or_precision_budget", nodes, depth};
    }
    queue.push_back({n.l, mid, n.d + 1});
    queue.push_back({mid, n.h, n.d + 1});
  }
  return {0, "independent_partition_free", nodes, depth};
}
std::pair<int, int> Independent::certificate(P source, P lo, P hi,
                                             const Options &o, Stats &s,
                                             int known_y, int known_m) const {
  int mask = known_m >= 0 ? known_m : box(lo, hi, o, s).status;
  if (known_m >= 0)
    s.add("reference_inherited_mask_certificates");
  if (mask == 1)
    return {0, 1};
  if (known_y >= 0) {
    s.add("reference_inherited_visibility_certificates");
    return {known_y, mask == 0 ? 0 : -1};
  }
  P a, b;
  for (int j = 0; j < 3; j++) {
    a[j] = std::min(source[j], lo[j]);
    b[j] = std::max(source[j], hi[j]);
  }
  if (mask == 0 && box(a, b, o, s).status == 0) {
    s.add("reference_free_hull");
    return {1, 0};
  }
  // A whole SOLID cross-section at one common parameter intercepts every ray
  // in the receiver box. This uses the full sum, not primary ellipsoid corners.
  std::vector<std::pair<double, double>> proposed;
  P center;
  double norm = 0;
  for (int j = 0; j < 3; j++) {
    center[j] = (lo[j] + hi[j]) / 2;
    norm += (center[j] - source[j]) * (center[j] - source[j]);
  }
  if (norm > 0)
    for (auto &k : impl->kernels) {
      double t = 0;
      for (int j = 0; j < 3; j++)
        t += (k.c[j] - source[j]) * (center[j] - source[j]);
      t = std::clamp(t / norm, 0., 1.);
      double dist = 0;
      for (int j = 0; j < 3; j++) {
        double q = source[j] + t * (center[j] - source[j]) - k.c[j];
        dist += q * q;
      }
      proposed.push_back({dist, t});
    }
  std::sort(proposed.begin(), proposed.end());
  std::vector<double> times;
  for (std::size_t i = 0; i < std::min<std::size_t>(2, proposed.size()); i++)
    times.push_back(proposed[i].second);
  times.push_back(.5);
  MP v(bits);
  for (double t : times) {
    for (int j = 0; j < 3; j++) {
      Q l = Q(source[j]) + Q(t) * (Q(lo[j]) - Q(source[j])),
        h = Q(source[j]) + Q(t) * (Q(hi[j]) - Q(source[j]));
      mpfr_set_q(v.v, l.get_mpq_t(), MPFR_RNDD);
      a[j] = mpfr_get_d(v.v, MPFR_RNDD);
      mpfr_set_q(v.v, h.get_mpq_t(), MPFR_RNDU);
      b[j] = mpfr_get_d(v.v, MPFR_RNDU);
    }
    if (box(a, b, o, s).status == 1) {
      s.add("reference_solid_cross_section_shadow");
      return {0, mask == 0 ? 0 : -1};
    }
  }
  return {-1, mask == 0 ? 0 : -1};
}
Field Independent::field(P s, int n, double seconds, const Options &o,
                         bool scalar) const {
  return walk_field(*this, s, n, seconds, o, scalar);
}
} // namespace b2
