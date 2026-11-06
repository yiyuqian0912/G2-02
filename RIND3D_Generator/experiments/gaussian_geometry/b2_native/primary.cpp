#include "field.hpp"
#include "interval.hpp"
#include <functional>
#include <numeric>
#include <stdexcept>

namespace b2 {
bool finite(P p) {
  return std::all_of(p.begin(), p.end(),
                     [](double v) { return std::isfinite(v); });
}
void require_environment() {
  if (std::fegetround() != FE_TONEAREST ||
      (_mm_getcsr() & ((1 << 15) | (1 << 6))))
    throw std::runtime_error("Requires nearest rounding and gradual underflow");
}
void Options::validate() const {
  if (max_nodes < 0 || max_nodes > 1000000 || max_depth < 0 || max_depth > 50 ||
      precision < 80 || precision > 2048 || query_seconds <= 0 ||
      exact_depth < 0 || exact_depth > 51 || taylor_depth < 0 ||
      taylor_depth > 50 || !std::isfinite(query_seconds))
    throw std::invalid_argument("Invalid proof budget");
}
namespace {
using Inv = std::array<Q, 9>;
Inv invert(const Matrix &m) {
  Q a[3][6];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 6; j++)
      a[i][j] = j < 3 ? Q(m[i * 3 + j]) : Q(j - 3 == i);
  for (int j = 0; j < 3; j++) {
    int p = j;
    while (p < 3 && a[p][j] == 0)
      p++;
    if (p == 3)
      throw std::invalid_argument("Singular canonical M");
    for (int k = 0; k < 6; k++)
      std::swap(a[p][k], a[j][k]);
    Q d = a[j][j];
    for (int k = 0; k < 6; k++)
      a[j][k] /= d;
    for (int i = 0; i < 3; i++)
      if (i != j) {
        d = a[i][j];
        for (int k = 0; k < 6; k++)
          a[i][k] -= d * a[j][k];
      }
  }
  Inv out;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      out[i * 3 + j] = a[i][j + 3];
  return out;
}
struct Kernel {
  P c;
  Matrix m;
  double w;
  Inv qi;
  std::array<I, 9> inv;
  I inner, outer;
  P low, high;
  double norm2;
};
struct Poly {
  I a, b, c;
};
struct QPoly {
  Q a, b, c;
};
I at(const Poly &p, I t) { return add(mul(add(mul(p.a, t), p.b), t), p.c); }
I range(const Poly &p, double lo, double hi) {
  I left = at(p, I(lo)), right = at(p, I(hi)), t(lo, hi);
  if (p.a.lo > 0) {
    I v = divi(neg(p.b), mul(I(2), p.a));
    double l = std::max(lo, v.lo), h = std::min(hi, v.hi);
    if (l <= h)
      t = I(l, h);
    else
      return {std::max(0., std::min(left.lo, right.lo)),
              std::max(left.hi, right.hi)};
  }
  return {std::max(0., std::min({left.lo, right.lo, at(p, t).lo})),
          std::max(left.hi, right.hi)};
}
double peak(const Poly &p, double lo = 0, double hi = 1) {
  double a = (p.a.lo + p.a.hi) / 2, b = (p.b.lo + p.b.hi) / 2;
  return std::isfinite(a) && std::isfinite(b) && a > 0
             ? std::clamp(-b / (2 * a), lo, hi)
             : (lo + hi) / 2;
}
std::pair<Q, Q> qrange(const QPoly &p, Q lo, Q hi) {
  Q l = (p.a * lo + p.b) * lo + p.c, r = (p.a * hi + p.b) * hi + p.c, t = lo;
  if (p.a != 0) {
    t = -p.b / (2 * p.a);
    t = std::max(lo, std::min(hi, t));
  }
  Q minimum = (p.a * t + p.b) * t + p.c;
  return {minimum, std::max(l, r)};
}
Poly coefficients(const Kernel &k, P s, P r) {
  I a, b, c;
  for (int j = 0; j < 3; j++) {
    I u, v;
    for (int d = 0; d < 3; d++) {
      u = add(u, mul(k.inv[j * 3 + d], sub(I(s[d]), I(k.c[d]))));
      v = add(v, mul(k.inv[j * 3 + d], sub(I(r[d]), I(s[d]))));
    }
    a = add(a, square(v));
    b = add(b, mul(I(2), mul(u, v)));
    c = add(c, square(u));
  }
  return {a, b, c};
}
QPoly exact_coefficients(const Kernel &k, P s, P r) {
  Q a, b, c;
  for (int j = 0; j < 3; j++) {
    Q u, v;
    for (int d = 0; d < 3; d++) {
      u += k.qi[j * 3 + d] * (Q(s[d]) - Q(k.c[d]));
      v += k.qi[j * 3 + d] * (Q(r[d]) - Q(s[d]));
    }
    a += v * v;
    b += 2 * u * v;
    c += u * u;
  }
  return {a, b, c};
}
I spatial(const Kernel &k, P lo, P hi) {
  I q;
  for (int j = 0; j < 3; j++) {
    I a;
    for (int d = 0; d < 3; d++)
      a = add(a, mul(k.inv[j * 3 + d], sub(I(lo[d], hi[d]), I(k.c[d]))));
    q = add(q, square(a));
  }
  return {std::max(0., q.lo), q.hi};
}
bool intersects(P s, P r, P lo, P hi) {
  double low = 0, high = 1;
  for (int j = 0; j < 3; j++) {
    if (s[j] == r[j]) {
      if (s[j] < lo[j] || s[j] > hi[j])
        return false;
      continue;
    }
    I d = sub(I(r[j]), I(s[j]));
    if (d.lo <= 0 && d.hi >= 0)
      continue;
    I a = divi(sub(I(lo[j]), I(s[j])), d), b = divi(sub(I(hi[j]), I(s[j])), d);
    low = std::max(low, std::min(a.lo, b.lo));
    high = std::min(high, std::max(a.hi, b.hi));
    if (low > high)
      return false;
  }
  return true;
}
struct Group {
  P low, high;
  I weight;
  double norm2;
  int left = -1, right = -1;
  std::vector<int> ids;
};
} // namespace

struct Primary::Impl {
  Inputs input;
  std::vector<Kernel> kernels;
  I tau;
  std::vector<Group> groups;
  bool analytic = false;
  double setup_seconds = 0;
  explicit Impl(const Inputs &in) : input(in), tau(threshold(in)) {
    auto start = Clock::now();
    require_environment();
    int n = in.centers.size();
    if (n < 1 || n > 1024 || in.matrices.size() != std::size_t(n) ||
        in.weights.size() != std::size_t(n) || !std::isfinite(in.threshold) ||
        (!in.exponential && in.threshold <= 0))
      throw std::invalid_argument("Invalid canonical Gaussian inputs");
    Q total_weight;
    for (int i = 0; i < n; i++) {
      if (!finite(in.centers[i]) || !std::isfinite(in.weights[i]) ||
          in.weights[i] <= 0 ||
          !std::all_of(in.matrices[i].begin(), in.matrices[i].end(),
                       [](double v) { return std::isfinite(v); }))
        throw std::invalid_argument("Invalid kernel parameters");
      Kernel k;
      k.c = in.centers[i];
      k.m = in.matrices[i];
      k.w = in.weights[i];
      k.qi = invert(k.m);
      for (int j = 0; j < 9; j++)
        k.inv[j] = enclose(k.qi[j]);
      MP a(192), b(192), c(192);
      mpfr_set_d(a.v, k.w, MPFR_RNDN);
      mpfr_log(a.v, a.v, MPFR_RNDD);
      if (in.exponential)
        mpfr_set_d(b.v, in.threshold, MPFR_RNDN);
      else {
        mpfr_set_d(b.v, in.threshold, MPFR_RNDN);
        mpfr_log(b.v, b.v, MPFR_RNDU);
      }
      mpfr_sub(a.v, a.v, b.v, MPFR_RNDD);
      mpfr_mul_2ui(a.v, a.v, 1, MPFR_RNDD);
      k.inner = I(mpfr_get_d(a.v, MPFR_RNDD));
      Q nw = Q(k.w) * n;
      mpfr_set_q(a.v, nw.get_mpq_t(), MPFR_RNDU);
      mpfr_log(a.v, a.v, MPFR_RNDU);
      if (in.exponential)
        mpfr_set_d(b.v, in.threshold, MPFR_RNDN);
      else {
        mpfr_set_d(b.v, in.threshold, MPFR_RNDN);
        mpfr_log(b.v, b.v, MPFR_RNDD);
      }
      mpfr_sub(a.v, a.v, b.v, MPFR_RNDU);
      mpfr_mul_2ui(a.v, a.v, 1, MPFR_RNDU);
      k.outer = I(mpfr_get_d(a.v, MPFR_RNDU));
      Q norm;
      for (int j = 0; j < 3; j++) {
        Q row;
        for (int d = 0; d < 3; d++) {
          Q v(k.m[j * 3 + d]);
          row += v * v;
        }
        norm += row;
        if (k.outer.hi >= 0 && std::isfinite(k.outer.hi)) {
          Q v = row * Q(k.outer.hi);
          mpfr_set_q(c.v, v.get_mpq_t(), MPFR_RNDU);
          mpfr_sqrt(c.v, c.v, MPFR_RNDU);
          double radius = mpfr_get_d(c.v, MPFR_RNDU);
          k.low[j] = sub(I(k.c[j]), I(radius)).lo;
          k.high[j] = add(I(k.c[j]), I(radius)).hi;
        } else {
          k.low[j] = -INFINITY;
          k.high[j] = INFINITY;
        }
      }
      k.norm2 = enclose(norm).hi;
      kernels.push_back(k);
      total_weight += Q(k.w);
    }
    analytic = in.exponential && total_weight == 1;
    for (auto &k : kernels)
      if (k.c != kernels[0].c || k.m != kernels[0].m)
        analytic = false;
    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    build(ids);
    setup_seconds = elapsed(start);
  }
  int build(std::vector<int> ids) {
    Group g;
    g.low.fill(INFINITY);
    g.high.fill(-INFINITY);
    g.norm2 = 0;
    Q weight;
    for (int i : ids) {
      auto &k = kernels[i];
      for (int a = 0; a < 3; a++) {
        g.low[a] = std::min(g.low[a], k.c[a]);
        g.high[a] = std::max(g.high[a], k.c[a]);
      }
      g.norm2 = std::max(g.norm2, k.norm2);
      weight += Q(k.w);
    }
    g.weight = enclose(weight);
    int index = groups.size();
    groups.push_back(g);
    if (ids.size() <= 3) {
      groups[index].ids = ids;
      return index;
    }
    int axis = 0;
    for (int a = 1; a < 3; a++)
      if (g.high[a] - g.low[a] > g.high[axis] - g.low[axis])
        axis = a;
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
      return kernels[a].c[axis] < kernels[b].c[axis];
    });
    auto mid = ids.begin() + ids.size() / 2;
    int l = build({ids.begin(), mid}), r = build({mid, ids.end()});
    groups[index].left = l;
    groups[index].right = r;
    return index;
  }
  bool outside(P lo, P hi) const {
    for (auto &k : kernels) {
      if (k.outer.hi < 0)
        continue;
      bool possible = true;
      for (int a = 0; a < 3; a++)
        if (hi[a] < k.low[a] || lo[a] > k.high[a])
          possible = false;
      if (possible)
        return false;
    }
    return true;
  }
  template <class Range>
  I sum(Range range, P lo, P hi, Exp &exp, Stats &s, bool grouped) const {
    std::function<I(int)> visit = [&](int id) -> I {
      const auto &g = groups[id];
      if (grouped && g.ids.empty()) {
        I dist;
        for (int a = 0; a < 3; a++) {
          I d;
          if (hi[a] < g.low[a])
            d = sub(I(g.low[a]), I(hi[a]));
          else if (lo[a] > g.high[a])
            d = sub(I(lo[a]), I(g.high[a]));
          dist = add(dist, square(d));
        }
        I bound = mul(g.weight, exp.value(divi(dist, I(g.norm2)), s));
        s.add("group_bounds");
        if (bound.hi < tau.lo / 64) {
          s.add("group_remainders");
          s.add("group_kernel_terms_retained_as_remainder", count_group(id));
          return {0, bound.hi};
        }
      }
      if (g.ids.empty())
        return add(visit(g.left), visit(g.right));
      I result;
      for (int i : g.ids) {
        s.add("gaussian_terms");
        result = add(result, mul(I(kernels[i].w), exp.value(range(i), s)));
      }
      return result;
    };
    if (grouped)
      return visit(0);
    I result;
    for (std::size_t i = 0; i < kernels.size(); i++) {
      s.add("gaussian_terms");
      result = add(result, mul(I(kernels[i].w), exp.value(range(i), s)));
    }
    return result;
  }
  int count_group(int id) const {
    auto &g = groups[id];
    return g.ids.empty() ? count_group(g.left) + count_group(g.right)
                         : g.ids.size();
  }
  int exact_sum(const std::vector<QPoly> &p, Q lo, Q hi, int bits,
                Stats &s) const {
    s.add("high_precision_calls");
    MP l(bits), h(bits), v(bits), w(bits), tl(bits), th(bits);
    for (std::size_t i = 0; i < kernels.size(); i++) {
      auto [ql, qh] = qrange(p[i], lo, hi);
      mpfr_set_q(v.v, qh.get_mpq_t(), MPFR_RNDU);
      mpfr_neg(v.v, v.v, MPFR_RNDN);
      mpfr_div_2ui(v.v, v.v, 1, MPFR_RNDN);
      mpfr_exp(v.v, v.v, MPFR_RNDD);
      mpfr_mul_d(v.v, v.v, kernels[i].w, MPFR_RNDD);
      mpfr_add(l.v, l.v, v.v, MPFR_RNDD);
      mpfr_set_q(v.v, ql.get_mpq_t(), MPFR_RNDD);
      mpfr_neg(v.v, v.v, MPFR_RNDN);
      mpfr_div_2ui(v.v, v.v, 1, MPFR_RNDN);
      mpfr_exp(v.v, v.v, MPFR_RNDU);
      mpfr_mul_d(v.v, v.v, kernels[i].w, MPFR_RNDU);
      mpfr_add(h.v, h.v, v.v, MPFR_RNDU);
      s.add("high_precision_exp_endpoints", 2);
    }
    mpfr_set_d(tl.v, input.threshold, MPFR_RNDN);
    mpfr_set(th.v, tl.v, MPFR_RNDN);
    if (input.exponential) {
      mpfr_exp(tl.v, tl.v, MPFR_RNDD);
      mpfr_exp(th.v, th.v, MPFR_RNDU);
    }
    if (mpfr_cmp(l.v, th.v) >= 0)
      return 1;
    if (mpfr_cmp(h.v, tl.v) < 0)
      return 0;
    return -1;
  }
};

Primary::Primary(const Inputs &i) : impl(std::make_shared<Impl>(i)) {}
Decision Primary::point(P p, const Options &o, Stats &s) const {
  s.add("point_queries");
  if (!finite(p))
    return {-2, "nonfinite_point"};
  const auto &e = *impl;
  if (o.filters && e.outside(p, p)) {
    s.add("outer_point_free");
    return {0, "outer_support"};
  }
  if (e.analytic) {
    auto q = exact_coefficients(e.kernels[0], p, p).c;
    return {q <= -2 * Q(e.input.threshold) ? 1 : 0,
            "exact_single_shape_reduction"};
  }
  Exp exp;
  auto range = [&](int i) { return spatial(e.kernels[i], p, p); };
  I v = e.sum(range, p, p, exp, s, o.groups);
  int result = compare(v, e.tau);
  if (result < 0 && o.groups) {
    s.add("group_retries");
    result = compare(e.sum(range, p, p, exp, s, false), e.tau);
  }
  if (result >= 0)
    return {result, "directed_interval_point"};
  if (o.fallback) {
    std::vector<QPoly> q;
    for (auto &k : e.kernels)
      q.push_back(exact_coefficients(k, p, p));
    for (int bits : {o.precision, o.precision * 2}) {
      result = e.exact_sum(q, 0, 0, bits, s);
      if (result >= 0)
        return {result, "exact_quadratic_MPFR_point"};
    }
  }
  s.add("unknown_points");
  return {-1, "point_precision_exhausted"};
}
Decision Primary::box(P lo, P hi, const Options &o, Stats &s) const {
  s.add("region_queries");
  if (!finite(lo) || !finite(hi))
    return {-2, "nonfinite_box"};
  for (int a = 0; a < 3; a++)
    if (lo[a] > hi[a])
      return {-2, "reversed_box"};
  const auto &e = *impl;
  if (o.filters && e.outside(lo, hi)) {
    s.add("outer_region_free");
    return {0, "outer_support_box"};
  }
  Exp exp;
  auto range = [&](int i) { return spatial(e.kernels[i], lo, hi); };
  I v = e.sum(range, lo, hi, exp, s, o.groups);
  int r = compare(v, e.tau);
  if (r < 0 && o.groups) {
    s.add("group_retries");
    r = compare(e.sum(range, lo, hi, exp, s, false), e.tau);
  }
  if (r >= 0)
    s.add(r ? "solid_region_certificates" : "free_region_certificates");
  return {r, r < 0 ? "mixed_or_unresolved_box" : "directed_field_box"};
}
Decision Primary::ray(P source, P receiver, const Options &o, Stats &s) const {
  s.add("ray_queries");
  if (!finite(source) || !finite(receiver))
    return {-2, "nonfinite_segment"};
  const auto &e = *impl;
  if (o.filters) {
    bool hit = false;
    for (auto &k : e.kernels)
      if (k.outer.hi >= 0 && intersects(source, receiver, k.low, k.high)) {
        hit = true;
        break;
      }
    if (!hit) {
      s.add("outer_ray_miss");
      return {0, "outer_support"};
    }
  }
  if (e.analytic) {
    auto q = qrange(exact_coefficients(e.kernels[0], source, receiver), 0, 1);
    return {q.first <= -2 * Q(e.input.threshold) ? 1 : 0,
            "exact_closed_quadratic"};
  }
  std::vector<Poly> coefficients_cache;
  auto prepare = [&]() {
    std::vector<Poly> v;
    v.reserve(e.kernels.size());
    for (auto &k : e.kernels) {
      v.push_back(coefficients(k, source, receiver));
      s.add("coefficient_builds");
    }
    return v;
  };
  if (o.coefficients)
    coefficients_cache = prepare();
  std::vector<QPoly> exact;
  auto fallback = [&](Q lo, Q hi) {
    if (exact.empty())
      for (auto &k : e.kernels)
        exact.push_back(exact_coefficients(k, source, receiver));
    return e.exact_sum(exact, lo, hi, o.precision, s);
  };
  struct Node {
    double l, r;
    int depth;
  };
  std::vector<Node> stack{{0, 1, 0}};
  Exp exp;
  auto began = Clock::now();
  int nodes = 0, depth = 0;
  double second_derivative_bound = -1;
  while (!stack.empty()) {
    if (nodes >= o.max_nodes || elapsed(began) > o.query_seconds) {
      s.add("unknown_rays");
      return {-1, nodes >= o.max_nodes ? "node_budget" : "time_budget", nodes,
              depth};
    }
    auto node = stack.back();
    stack.pop_back();
    nodes++;
    depth = std::max(depth, node.depth);
    s.add("ray_nodes");
    s.maximum("max_ray_depth", depth);
    std::vector<Poly> local;
    if (!o.coefficients)
      local = prepare();
    const auto &c = o.coefficients ? coefficients_cache : local;
    if (o.filters && nodes == 1)
      for (std::size_t i = 0; i < c.size(); i++)
        if (e.kernels[i].inner.lo >= 0 &&
            at(c[i], I(peak(c[i]))).hi <= e.kernels[i].inner.lo) {
          s.add("inner_ray_hit");
          return {1, "contained_ellipsoid_witness", nodes, depth};
        }
    P lo, hi;
    for (int a = 0; a < 3; a++) {
      I t = add(I(source[a]),
                mul(sub(I(receiver[a]), I(source[a])), I(node.l, node.r)));
      lo[a] = t.lo;
      hi[a] = t.hi;
    }
    auto qr = [&](int i) { return range(c[i], node.l, node.r); };
    I value = e.sum(qr, lo, hi, exp, s, o.groups);
    int state = compare(value, e.tau);
    if (state < 0 && o.groups) {
      s.add("group_retries");
      state = compare(e.sum(qr, lo, hi, exp, s, false), e.tau);
    }
    if (state == 0)
      continue;
    if (state == 1)
      return {1, "entire_parameter_interval_solid", nodes, depth};
    if (o.fallback && node.depth >= o.exact_depth) {
      int exact_state = fallback(Q(node.l), Q(node.r));
      if (exact_state == 0) {
        s.add("early_exact_interval_free");
        continue;
      }
      if (exact_state == 1)
        return {1, "early_exact_interval_solid", nodes, depth};
    }
    std::vector<double> witnesses{(node.l + node.r) / 2};
    if (nodes == 1) {
      std::vector<std::pair<double, int>> rank;
      for (std::size_t i = 0; i < c.size(); i++)
        rank.push_back({qr(i).lo, i});
      std::sort(rank.begin(), rank.end());
      for (std::size_t i = 0; i < std::min<std::size_t>(2, rank.size()); i++)
        witnesses.push_back(peak(c[rank[i].second], node.l, node.r));
    }
    bool interval_free = false;
    for (std::size_t wi = 0; wi < witnesses.size(); wi++) {
      double t = witnesses[wi];
      P xlo, xhi;
      for (int a = 0; a < 3; a++) {
        I x = add(I(source[a]), mul(sub(I(receiver[a]), I(source[a])), I(t)));
        xlo[a] = x.lo;
        xhi[a] = x.hi;
      }
      auto qp = [&](int i) {
        I v = at(c[i], I(t));
        return I(std::max(0., v.lo), std::max(0., v.hi));
      };
      I v, derivative;
      bool taylor = o.taylor && wi == 0 && node.depth >= o.taylor_depth;
      if (taylor) {
        // F' is summed with signs, preserving cancellation between kernels.
        // |g_i''| <= 2*w_i*a_i because (q+1)exp(-q/2) < 2 for q>=0.
        if (second_derivative_bound < 0) {
          // 2*w is formed as an interval too; avoid a rounded coefficient.
          I bound;
          for (std::size_t i = 0; i < c.size(); i++)
            bound = add(bound, mul(mul(I(2), I(e.kernels[i].w)),
                                   I(std::max(0., c[i].a.hi))));
          second_derivative_bound = bound.hi;
          s.add("taylor_curvature_setups");
        }
        for (std::size_t i = 0; i < c.size(); i++) {
          I g = mul(I(e.kernels[i].w), exp.value(qp(i), s));
          v = add(v, g);
          I qprime = add(mul(mul(I(2), c[i].a), I(t)), c[i].b);
          derivative = add(derivative, mul(mul(I(-.5), qprime), g));
          s.add("gaussian_terms");
        }
        s.add("taylor_tests");
      } else
        v = e.sum(qp, xlo, xhi, exp, s, false);
      int st = compare(v, e.tau);
      s.add("point_witness_tests");
      if (st == 1)
        return {1, "certified_parameter_witness", nodes, depth};
      if (taylor) {
        double h = (node.r - node.l) / 2;
        I error = add(
            mul(I(std::max(std::abs(derivative.lo), std::abs(derivative.hi))),
                I(h)),
            mul(mul(I(.5), I(second_derivative_bound)), square(I(h))));
        if (add(v, error).hi < e.tau.lo) {
          s.add("taylor_interval_free");
          interval_free = true;
          break;
        }
      }
      if (st < 0 && o.fallback && fallback(Q(t), Q(t)) == 1)
        return {1, "high_precision_point_witness", nodes, depth};
    }
    if (interval_free)
      continue;
    if (node.depth >= o.max_depth) {
      if (o.fallback) {
        int r = fallback(Q(node.l), Q(node.r));
        if (r == 0)
          continue;
        if (r == 1)
          return {1, "high_precision_interval", nodes, depth};
      }
      s.add("unknown_rays");
      return {-1, "depth_or_precision_budget", nodes, depth};
    }
    double mid = (node.l + node.r) / 2;
    stack.push_back({mid, node.r, node.depth + 1});
    stack.push_back({node.l, mid, node.depth + 1});
  }
  return {0, "all_closed_intervals_certified_free", nodes, depth};
}
std::pair<int, int> Primary::certificate(P source, P lo, P hi, const Options &o,
                                         Stats &s, int known_y,
                                         int known_m) const {
  int mask = known_m >= 0 ? known_m : box(lo, hi, o, s).status;
  if (known_m >= 0)
    s.add("inherited_mask_certificates");
  if (mask == 1)
    return {0, 1};
  if (known_y >= 0) {
    s.add("inherited_visibility_certificates");
    return {known_y, mask == 0 ? 0 : -1};
  }
  P a, b;
  for (int j = 0; j < 3; j++) {
    a[j] = std::min(source[j], lo[j]);
    b[j] = std::max(source[j], hi[j]);
  }
  if (mask == 0 && box(a, b, o, s).status == 0) {
    s.add("free_visibility_hull");
    return {1, 0};
  }
  if (o.filters)
    for (auto &k : impl->kernels) {
      if (k.inner.lo < 0)
        continue;
      bool all = true;
      for (int corner = 0; corner < 8; corner++) {
        P p;
        for (int j = 0; j < 3; j++)
          p[j] = corner & (1 << j) ? hi[j] : lo[j];
        auto q = coefficients(k, source, p);
        if (at(q, I(peak(q))).hi > k.inner.lo) {
          all = false;
          break;
        }
      }
      if (all) {
        s.add("single_proxy_shadow_region");
        return {0, mask == 0 ? 0 : -1};
      }
    }
  return {-1, mask == 0 ? 0 : -1};
}
Field Primary::field(P source, int n, double sec, const Options &o,
                     bool scalar) const {
  return walk_field(*this, source, n, sec, o, scalar);
}
std::map<std::string, double> Primary::info() const {
  return {{"kernel_count", double(impl->kernels.size())},
          {"group_count", double(impl->groups.size())},
          {"setup_seconds", impl->setup_seconds},
          {"ready_capable", 0}};
}
} // namespace b2
