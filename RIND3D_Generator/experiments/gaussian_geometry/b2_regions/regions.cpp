#include "regions.hpp"
#include "exp.hpp"
#include <numeric>
#include <stdexcept>

namespace b2r {
void Settings::validate() const {
  if (mode < 0 || mode > 5 || cover_depth < 0 || cover_depth > 10 ||
      cover_nodes < 1 || cover_nodes > 1000000 || shadow_boxes < 0 ||
      shadow_boxes > 100000 || parameter_nodes < 0 || parameter_nodes > 65535 ||
      parameter_depth < 0 || parameter_depth > 30 || !(cover_seconds > 0) ||
      !std::isfinite(cover_seconds))
    throw std::invalid_argument("Invalid region proof budget");
}
I certified_exp(double x, bool fast, Stats &s) {
  if (std::isnan(x))
    throw std::invalid_argument("NaN exponent");
  if (fast && x <= 0) {
    if (x == -INFINITY)
      return I(0);
    if (x <= -1074)
      return {0, std::numeric_limits<double>::denorm_min()};
    return negative_half_exp(-2 * x, s);
  }
  MP v(53);
  mpfr_set_d(v.v, x, MPFR_RNDN);
  mpfr_exp(v.v, v.v, MPFR_RNDD);
  double l = mpfr_get_d(v.v, MPFR_RNDD);
  mpfr_set_d(v.v, x, MPFR_RNDN);
  mpfr_exp(v.v, v.v, MPFR_RNDU);
  double h = mpfr_get_d(v.v, MPFR_RNDU);
  s.add("generic_MPFR_exp_endpoints", 2);
  return {l, h};
}
namespace {
struct Box {
  P lo, hi;
};
bool good(Box b) {
  if (!finite(b.lo) || !finite(b.hi))
    return false;
  for (int j = 0; j < 3; ++j)
    if (b.lo[j] > b.hi[j])
      return false;
  return true;
}
bool contained(Box a, Box b) {
  for (int j = 0; j < 3; ++j)
    if (a.lo[j] < b.lo[j] || a.hi[j] > b.hi[j])
      return false;
  return true;
}
bool overlaps(Box a, Box b) {
  for (int j = 0; j < 3; ++j)
    if (a.lo[j] > b.hi[j] || a.hi[j] < b.lo[j])
      return false;
  return true;
}
// Necessary time interval for ANY ray from s to b to touch obstacle a.
// Empty is a rigorous whole-ray-volume exclusion. Equality stays possible.
bool possible_clip(P s, Box b, Box a, Stats &stats) {
  stats.add("primary_volume_clip_tests");
  double low = 0, high = 1;
  for (int j = 0; j < 3; ++j) {
    if (b.lo[j] == s[j]) {
      if (s[j] > a.hi[j])
        return false;
    } else {
      I ratio = divi(sub(I(a.hi[j]), I(s[j])), sub(I(b.lo[j]), I(s[j])));
      if (b.lo[j] > s[j])
        high = std::min(high, ratio.hi);
      else
        low = std::max(low, ratio.lo);
    }
    if (b.hi[j] == s[j]) {
      if (s[j] < a.lo[j])
        return false;
    } else {
      I ratio = divi(sub(I(a.lo[j]), I(s[j])), sub(I(b.hi[j]), I(s[j])));
      if (b.hi[j] > s[j])
        low = std::max(low, ratio.lo);
      else
        high = std::min(high, ratio.hi);
    }
    if (low > high)
      return false;
  }
  return true;
}
// Inner time interval: every t in it lies in a for the chosen corner ray.
bool corner_hit(P s, P r, Box a, Stats &stats) {
  stats.add("primary_inner_corner_tests");
  double low = 0, high = 1;
  for (int j = 0; j < 3; ++j) {
    if (r[j] == s[j]) {
      if (s[j] < a.lo[j] || s[j] > a.hi[j])
        return false;
      continue;
    }
    I d = sub(I(r[j]), I(s[j]));
    I l = divi(sub(I(a.lo[j]), I(s[j])), d),
      h = divi(sub(I(a.hi[j]), I(s[j])), d);
    if (r[j] < s[j])
      std::swap(l, h);
    low = std::max(low, l.hi);
    high = std::min(high, h.lo);
    if (low > high)
      return false;
  }
  return true;
}
struct Plane {
  std::array<I, 3> a;
  I rhs;
};
// Independent Fourier-Motzkin projection of lambda>=1 and
// lambda*(a-s) <= x-s <= lambda*(b-s). Exact rational elimination.
std::vector<Plane> shadow_projection(P s, Box b) {
  struct Row {
    Q k;
    std::array<Q, 3> a;
    Q rhs;
  };
  std::vector<Row> pos, neg, zero;
  auto insert = [&](Row r) {
    (r.k > 0 ? pos : r.k < 0 ? neg : zero).push_back(r);
  };
  insert({Q(-1), {}, Q(-1)});
  for (int j = 0; j < 3; ++j) {
    Row l{Q(b.lo[j]) - Q(s[j]), {}, -Q(s[j])};
    l.a[j] = -1;
    insert(l);
    Row h{Q(s[j]) - Q(b.hi[j]), {}, Q(s[j])};
    h.a[j] = 1;
    insert(h);
  }
  std::vector<Plane> out;
  auto append = [&](const Row &r) {
    Plane p;
    for (int j = 0; j < 3; ++j)
      p.a[j] = enclose(r.a[j]);
    p.rhs = enclose(r.rhs);
    out.push_back(p);
  };
  for (auto &r : zero)
    append(r);
  for (auto &p : pos)
    for (auto &n : neg) {
      Row r;
      for (int j = 0; j < 3; ++j)
        r.a[j] = p.a[j] * (-n.k) + n.a[j] * p.k;
      r.rhs = p.rhs * (-n.k) + n.rhs * p.k;
      append(r);
    }
  return out;
}
bool projected_excludes(const std::vector<Plane> &planes, Box b, Stats &s) {
  for (auto &p : planes) {
    I v;
    for (int j = 0; j < 3; ++j)
      v = add(v, mul(p.a[j], I(b.lo[j], b.hi[j])));
    s.add("reference_projection_plane_tests");
    if (v.lo > p.rhs.hi)
      return true;
  }
  return false;
}
bool projected_contains(const std::vector<Plane> &planes, Box b, Stats &s) {
  for (auto &p : planes) {
    I v;
    for (int j = 0; j < 3; ++j)
      v = add(v, mul(p.a[j], I(b.lo[j], b.hi[j])));
    s.add("reference_projection_plane_tests");
    if (v.hi > p.rhs.lo)
      return false;
  }
  return true;
}
struct Node {
  Box box;
  int label = -1, depth = 0;
  std::array<int, 8> children{};
  bool split = false, projected = false;
  std::vector<Plane> planes;
};
struct Cover {
  std::vector<Node> nodes;
  std::vector<int> inner;
};
using Inv = std::array<Q, 9>;
Inv inverse(const Matrix &m) {
  Q a[3][6];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j)
      a[i][j] = j < 3 ? Q(m[3 * i + j]) : Q(j - 3 == i);
  for (int k = 0; k < 3; ++k) {
    int p = k;
    while (p < 3 && a[p][k] == 0)
      ++p;
    if (p == 3)
      throw std::invalid_argument("singular M");
    for (int j = 0; j < 6; ++j)
      std::swap(a[k][j], a[p][j]);
    Q d = a[k][k];
    for (int j = 0; j < 6; ++j)
      a[k][j] /= d;
    for (int i = 0; i < 3; ++i)
      if (i != k) {
        d = a[i][k];
        for (int j = 0; j < 6; ++j)
          a[i][j] -= d * a[k][j];
      }
  }
  Inv out;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      out[3 * i + j] = a[i][j + 3];
  return out;
}
struct Kernel {
  P center;
  std::array<I, 9> inv;
  double w;
};
struct Parametric {
  std::array<I, 3> source, delta;
  double w;
};
} // namespace
std::map<std::string, bool> geometry_screen(P source, P receiver_lo,
                                            P receiver_hi, P obstacle_lo,
                                            P obstacle_hi) {
  require_environment();
  Box r{receiver_lo, receiver_hi}, k{obstacle_lo, obstacle_hi};
  if (!finite(source) || !good(r) || !good(k))
    throw std::invalid_argument("Invalid box screen");
  Stats stats;
  bool all = true;
  for (int c = 0; c < 8; ++c) {
    P p;
    for (int j = 0; j < 3; ++j)
      p[j] = (c & (1 << j)) ? r.hi[j] : r.lo[j];
    if (!corner_hit(source, p, k, stats)) {
      all = false;
      break;
    }
  }
  auto planes = shadow_projection(source, k);
  return {
      {"primary_volume_excluded", !possible_clip(source, r, k, stats)},
      {"primary_all_shadow", all},
      {"reference_volume_excluded",
       !contained({source, source}, k) && projected_excludes(planes, r, stats)},
      {"reference_all_shadow", projected_contains(planes, r, stats)}};
}
struct Engine::Impl {
  Inputs input;
  bool independent;
  std::unique_ptr<Primary> a;
  std::unique_ptr<Independent> b;
  std::vector<Kernel> kernels;
  I tau;
  double setup_seconds;
  Impl(const Inputs &i, bool ref)
      : input(i), independent(ref), tau(threshold(i)) {
    auto start = Clock::now();
    require_environment();
    if (ref)
      b = std::make_unique<Independent>(i);
    else {
      a = std::make_unique<Primary>(i);
      for (std::size_t j = 0; j < i.centers.size(); ++j) {
        Kernel k;
        k.center = i.centers[j];
        k.w = i.weights[j];
        auto inv = inverse(i.matrices[j]);
        for (int d = 0; d < 9; ++d)
          k.inv[d] = enclose(inv[d]);
        kernels.push_back(k);
      }
    }
    setup_seconds = elapsed(start);
  }
  Decision box(Box x, const Options &o, Stats &s) const {
    return independent ? b->box(x.lo, x.hi, o, s) : a->box(x.lo, x.hi, o, s);
  }
  Decision point(P x, const Options &o, Stats &s) const {
    return independent ? b->point(x, o, s) : a->point(x, o, s);
  }
  Decision ray(P source, P x, const Options &o, Stats &s) const {
    return independent ? b->ray(source, x, o, s) : a->ray(source, x, o, s);
  }
  std::pair<int, int> old_cert(P source, Box x, const Options &o, Stats &s,
                               int y, int m) const {
    return independent ? b->certificate(source, x.lo, x.hi, o, s, y, m)
                       : a->certificate(source, x.lo, x.hi, o, s, y, m);
  }
  std::vector<Parametric> parameters(P source, Box r) const {
    std::vector<Parametric> all;
    for (auto &k : kernels) {
      Parametric p;
      p.w = k.w;
      for (int row = 0; row < 3; ++row) {
        I u, v;
        for (int j = 0; j < 3; ++j) {
          u = add(u,
                  mul(k.inv[3 * row + j], sub(I(source[j]), I(k.center[j]))));
          v = add(v, mul(k.inv[3 * row + j],
                         sub(I(r.lo[j], r.hi[j]), I(source[j]))));
        }
        p.source[row] = u;
        p.delta[row] = v;
      }
      all.push_back(p);
    }
    return all;
  }
  int parameter_bound(const std::vector<Parametric> &all, double lo, double hi,
                      const Settings &cfg, Stats &s) const {
    Exp exp;
    I sum;
    for (auto &k : all) {
      I q;
      for (int j = 0; j < 3; ++j)
        q = add(q, square(add(k.source[j], mul(I(lo, hi), k.delta[j]))));
      sum = add(sum, mul(I(k.w), exp_q(q, cfg.fast_exp, exp, s)));
      s.add("parameter_gaussian_terms");
    }
    return compare(sum, tau);
  }
  Box section(P source, Box r, double lo, double hi) const {
    Box out;
    for (int j = 0; j < 3; ++j) {
      I v = add(I(source[j]),
                mul(I(lo, hi), sub(I(r.lo[j], r.hi[j]), I(source[j]))));
      out.lo[j] = v.lo;
      out.hi[j] = v.hi;
    }
    return out;
  }
  int bound(P source, Box r, double lo, double hi, const Options &o,
            const Settings &cfg, Stats &s,
            const std::vector<Parametric> &params) const {
    // Reference uses separate natural 256-bit spatial bounds, not primary's
    // local affine parameter bounds or exponential filter.
    return independent ? box(section(source, r, lo, hi), o, s).status
                       : parameter_bound(params, lo, hi, cfg, s);
  }
  int volume(P source, Box r, const Options &o, const Settings &cfg,
             Stats &s) const {
    auto params = parameters(source, r);
    struct T {
      double l, h;
      int d;
    };
    std::vector<T> stack{{0, 1, 0}};
    int count = 0;
    while (!stack.empty()) {
      if (count++ >= cfg.parameter_nodes) {
        s.add("parameter_budget_unknown");
        return -1;
      }
      auto n = stack.back();
      stack.pop_back();
      s.add("parameter_volume_slabs");
      int state = bound(source, r, n.l, n.h, o, cfg, s, params);
      if (state == 0)
        continue;
      if (state == 1) {
        s.add("parameter_solid_slab");
        return 0;
      }
      if (n.d >= cfg.parameter_depth) {
        s.add("parameter_depth_unknown");
        return -1;
      }
      double m = (n.l + n.h) / 2;
      stack.push_back({m, n.h, n.d + 1});
      stack.push_back({n.l, m, n.d + 1});
    }
    s.add("parameter_whole_volume_FREE");
    return 1;
  }
  int fused(P source, Box r, const Options &o, const Settings &cfg,
            Stats &s) const {
    auto params = parameters(source, r);
    P center;
    for (int j = 0; j < 3; ++j)
      center[j] = (r.lo[j] + r.hi[j]) / 2;
    std::vector<std::pair<double, double>> times;
    double d2 = 0;
    for (int j = 0; j < 3; ++j)
      d2 += (center[j] - source[j]) * (center[j] - source[j]);
    if (d2 > 0)
      for (std::size_t k = 0; k < input.centers.size(); ++k) {
        double t = 0;
        for (int j = 0; j < 3; ++j)
          t += (input.centers[k][j] - source[j]) * (center[j] - source[j]);
        t = std::clamp(t / d2, 0., 1.);
        double dist = 0;
        for (int j = 0; j < 3; ++j) {
          double v =
              source[j] + t * (center[j] - source[j]) - input.centers[k][j];
          dist += v * v;
        }
        times.push_back({dist, t});
      }
    std::sort(times.begin(), times.end());
    std::vector<double> trial{.5, .25, .75};
    for (std::size_t i = 0; i < std::min<std::size_t>(8, times.size()); ++i)
      trial.push_back(times[i].second);
    for (double t : trial) {
      s.add("fused_cross_section_tests");
      if (bound(source, r, t, t, o, cfg, s, params) == 1) {
        s.add("fused_cross_section_SHADOW");
        return 0;
      }
    }
    return -1;
  }
  Cover build(const Options &o, const Settings &cfg, Stats &s) const {
    Cover tree;
    Node root;
    root.box = {{0, 0, 0}, {1, 1, 1}};
    tree.nodes.push_back(root);
    std::vector<int> todo{0};
    auto start = Clock::now();
    while (!todo.empty()) {
      if (elapsed(start) > cfg.cover_seconds) {
        s.add("cover_budget_exhausted");
        break;
      }
      int id = todo.back();
      todo.pop_back();
      Node node = tree.nodes[id];
      s.add("cover_box_queries");
      int label = box(node.box, o, s).status;
      tree.nodes[id].label = label;
      if (label == 0) {
        s.add("cover_FREE_leaves");
        continue;
      }
      if (label == 1) {
        tree.inner.push_back(id);
        s.add("cover_SOLID_leaves");
        continue;
      }
      if (node.depth >= cfg.cover_depth ||
          int(tree.nodes.size() + 8) > cfg.cover_nodes) {
        s.add("cover_UNKNOWN_leaves");
        continue;
      }
      tree.nodes[id].split = true;
      for (int c = 0; c < 8; ++c) {
        Node child;
        child.depth = node.depth + 1;
        for (int j = 0; j < 3; ++j) {
          double m = (node.box.lo[j] + node.box.hi[j]) / 2;
          child.box.lo[j] = (c & (1 << j)) ? m : node.box.lo[j];
          child.box.hi[j] = (c & (1 << j)) ? node.box.hi[j] : m;
        }
        tree.nodes[id].children[c] = tree.nodes.size();
        todo.push_back(tree.nodes.size());
        tree.nodes.push_back(child);
      }
    }
    s.add("cover_nodes", tree.nodes.size());
    s.add("cover_pending_UNKNOWN", todo.size());
    std::stable_sort(tree.inner.begin(), tree.inner.end(), [&](int x, int y) {
      return tree.nodes[x].depth < tree.nodes[y].depth;
    });
    if (int(tree.inner.size()) > cfg.shadow_boxes)
      tree.inner.resize(cfg.shadow_boxes);
    s.add("cover_inner_used", tree.inner.size());
    return tree;
  }
  int mask(Cover &tree, int id, Box r, Stats &s) const {
    auto &n = tree.nodes[id];
    if (!overlaps(n.box, r))
      return -3;
    s.add("cover_membership_visits");
    if (n.label >= 0)
      return n.label;
    if (!n.split)
      return -1;
    int result = -3;
    for (int child : n.children) {
      int label = mask(tree, child, r, s);
      if (label == -3)
        continue;
      if (label < 0)
        return -1;
      if (result == -3)
        result = label;
      else if (result != label)
        return -1;
    }
    return result == -3 ? -1 : result;
  }
  bool excluded(Cover &tree, int id, P source, Box r, Stats &s) const {
    auto &n = tree.nodes[id];
    if (n.label == 0)
      return true;
    s.add("cover_volume_visits");
    if (!independent)
      return !possible_clip(source, r, n.box, s);
    // Projection with finite lambda represents t>0. Closed t=0 contact must
    // remain possible when the source is on/in this (outer) proof box.
    if (contained({source, source}, n.box))
      return false;
    if (!n.projected) {
      auto prep = Clock::now();
      n.planes = shadow_projection(source, n.box);
      n.projected = true;
      s.add("reference_projection_builds");
      s.add("reference_projection_build_nanoseconds",
            std::uint64_t(elapsed(prep) * 1e9));
    }
    return projected_excludes(n.planes, r, s);
  }
  bool free_volume(Cover &tree, int id, P source, Box r, Stats &s) const {
    if (excluded(tree, id, source, r, s))
      return true;
    auto &n = tree.nodes[id];
    if (!n.split)
      return false;
    for (int child : n.children)
      if (!free_volume(tree, child, source, r, s))
        return false;
    return true;
  }
  bool shadow(Cover &tree, P source, Box r, Stats &s) const {
    P center;
    for (int j = 0; j < 3; ++j)
      center[j] = (r.lo[j] + r.hi[j]) / 2;
    for (int id : tree.inner) {
      auto &n = tree.nodes[id];
      s.add("inner_shadow_candidates");
      if (independent) {
        if (!n.projected) {
          auto prep = Clock::now();
          n.planes = shadow_projection(source, n.box);
          n.projected = true;
          s.add("reference_projection_builds");
          s.add("reference_projection_build_nanoseconds",
                std::uint64_t(elapsed(prep) * 1e9));
        }
        if (projected_contains(n.planes, r, s))
          return true;
      } else {
        if (!corner_hit(source, center, n.box, s))
          continue;
        bool all = true;
        for (int c = 0; c < 8; ++c) {
          P p;
          for (int j = 0; j < 3; ++j)
            p[j] = (c & (1 << j)) ? r.hi[j] : r.lo[j];
          if (!corner_hit(source, p, n.box, s)) {
            all = false;
            break;
          }
        }
        if (all)
          return true;
      }
    }
    return false;
  }
};
Engine::Engine(const Inputs &i, bool independent)
    : p(std::make_shared<Impl>(i, independent)) {}
std::pair<int, int> Engine::region(P source, P lo, P hi, const Options &o,
                                   const Settings &cfg, Stats &s) const {
  o.validate();
  cfg.validate();
  require_environment();
  Box r{lo, hi};
  if (!finite(source) || !good(r) ||
      !contained({source, source}, {{0, 0, 0}, {1, 1, 1}}) ||
      !contained(r, {{0, 0, 0}, {1, 1, 1}}))
    return {-2, -2};
  int m = p->box(r, o, s).status;
  if (m == 1)
    return {0, 1};
  int y = -1;
  if (cfg.mode == 2 || cfg.mode == 4)
    y = p->volume(source, r, o, cfg, s);
  if (y < 0 && (cfg.mode == 3 || cfg.mode == 4))
    y = p->fused(source, r, o, cfg, s);
  if (y < 0) {
    auto old = p->old_cert(source, r, o, s, -1, m);
    return old;
  }
  return {y, y == 1 ? 0 : m};
}
Result Engine::field(P source, int n, double seconds, const Options &o,
                     const Settings &cfg) const {
  return tile(source, n, {0, 0, 0}, n, seconds, o, cfg);
}
Result Engine::tile(P source, int grid, std::array<int, 3> origin, int n,
                    double seconds, const Options &o,
                    const Settings &cfg) const {
  o.validate();
  cfg.validate();
  require_environment();
  if (!finite(source) || !contained({source, source}, {{0, 0, 0}, {1, 1, 1}}) ||
      !(seconds > 0) || !std::isfinite(seconds) ||
      (n != 8 && n != 16 && n != 32 && n != 64 && n != 128) || grid < 8 ||
      grid > 1024 || (grid & (grid - 1)) ||
      std::any_of(origin.begin(), origin.end(),
                  [&](int v) { return v < 0 || v > grid - n; }))
    throw std::invalid_argument("Bounded small-field input required");
  Result result;
  result.setup_seconds = p->setup_seconds;
  if (cfg.mode == 0 && !cfg.trace && grid == n &&
      origin == std::array<int, 3>{0, 0, 0}) {
    result.field = p->independent
                       ? p->b->field(source, n, seconds, o, cfg.mode == 0)
                       : p->a->field(source, n, seconds, o, cfg.mode == 0);
    return result;
  }
  Field &f = result.field;
  f.resolution = n;
  std::size_t total = std::size_t(n) * n * n;
  f.response.resize(total);
  f.solid.resize(total);
  f.valid.resize(total);
  if (cfg.trace) {
    result.evaluation_path.resize(total);
    result.ray_nanoseconds.resize(total);
    result.point_nanoseconds.resize(total);
  }
  auto start = Clock::now();
  Cover tree;
  bool usecover =
      cfg.mode == 5 && contained({source, source}, {{0, 0, 0}, {1, 1, 1}});
  if (usecover) {
    auto t = Clock::now();
    Settings limited = cfg;
    limited.cover_seconds = std::min(seconds, cfg.cover_seconds);
    tree = p->build(o, limited, f.stats);
    result.cover_seconds = elapsed(t);
  }
  struct Cell {
    std::array<int, 3> origin;
    int side, y, m;
  };
  std::vector<Cell> stack{{{0, 0, 0}, n, -1, -1}};
  auto index = [n](int x, int y, int z) {
    return (std::size_t(z) * n + y) * n + x;
  };
  while (!stack.empty()) {
    if (elapsed(start) > seconds) {
      f.reason = "field_time_budget";
      break;
    }
    Cell c = stack.back();
    stack.pop_back();
    f.stats.add("field_cells");
    Box r;
    for (int j = 0; j < 3; ++j) {
      r.lo[j] = (origin[j] + c.origin[j] + .5) / grid;
      r.hi[j] = (origin[j] + c.origin[j] + c.side - .5) / grid;
    }
    int m = c.m, y = c.y;
    auto ct = Clock::now();
    if (m < 0 && usecover) {
      m = p->mask(tree, 0, r, f.stats);
      if (m >= 0)
        f.stats.add("cached_membership_certificates");
    }
    if (m == 1)
      y = 0;
    if (y < 0 && usecover) {
      if (p->free_volume(tree, 0, source, r, f.stats)) {
        y = 1;
        m = 0;
        f.stats.add("cached_whole_volume_FREE");
      } else if (p->shadow(tree, source, r, f.stats)) {
        y = 0;
        f.stats.add("cached_inner_SHADOW");
      }
    }
    if (cfg.mode != 0 && (y < 0 || m < 0)) {
      auto old = p->old_cert(source, r, o, f.stats, y, m);
      y = old.first;
      m = old.second;
    }
    if (y < 0 && c.side >= 4 &&
        (cfg.mode == 2 || cfg.mode == 4 ||
         (cfg.mode == 5 && cfg.volume_after_cache))) {
      y = p->volume(source, r, o, cfg, f.stats);
      if (y == 1)
        m = 0;
    }
    if (y < 0 && c.side >= 4 && (cfg.mode == 3 || cfg.mode == 4))
      y = p->fused(source, r, o, cfg, f.stats);
    result.region_seconds += elapsed(ct);
    if (y >= 0 && m >= 0) {
      auto t = Clock::now();
      for (int z = c.origin[2]; z < c.origin[2] + c.side; ++z)
        for (int yy = c.origin[1]; yy < c.origin[1] + c.side; ++yy)
          for (int x = c.origin[0]; x < c.origin[0] + c.side; ++x) {
            auto k = index(x, yy, z);
            f.response[k] = y;
            f.solid[k] = m;
            f.valid[k] = 1;
            if (cfg.trace)
              result.evaluation_path[k] = m == 1 ? 3 : y == 1 ? 1 : 2;
          }
      f.stats.add("region_filled_voxels",
                  std::uint64_t(c.side) * c.side * c.side);
      f.stats.add("certificate_side_" + std::to_string(c.side));
      result.assembly_seconds += elapsed(t);
      continue;
    }
    if (c.side > 2 && cfg.mode != 0) {
      int h = c.side / 2;
      for (int z : {0, h})
        for (int yy : {0, h})
          for (int x : {0, h})
            stack.push_back(
                {{c.origin[0] + x, c.origin[1] + yy, c.origin[2] + z},
                 h,
                 y,
                 m});
      continue;
    }
    auto t = Clock::now();
    for (int z = c.origin[2]; z < c.origin[2] + c.side; ++z)
      for (int yy = c.origin[1]; yy < c.origin[1] + c.side; ++yy)
        for (int x = c.origin[0]; x < c.origin[0] + c.side; ++x) {
          if (elapsed(start) > seconds) {
            f.reason = "field_time_budget";
            result.fallback_seconds += elapsed(t);
            goto done;
          }
          P v{(origin[0] + x + .5) / grid, (origin[1] + yy + .5) / grid,
              (origin[2] + z + .5) / grid};
          auto k = index(x, yy, z);
          int mask = m;
          if (mask < 0) {
            auto t0 = Clock::now();
            mask = p->point(v, o, f.stats).status;
            if (cfg.trace)
              result.point_nanoseconds[k] = std::uint64_t(elapsed(t0) * 1e9);
          }
          if (mask < 0) {
            f.stats.add("unresolved_membership_voxels");
            if (cfg.trace)
              result.evaluation_path[k] = 4;
            continue;
          }
          if (mask == 1) {
            f.solid[k] = 1;
            f.valid[k] = 1;
            if (cfg.trace)
              result.evaluation_path[k] = 5;
            continue;
          }
          int blocked = 1 - y;
          if (y < 0) {
            auto t0 = Clock::now();
            blocked = p->ray(source, v, o, f.stats).status;
            if (cfg.trace)
              result.ray_nanoseconds[k] = std::uint64_t(elapsed(t0) * 1e9);
          }
          if (blocked < 0) {
            f.stats.add("unresolved_visibility_voxels");
            if (cfg.trace)
              result.evaluation_path[k] = 8;
            continue;
          }
          f.response[k] = 1 - blocked;
          f.valid[k] = 1;
          if (cfg.trace)
            result.evaluation_path[k] =
                y < 0 ? (blocked ? 7 : 6) : (blocked ? 10 : 9);
        }
    result.fallback_seconds += elapsed(t);
  }
done:
  f.complete = std::all_of(f.valid.begin(), f.valid.end(),
                           [](auto v) { return v == 1; });
  if (f.reason.empty())
    f.reason = f.complete ? "complete" : "numerically_incomplete";
  f.seconds = elapsed(start);
  return result;
}
} // namespace b2r
