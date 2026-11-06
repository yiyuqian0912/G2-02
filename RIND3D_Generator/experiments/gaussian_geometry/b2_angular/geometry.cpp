#include "angular.hpp"
#include <numeric>
#include <set>
#include <stdexcept>

namespace b2a {
bool good(Box b) {
  if (!finite(b.lo) || !finite(b.hi))
    return false;
  for (int j = 0; j < 3; ++j)
    if (b.lo[j] > b.hi[j])
      return false;
  return true;
}
bool contains(Box a, Box b) {
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
bool beam_clip(P s, const std::array<I, 3> &d, Box a, double cap, double &low,
               Stats &stats) {
  stats.add("beam_clip_tests");
  low = 0;
  double high = cap;
  for (int j = 0; j < 3; ++j) {
    // Existence requires rho*d.lo <= a.hi-s, rho*d.hi >= a.lo-s.
    if (d[j].lo == 0) {
      if (s[j] > a.hi[j])
        return false;
    } else {
      I q = divi(sub(I(a.hi[j]), I(s[j])), I(d[j].lo));
      if (d[j].lo > 0)
        high = std::min(high, q.hi);
      else
        low = std::max(low, q.lo);
    }
    if (d[j].hi == 0) {
      if (s[j] < a.lo[j])
        return false;
    } else {
      I q = divi(sub(I(a.lo[j]), I(s[j])), I(d[j].hi));
      if (d[j].hi > 0)
        low = std::max(low, q.lo);
      else
        high = std::min(high, q.hi);
    }
    if (low > high)
      return false;
  }
  return true;
}
bool volume_possible(P s, Box r, Box q, Stats &stats) {
  std::array<I, 3> d;
  for (int j = 0; j < 3; ++j)
    d[j] = sub(I(r.lo[j], r.hi[j]), I(s[j]));
  double low;
  return beam_clip(s, d, q, 1, low, stats);
}
std::vector<ExactPlane> project_exact(P s, Box b) {
  // Reference only: eliminate lambda from lambda>=1 and
  // lambda*(b.lo-s) <= x-s <= lambda*(b.hi-s), using exact binary64 rationals.
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
  std::vector<ExactPlane> out;
  for (auto &r : zero)
    out.push_back({r.a, r.rhs});
  for (auto &a : pos)
    for (auto &b : neg) {
      ExactPlane r;
      for (int j = 0; j < 3; ++j)
        r.a[j] = a.a[j] * (-b.k) + b.a[j] * a.k;
      r.rhs = a.rhs * (-b.k) + b.rhs * a.k;
      out.push_back(r);
    }
  return out;
}
std::vector<Plane> enclose_planes(const std::vector<ExactPlane> &planes) {
  std::vector<Plane> out;
  for (auto &p : planes) {
    Plane r;
    for (int j = 0; j < 3; ++j)
      r.a[j] = enclose(p.a[j]);
    r.rhs = enclose(p.rhs);
    out.push_back(r);
  }
  return out;
}
bool plane_excludes(const std::vector<Plane> &planes, Box r, Stats &s) {
  for (auto &p : planes) {
    I v;
    for (int j = 0; j < 3; ++j)
      v = add(v, mul(p.a[j], I(r.lo[j], r.hi[j])));
    s.add("reference_shadow_plane_tests");
    if (v.lo > p.rhs.hi)
      return true;
  }
  return false;
}
bool plane_contains(const std::vector<Plane> &planes, Box r, Stats &s) {
  for (auto &p : planes) {
    I v;
    for (int j = 0; j < 3; ++j)
      v = add(v, mul(p.a[j], I(r.lo[j], r.hi[j])));
    s.add("reference_shadow_plane_tests");
    if (v.hi > p.rhs.lo)
      return false;
  }
  return true;
}
namespace {
// Sufficient corner HIT. Every rho in [lo,hi] really lies in the closed box.
double inner_entry(P s, const std::array<I, 3> &d, Box q, Stats &stats) {
  double lo = 0, hi = INFINITY;
  stats.add("primary_angular_corner_clips");
  for (int j = 0; j < 3; ++j) {
    if (d[j].lo == 0 && d[j].hi == 0) {
      if (s[j] < q.lo[j] || s[j] > q.hi[j])
        return INFINITY;
      continue;
    }
    if (d[j].lo <= 0 && d[j].hi >= 0)
      return INFINITY;
    I l = divi(sub(I(q.lo[j]), I(s[j])), d[j]);
    I h = divi(sub(I(q.hi[j]), I(s[j])), d[j]);
    if (d[j].hi < 0)
      std::swap(l, h);
    lo = std::max(lo, l.hi);
    hi = std::min(hi, h.lo);
    if (lo > hi)
      return INFINITY;
  }
  return lo;
}
struct Rect {
  double u0 = -1, u1 = 1, v0 = -1, v1 = 1;
};
bool overlap(Rect a, Rect b) {
  return !(a.u0 > b.u1 || a.u1 < b.u0 || a.v0 > b.v1 || a.v1 < b.v0);
}
Rect meet(Rect a, Rect b) {
  return {std::max(a.u0, b.u0), std::min(a.u1, b.u1), std::max(a.v0, b.v0),
          std::min(a.v1, b.v1)};
}
std::array<I, 3> direction(int face, I u, I v) {
  int axis = face / 2;
  std::array<I, 3> d;
  d[axis] = I(face % 2 ? -1 : 1);
  d[(axis + 1) % 3] = u;
  d[(axis + 2) % 3] = v;
  return d;
}
} // namespace
bool primary_same_box(P s, Box r, Box q, Stats &stats) {
  if (contains({s, s}, q))
    return true;
  std::array<I, 3> center;
  for (int j = 0; j < 3; ++j)
    center[j] = sub(I((r.lo[j] + r.hi[j]) / 2), I(s[j]));
  if (inner_entry(s, center, q, stats) > 1)
    return false;
  for (int c = 0; c < 8; ++c) {
    std::array<I, 3> d;
    for (int j = 0; j < 3; ++j)
      d[j] = sub(I((c & (1 << j)) ? r.hi[j] : r.lo[j]), I(s[j]));
    if (inner_entry(s, d, q, stats) > 1)
      return false;
  }
  return true;
}
ChartQuery chart(P s, Box r) {
  ChartQuery out;
  std::array<I, 3> d;
  for (int j = 0; j < 3; ++j)
    d[j] = sub(I(r.lo[j], r.hi[j]), I(s[j]));
  for (int axis = 0; axis < 3; ++axis)
    for (int sign : {1, -1}) {
      I rho = sign > 0 ? d[axis] : neg(d[axis]);
      int a = (axis + 1) % 3, b = (axis + 2) % 3;
      if (!(rho.lo > 0) ||
          rho.lo < std::max(std::abs(d[a].lo), std::abs(d[a].hi)) ||
          rho.lo < std::max(std::abs(d[b].lo), std::abs(d[b].hi)))
        continue;
      I u = divi(d[a], rho), v = divi(d[b], rho);
      out = {2 * axis + (sign < 0),
             std::max(-1., u.lo),
             std::min(1., u.hi),
             std::max(-1., v.lo),
             std::min(1., v.hi),
             rho.lo,
             rho.hi};
      return out;
    }
  return out;
}
void AngularOptions::validate() const {
  if (depth < 0 || depth > 18 || nodes < 6 || nodes > 1000000 ||
      references < 0 || references > 10000000 || prefix_builds < 0 ||
      prefix_builds > 100000)
    throw std::invalid_argument("Invalid bounded angular map budget");
}
struct AngularMap::Impl {
  struct Body {
    Box box;
    std::array<Rect, 6> bounds;
    std::array<bool, 6> active{};
    bool prepared = false;
    std::vector<ExactPlane> exact;
    std::vector<Plane> planes;
  };
  struct Node {
    Rect uv;
    int face = 0, depth = 0;
    std::array<int, 4> child{};
    bool split = false;
    double H = INFINITY, L = -1;
    int witness = -1;
    std::size_t seen = 0;
    std::size_t prefix_generation = std::numeric_limits<std::size_t>::max();
    std::vector<int> candidate, proof;
    bool tested = false;
  };
  P source;
  bool reference;
  AngularOptions cfg;
  Stats &stats;
  std::vector<Body> bodies;
  std::vector<Node> nodes;
  std::uint64_t refs = 0;
  std::size_t generation = 0;
  int prefix_builds = 0;
  Impl(P s, bool ref, const AngularOptions &c, Stats &work)
      : source(s), reference(ref), cfg(c), stats(work) {
    cfg.validate();
    require_environment();
    if (!finite(source))
      throw std::invalid_argument("Nonfinite source");
    for (int f = 0; f < 6; ++f) {
      Node n;
      n.face = f;
      nodes.push_back(n);
    }
  }
  void insert(Box q) {
    if (!good(q))
      throw std::invalid_argument("Invalid inner box");
    Body b;
    b.box = q;
    for (int f = 0; f < 6; ++f) {
      int axis = f / 2, a = (axis + 1) % 3, c = (axis + 2) % 3;
      I z = sub(I(q.lo[axis], q.hi[axis]), I(source[axis]));
      if (f % 2)
        z = neg(z);
      if (z.hi <= 0 && !contains({source, source}, q))
        continue;
      b.active[f] = true;
      if (z.lo <= 0)
        b.bounds[f] = {};
      else {
        I u = divi(sub(I(q.lo[a], q.hi[a]), I(source[a])), z);
        I v = divi(sub(I(q.lo[c], q.hi[c]), I(source[c])), z);
        b.bounds[f] = {std::max(-1., u.lo), std::min(1., u.hi),
                       std::max(-1., v.lo), std::min(1., v.hi)};
        if (b.bounds[f].u0 > b.bounds[f].u1 || b.bounds[f].v0 > b.bounds[f].v1)
          b.active[f] = false;
      }
    }
    bodies.push_back(std::move(b));
    stats.add("angular_inner_boxes");
  }
  void refresh(int id) {
    auto &n = nodes[id];
    if (n.seen == bodies.size())
      return;
    for (std::size_t j = n.seen; j < bodies.size(); ++j) {
      auto &b = bodies[j];
      if (b.active[n.face] && overlap(n.uv, b.bounds[n.face])) {
        if (refs >= std::uint64_t(cfg.references)) {
          stats.add("angular_reference_budget_exhausted");
          break;
        }
        n.candidate.push_back(j);
        ++refs;
      }
    }
    n.seen = bodies.size();
    n.tested = false;
  }
  double upper(int body, int face, Rect uv) {
    Body &b = bodies[body];
    stats.add("angular_body_tile_tests");
    if (contains({source, source}, b.box))
      return 0;
    if (!reference) {
      double h = 0;
      for (double u : {uv.u0, uv.u1})
        for (double v : {uv.v0, uv.v1}) {
          double x =
              inner_entry(source, direction(face, I(u), I(v)), b.box, stats);
          if (!std::isfinite(x))
            return INFINITY;
          h = std::max(h, x);
        }
      return h;
    }
    // Independent proof: exact rational projected halfspaces, tested at a
    // far-axis plane. Interval screen first, exact rational corner fallback.
    if (!b.prepared) {
      b.exact = project_exact(source, b.box);
      b.planes = enclose_planes(b.exact);
      b.prepared = true;
      stats.add("angular_reference_projection_builds");
    }
    int axis = face / 2;
    Q depth = face % 2 ? Q(source[axis]) - Q(b.box.lo[axis])
                       : Q(b.box.hi[axis]) - Q(source[axis]);
    if (depth <= 0)
      return INFINITY;
    double h = enclose(depth).hi;
    auto dirs = direction(face, I(uv.u0, uv.u1), I(uv.v0, uv.v1));
    for (std::size_t k = 0; k < b.planes.size(); ++k) {
      I val;
      for (int j = 0; j < 3; ++j)
        val = add(val,
                  mul(b.planes[k].a[j], add(I(source[j]), mul(I(h), dirs[j]))));
      stats.add("angular_reference_plane_tests");
      if (val.hi <= b.planes[k].rhs.lo)
        continue;
      // The four exact corner evaluations prove this linear inequality on U.
      const auto &p = b.exact[k];
      for (double u : {uv.u0, uv.u1})
        for (double v : {uv.v0, uv.v1}) {
          std::array<Q, 3> d;
          d[axis] = face % 2 ? -1 : 1;
          d[(axis + 1) % 3] = Q(u);
          d[(axis + 2) % 3] = Q(v);
          Q value;
          for (int j = 0; j < 3; ++j)
            value += p.a[j] * (Q(source[j]) + Q(h) * d[j]);
          stats.add("angular_reference_exact_corner_tests");
          if (value > p.rhs)
            return INFINITY;
        }
    }
    return h;
  }
  void test(int id) {
    refresh(id);
    auto &n = nodes[id];
    if (n.tested)
      return;
    n.tested = true;
    for (int k : n.candidate) {
      double h = upper(k, n.face, n.uv);
      if (h < n.H) {
        n.H = h;
        n.witness = k;
        n.proof = {k};
      }
      // First useful certificate is enough. Trying every smaller box is costly.
      if (std::isfinite(n.H))
        break;
    }
    if (std::isfinite(n.H))
      stats.add("angular_single_tile_proofs");
  }
  bool split(int id) {
    if (nodes[id].split)
      return true;
    if (nodes[id].depth >= cfg.depth || int(nodes.size() + 4) > cfg.nodes) {
      stats.add("angular_node_or_depth_unknown");
      return false;
    }
    Node parent = nodes[id];
    double um = (parent.uv.u0 + parent.uv.u1) / 2,
           vm = (parent.uv.v0 + parent.uv.v1) / 2;
    nodes[id].split = true;
    for (int c = 0; c < 4; ++c) {
      Node child;
      child.face = parent.face;
      child.depth = parent.depth + 1;
      child.uv = {(c & 1) ? um : parent.uv.u0, (c & 1) ? parent.uv.u1 : um,
                  (c & 2) ? vm : parent.uv.v0, (c & 2) ? parent.uv.v1 : vm};
      // Child candidate construction can be restricted to its parent's set.
      child.seen = parent.seen;
      for (int k : parent.candidate)
        if (overlap(child.uv, bodies[k].bounds[child.face])) {
          if (refs >= std::uint64_t(cfg.references))
            break;
          child.candidate.push_back(k);
          ++refs;
        }
      nodes[id].child[c] = nodes.size();
      nodes.push_back(std::move(child));
    }
    stats.add("angular_subdivisions");
    stats.maximum("angular_max_depth", parent.depth + 1);
    return true;
  }
  bool covered(int id, Rect query, double near, int &work, std::set<int> &used,
               double &height) {
    if (++work > 4096) {
      stats.add("angular_query_work_unknown");
      return false;
    }
    stats.add("angular_query_node_visits");
    test(id);
    if (nodes[id].H <= near) {
      used.insert(nodes[id].proof.begin(), nodes[id].proof.end());
      height = std::max(height, nodes[id].H);
      return true;
    }
    if (nodes[id].candidate.empty())
      return false;
    if (!cfg.fusion) {
      // Single-occluder cache: only descend when one child contains all U.
      if (!split(id))
        return false;
      for (int child : nodes[id].child) {
        Rect r = nodes[child].uv;
        if (query.u0 >= r.u0 && query.u1 <= r.u1 && query.v0 >= r.v0 &&
            query.v1 <= r.v1)
          return covered(child, query, near, work, used, height);
      }
      return false;
    }
    if (!split(id))
      return false;
    auto children = nodes[id].child;
    for (int child : children)
      if (overlap(query, nodes[child].uv))
        if (!covered(child, meet(query, nodes[child].uv), near, work, used,
                     height))
          return false;
    // Promote only complete angular children, never a partially covered query.
    double h = 0;
    bool all = true;
    for (int child : children) {
      all &= std::isfinite(nodes[child].H);
      h = std::max(h, nodes[child].H);
    }
    if (all && h < nodes[id].H) {
      std::set<int> proof;
      for (int child : children)
        proof.insert(nodes[child].proof.begin(), nodes[child].proof.end());
      nodes[id].H = h;
      nodes[id].witness = -2;
      nodes[id].proof.assign(proof.begin(), proof.end());
      stats.add("angular_promoted_union_tiles");
    }
    return true;
  }
  AngularDecision
  query(Box r, const std::function<double(int, double, double, double, double)>
                   &prefix) {
    if (!good(r))
      throw std::invalid_argument("Invalid receiver box");
    AngularDecision out;
    auto c = chart(source, r);
    stats.add("angular_queries");
    if (c.face < 0) {
      stats.add("angular_cross_chart_unknown");
      return out;
    }
    Rect q{c.u0, c.u1, c.v0, c.v1};
    int work = 0;
    std::set<int> used;
    double height = 0;
    if (covered(c.face, q, c.near, work, used, height)) {
      out.visibility = 0;
      out.first_hit_upper = height;
      out.distinct_blockers = used.size();
      stats.add("angular_shadow_queries");
      if (used.size() > 1)
        stats.add("angular_multi_blocker_queries");
    } else if (cfg.empty_prefix && prefix) {
      // Cache at the deepest node containing the whole query. A prefix proof
      // is independent of inner blockers and must inspect the full outer cover.
      int id = c.face;
      while (nodes[id].depth < cfg.depth) {
        refresh(id);
        if (!split(id))
          break;
        int next = -1;
        for (int child : nodes[id].child) {
          auto uv = nodes[child].uv;
          if (q.u0 >= uv.u0 && q.u1 <= uv.u1 && q.v0 >= uv.v0 &&
              q.v1 <= uv.v1) {
            next = child;
            break;
          }
        }
        if (next < 0)
          break;
        id = next;
      }
      auto &n = nodes[id];
      if (n.prefix_generation != generation &&
          prefix_builds < cfg.prefix_builds) {
        n.L = std::max(n.L, prefix(n.face, n.uv.u0, n.uv.u1, n.uv.v0, n.uv.v1));
        n.prefix_generation = generation;
        ++prefix_builds;
        stats.add("empty_prefix_builds");
      } else if (n.prefix_generation != generation)
        stats.add("empty_prefix_build_budget_exhausted");
      out.empty_prefix = n.L;
      if (c.far < n.L) {
        out.visibility = 1;
        stats.add("angular_empty_prefix_queries");
      }
    }
    stats.maximum("angular_nodes", nodes.size());
    stats.maximum("angular_references", refs);
    return out;
  }
};
AngularMap::AngularMap(P s, bool ref, const AngularOptions &cfg, Stats &stats)
    : p(std::make_shared<Impl>(s, ref, cfg, stats)) {}
void AngularMap::add_inner(Box q) { p->insert(q); }
void AngularMap::outer_refined() { ++p->generation; }
AngularDecision AngularMap::query(
    Box r,
    const std::function<double(int, double, double, double, double)> &f) {
  return p->query(r, f);
}
std::uint64_t AngularMap::bytes() const {
  std::uint64_t total = p->nodes.capacity() * sizeof(Impl::Node) +
                        p->bodies.capacity() * sizeof(Impl::Body);
  for (auto &n : p->nodes)
    total += (n.candidate.capacity() + n.proof.capacity()) * sizeof(int);
  for (auto &b : p->bodies) {
    total += b.planes.capacity() * sizeof(Plane) +
             b.exact.capacity() * sizeof(ExactPlane);
  }
  // GMP limb allocations are additionally reflected by process RSS.
  return total;
}
std::vector<std::array<double, 8>> AngularMap::snapshot() const {
  std::vector<std::array<double, 8>> out;
  for (auto &n : p->nodes)
    if (!n.split)
      out.push_back({double(n.face), n.uv.u0, n.uv.u1, n.uv.v0, n.uv.v1, n.H,
                     n.L, double(n.depth)});
  return out;
}
} // namespace b2a
