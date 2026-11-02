#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <xmmintrin.h>
namespace py = pybind11;
using Q = mpq_class;
using P = std::array<double, 3>;
using Rows = std::vector<std::vector<std::vector<std::string>>>;
void require_environment() {
  if (std::fegetround() != FE_TONEAREST ||
      (_mm_getcsr() & ((1 << 15) | (1 << 6))))
    throw std::runtime_error(
        "Certified intervals require round-to-nearest and gradual underflow");
}

struct I {
  double lo, hi;
  I(double x = 0) : lo(x), hi(x) {}
  I(double a, double b) : lo(a), hi(b) {}
};
// Equivalent to nextafter(x,+/-infinity), including signed zero/subnormals.
double step(double x, bool positive) {
  if (std::isnan(x) || x == (positive ? INFINITY : -INFINITY))
    return x;
  if (x == 0)
    return positive ? std::numeric_limits<double>::denorm_min()
                    : -std::numeric_limits<double>::denorm_min();
  auto b = std::bit_cast<std::uint64_t>(x);
  b += (positive == (x > 0)) ? 1 : std::uint64_t(-1);
  return std::bit_cast<double>(b);
}
I whole() { return {-INFINITY, INFINITY}; }
I add(I a, I b) {
  double l = a.lo + b.lo, h = a.hi + b.hi;
  return std::isnan(l) || std::isnan(h) ? whole()
                                        : I(step(l, false), step(h, true));
}
I sub(I a, I b) { return add(a, {-b.hi, -b.lo}); }
I mul(I a, I b) {
  if ((a.lo == 0 && a.hi == 0) || (b.lo == 0 && b.hi == 0))
    return I(0);
  double v[] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  for (double x : v)
    if (std::isnan(x))
      return whole();
  return {step(*std::min_element(v, v + 4), false),
          step(*std::max_element(v, v + 4), true)};
}
I divi(I a, I b) {
  if (b.lo <= 0 && b.hi >= 0)
    return whole();
  return mul(a, {step(1 / b.hi, false), step(1 / b.lo, true)});
}
I enclosure(const Q &q) {
  double x = q.get_d();
  if (!std::isfinite(x))
    throw std::invalid_argument("Nonfinite coefficient");
  double a = x, b = x;
  while (Q(a) > q)
    a = step(a, false);
  while (Q(b) < q)
    b = step(b, true);
  return {a, b};
}
struct Plane {
  std::array<Q, 4> q;
  std::array<I, 4> v;
  Plane(const std::vector<std::string> &r) {
    if (r.size() != 4)
      throw std::invalid_argument("Four coefficients required");
    for (int i = 0; i < 4; ++i) {
      q[i] = Q(r[i]);
      q[i].canonicalize();
      v[i] = enclosure(q[i]);
    }
  }
  I range(P lo, P hi) const {
    I s = v[3];
    for (int a = 0; a < 3; ++a)
      if (q[a] != 0)
        s = add(s, mul(v[a], {lo[a], hi[a]}));
    return s;
  }
  Q exact(P p) const {
    Q s = q[3];
    for (int a = 0; a < 3; ++a)
      s += q[a] * Q(p[a]);
    return s;
  }
};
using Poly = std::vector<Plane>;
// Independent projection evaluator: different summation order, std::nextafter
// rather than the clipping evaluator's bit-step arithmetic. Only exact rational
// plane extraction is shared; uncertain point signs use GMP.
I reference_range(const Plane &f, P lo, P hi) {
  double lower = f.v[3].lo, upper = f.v[3].hi;
  for (int a = 2; a >= 0; --a) {
    if (f.q[a] == 0)
      continue;
    double v[] = {f.v[a].lo * lo[a], f.v[a].lo * hi[a], f.v[a].hi * lo[a],
                  f.v[a].hi * hi[a]};
    for (double x : v)
      if (!std::isfinite(x))
        return whole();
    double l = std::nextafter(*std::min_element(v, v + 4), -INFINITY);
    double h = std::nextafter(*std::max_element(v, v + 4), INFINITY);
    lower = std::nextafter(lower + l, -INFINITY);
    upper = std::nextafter(upper + h, INFINITY);
    if (std::isnan(lower) || std::isnan(upper))
      return whole();
  }
  return {lower, upper};
}
int reference_union(const std::vector<Poly> &bodies, P lo, P hi) {
  bool unknown = false;
  for (const auto &body : bodies) {
    bool outside = false, inside = true;
    for (const auto &plane : body) {
      I v = reference_range(plane, lo, hi);
      if (v.lo > 0) {
        outside = true;
        break;
      }
      inside &= v.hi <= 0;
    }
    if (inside && !outside)
      return 1;
    unknown |= !outside;
  }
  return unknown ? -1 : 0;
}
std::vector<Poly> parse(const Rows &r) {
  std::vector<Poly> p;
  for (auto &b : r) {
    if (b.empty())
      throw std::invalid_argument("Empty convex body");
    p.emplace_back();
    for (auto &v : b)
      p.back().emplace_back(v);
  }
  return p;
}
using VQ = std::array<Q, 3>;
VQ cross(const Plane &a, const Plane &b) {
  return {a.q[1] * b.q[2] - a.q[2] * b.q[1], a.q[2] * b.q[0] - a.q[0] * b.q[2],
          a.q[0] * b.q[1] - a.q[1] * b.q[0]};
}
Q dot(const Plane &a, const VQ &v) {
  Q s;
  for (int k = 0; k < 3; ++k)
    s += a.q[k] * v[k];
  return s;
}
// Exact recession-cone boundedness test, same mathematical criterion as the
// canonical halfspace backend. Python also supplies a strict interior witness.
void bounded(const Poly &p) {
  bool rank3 = false;
  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      auto v = cross(p[i], p[j]);
      if (v[0] == 0 && v[1] == 0 && v[2] == 0)
        continue;
      bool pos = true, neg = true;
      for (auto &a : p) {
        Q d = dot(a, v);
        pos &= d <= 0;
        neg &= d >= 0;
        rank3 |= d != 0;
      }
      if (pos || neg)
        throw std::invalid_argument("Unbounded recession cone");
    }
  if (!rank3)
    throw std::invalid_argument("Unbounded rank-deficient halfspaces");
}
struct Geometry {
  std::vector<Poly> bodies;
  std::vector<std::vector<VQ>> vertices;
  py::dict proof(const std::vector<P> &anchors);
  py::array_t<std::int8_t>
  contains(py::array_t<double, py::array::c_style> points) const;
  py::array_t<std::int8_t>
  membership(py::array_t<double, py::array::c_style> points) const;
  int box_class(P lo, P hi) const;
  std::vector<int> clearance_planes(P point, double radius) const;
  explicit Geometry(const Rows &r) : bodies(parse(r)) {
    require_environment();
    for (auto &p : bodies)
      bounded(p);
  }
};
struct Stats {
  std::uint64_t nodes = 0, covered = 0, rays = 0, points = 0, clips = 0,
                exact = 0;
};
py::dict report(const Stats &s) {
  py::dict d;
  d["nodes"] = s.nodes;
  d["certificate_covered_voxels"] = s.covered;
  d["point_rays"] = s.rays;
  d["solid_points"] = s.points;
  d["interval_clips"] = s.clips;
  d["exact_fallbacks"] = s.exact;
  return d;
}
int classify(const Poly &p, P lo, P hi) {
  bool in = true;
  for (auto &f : p) {
    I v = f.range(lo, hi);
    if (v.lo > 0)
      return 0;
    in &= v.hi <= 0;
  }
  return in ? 1 : -1;
}
int united(const std::vector<Poly> &p, P lo, P hi) {
  bool unknown = false;
  for (auto &b : p) {
    int v = classify(b, lo, hi);
    if (v == 1)
      return 1;
    unknown |= v < 0;
  }
  return unknown ? -1 : 0;
}
Q value(const Plane &f, const VQ &p) { return dot(f, p) + f.q[3]; }
bool exact_member(const std::vector<Poly> &bodies, const VQ &p) {
  for (auto &b : bodies) {
    bool in = true;
    for (auto &f : b)
      if (value(f, p) > 0) {
        in = false;
        break;
      }
    if (in)
      return true;
  }
  return false;
}
VQ rational(P p) { return {Q(p[0]), Q(p[1]), Q(p[2])}; }
void valid_point(P p) {
  for (double x : p)
    if (!std::isfinite(x) || x < 0 || x > 1)
      throw std::invalid_argument("Invalid world point");
}

py::dict bound_dual(const Poly &planes, const std::vector<VQ> &vertices,
                    int axis, int sign, const Q &bound) {
  VQ target{Q(0), Q(0), Q(0)};
  target[axis] = sign;
  for (const auto &vertex : vertices) {
    if (vertex[axis] * sign != bound)
      continue;
    std::vector<int> active;
    for (std::size_t i = 0; i < planes.size(); ++i)
      if (value(planes[i], vertex) == 0)
        active.push_back(i);
    for (int count = 1; count <= 3; ++count)
      for (std::size_t i = 0; i < active.size(); ++i)
        for (std::size_t j = 0; j < (count >= 2 ? i : 1); ++j)
          for (std::size_t k = 0; k < (count == 3 ? j : 1); ++k) {
            std::vector<int> ids{active[i]};
            if (count >= 2)
              ids.push_back(active[j]);
            if (count == 3)
              ids.push_back(active[k]);
            std::vector<std::vector<Q>> a(count, std::vector<Q>(count + 1));
            for (int r = 0; r < count; ++r) {
              for (int c = 0; c < count; ++c)
                for (int d = 0; d < 3; ++d)
                  a[r][c] += planes[ids[r]].q[d] * planes[ids[c]].q[d];
              a[r][count] = dot(planes[ids[r]], target);
            }
            bool invertible = true;
            for (int c = 0; c < count; ++c) {
              int pivot = c;
              while (pivot < count && a[pivot][c] == 0)
                ++pivot;
              if (pivot == count) {
                invertible = false;
                break;
              }
              std::swap(a[c], a[pivot]);
              Q scale = a[c][c];
              for (int d = c; d <= count; ++d)
                a[c][d] /= scale;
              for (int r = 0; r < count; ++r)
                if (r != c) {
                  Q factor = a[r][c];
                  for (int d = c; d <= count; ++d)
                    a[r][d] -= factor * a[c][d];
                }
            }
            if (!invertible)
              continue;
            bool nonnegative = true;
            VQ normal{Q(0), Q(0), Q(0)};
            Q rhs;
            for (int r = 0; r < count; ++r) {
              Q w = a[r][count];
              nonnegative &= w >= 0;
              for (int d = 0; d < 3; ++d)
                normal[d] += w * planes[ids[r]].q[d];
              rhs -= w * planes[ids[r]].q[3];
            }
            if (!nonnegative || normal != target || rhs != bound)
              continue;
            std::vector<std::string> weights;
            for (int r = 0; r < count; ++r)
              weights.push_back(a[r][count].get_str());
            py::dict result;
            result["axis"] = axis;
            result["sign"] = sign;
            result["planes"] = ids;
            result["weights"] = weights;
            result["bound"] = rhs.get_str();
            return result;
          }
  }
  throw std::runtime_error("Exact LP bound dual certificate unavailable");
}

py::dict Geometry::proof(const std::vector<P> &anchors) {
  require_environment();
  if (anchors.size() != bodies.size())
    throw std::invalid_argument(
        "One exact interior anchor per component required");
  vertices.clear();
  py::list reports;
  for (std::size_t b = 0; b < bodies.size(); ++b) {
    auto &p = bodies[b];
    P c = anchors[b];
    valid_point(c);
    Q radius(1);
    for (double x : c) {
      if (x <= 0 || x >= 1)
        throw std::invalid_argument("Anchor outside open world");
      radius = std::min(radius, Q(x));
      Q world_hi = Q(1) - Q(x);
      radius = std::min(radius, world_hi);
    }
    for (auto &f : p) {
      Q slack = -f.exact(c), norm1;
      for (int a = 0; a < 3; ++a)
        norm1 += abs(f.q[a]);
      if (slack <= 0 || norm1 <= 0)
        throw std::invalid_argument("Anchor is not a strict interior witness");
      Q r = slack / norm1;
      if (r < radius)
        radius = r;
    }
    radius /= 2;
    std::vector<VQ> vv;
    for (std::size_t i = 0; i < p.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        for (std::size_t k = 0; k < j; ++k) {
          auto a = cross(p[j], p[k]), d = cross(p[k], p[i]),
               e = cross(p[i], p[j]);
          Q det = dot(p[i], a);
          if (det == 0)
            continue;
          VQ v;
          for (int z = 0; z < 3; ++z)
            v[z] =
                (-p[i].q[3] * a[z] - p[j].q[3] * d[z] - p[k].q[3] * e[z]) / det;
          bool inside = true;
          for (auto &f : p)
            if (value(f, v) > 0) {
              inside = false;
              break;
            }
          if (inside && std::find(vv.begin(), vv.end(), v) == vv.end())
            vv.push_back(v);
        }
    if (vv.size() < 4)
      throw std::invalid_argument(
          "Bounded full-dimensional vertex certificate failed");
    int faces = 0;
    for (auto &f : p) {
      std::vector<VQ> face;
      for (auto &v : vv)
        if (value(f, v) == 0)
          face.push_back(v);
      bool area = false;
      if (face.size() >= 3) {
        for (std::size_t i = 1; i < face.size() && !area; ++i)
          for (std::size_t j = 1; j < i; ++j) {
            VQ u, v;
            for (int a = 0; a < 3; ++a) {
              u[a] = face[i][a] - face[0][a];
              v[a] = face[j][a] - face[0][a];
            }
            area = (u[1] * v[2] - u[2] * v[1] != 0) ||
                   (u[2] * v[0] - u[0] * v[2] != 0) ||
                   (u[0] * v[1] - u[1] * v[0] != 0);
            if (area)
              break;
          }
      }
      faces += area;
    }
    Q gap(1);
    std::array<Q, 3> lo = vv[0], hi = vv[0];
    for (auto &v : vv)
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], v[a]);
        hi[a] = std::max(hi[a], v[a]);
        gap = std::min(gap, v[a]);
        Q h = Q(1) - v[a];
        gap = std::min(gap, h);
      }
    py::dict report;
    py::list values, lows, highs;
    for (auto &v : vv) {
      std::vector<std::string> row;
      for (auto &q : v)
        row.push_back(q.get_str());
      values.append(row);
    }
    for (int a = 0; a < 3; ++a) {
      lows.append(lo[a].get_str());
      highs.append(hi[a].get_str());
    }
    report["vertices"] = values;
    report["lower"] = lows;
    report["upper"] = highs;
    report["actual_faces"] = faces;
    report["declared_planes"] = p.size();
    report["interior_radius"] = radius.get_str();
    report["world_gap"] = gap.get_str();
    py::list duals;
    for (int axis = 0; axis < 3; ++axis) {
      duals.append(bound_dual(p, vv, axis, 1, hi[axis]));
      duals.append(bound_dual(p, vv, axis, -1, -lo[axis]));
    }
    report["bound_duals"] = duals;
    report["strictly_inside_world"] = gap > 0;
    reports.append(report);
    vertices.push_back(vv);
  }
  py::dict result;
  result["components"] = reports;
  result["bounded_full_dimensional"] = true;
  return result;
}

py::array_t<std::int8_t>
Geometry::contains(py::array_t<double, py::array::c_style> p) const {
  require_environment();
  if (p.ndim() != 2 || p.shape(1) != 3)
    throw std::invalid_argument("[N,3] points required");
  py::array_t<std::int8_t> out(p.shape(0));
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < p.shape(0); ++i) {
      P x{p.data()[3 * i], p.data()[3 * i + 1], p.data()[3 * i + 2]};
      valid_point(x);
      int v = united(bodies, x, x);
      out.mutable_data()[i] = v >= 0 ? v : exact_member(bodies, rational(x));
    }
  }
  return out;
}

py::array_t<std::int8_t>
Geometry::membership(py::array_t<double, py::array::c_style> p) const {
  require_environment();
  if (p.ndim() != 2 || p.shape(1) != 3)
    throw std::invalid_argument("[N,3] points required");
  py::array_t<std::int8_t> out(p.shape(0));
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < p.shape(0); ++i) {
      P x{p.data()[3 * i], p.data()[3 * i + 1], p.data()[3 * i + 2]};
      valid_point(x);
      std::vector<std::size_t> touching;
      bool interior = false;
      for (std::size_t k = 0; k < bodies.size(); ++k) {
        bool in = true, strict = true;
        for (auto &f : bodies[k]) {
          Q q = f.exact(x);
          if (q > 0) {
            in = false;
            break;
          }
          strict &= q < 0;
        }
        if (in) {
          interior |= strict;
          touching.push_back(k);
        }
      }
      int result = interior ? 1 : touching.empty() ? 0 : -1;
      if (!touching.empty() && std::any_of(x.begin(), x.end(), [](double a) {
            return a == 0 || a == 1;
          }))
        result = -1;
      else if (!interior && !touching.empty()) {
        if (touching.size() == 1)
          result = 2;
        else
          for (int a = 0; a < 3 && result < 0; ++a)
            for (int sign : {-1, 1}) {
              bool outside_all = true;
              for (auto k : touching) {
                bool leaves = false;
                for (auto &f : bodies[k])
                  if (f.exact(x) == 0 && f.q[a] * sign > 0) {
                    leaves = true;
                    break;
                  }
                outside_all &= leaves;
              }
              if (outside_all) {
                result = 2;
                break;
              }
            }
      }
      out.mutable_data()[i] = result;
    }
  }
  return out;
}

int Geometry::box_class(P lo, P hi) const {
  require_environment();
  valid_point(lo);
  valid_point(hi);
  for (int a = 0; a < 3; ++a)
    if (lo[a] > hi[a])
      throw std::invalid_argument("Inverted box");
  int v = united(bodies, lo, hi);
  if (v >= 0)
    return v;
  bool has_free = false, has_solid = false;
  for (int k = 0; k < 27; ++k) {
    int t = k;
    VQ p;
    for (int a = 0; a < 3; ++a) {
      int s = t % 3;
      t /= 3;
      p[a] = s == 0 ? Q(lo[a]) : s == 1 ? (Q(lo[a]) + Q(hi[a])) / 2 : Q(hi[a]);
    }
    bool m = exact_member(bodies, p);
    has_solid |= m;
    has_free |= !m;
  }
  bool all_disjoint = !vertices.empty();
  for (auto &vv : vertices) {
    bool disjoint = false;
    for (int a = 0; a < 3; ++a) {
      Q l = vv[0][a], h = l;
      for (auto &p : vv) {
        l = std::min(l, p[a]);
        h = std::max(h, p[a]);
      }
      disjoint |= Q(hi[a]) < l || Q(lo[a]) > h;
    }
    all_disjoint &= disjoint;
    for (auto &p : vv) {
      bool in = true;
      for (int a = 0; a < 3; ++a)
        in &= p[a] >= Q(lo[a]) && p[a] <= Q(hi[a]);
      has_solid |= in;
    }
  }
  if (all_disjoint)
    return 0;
  return has_free && has_solid ? 2 : -1;
}

std::vector<int> Geometry::clearance_planes(P point, double radius) const {
  require_environment();
  valid_point(point);
  if (!std::isfinite(radius) || radius < 0)
    throw std::invalid_argument("Invalid clearance radius");
  std::vector<int> result;
  Q rr = Q(radius) * Q(radius);
  for (auto &b : bodies) {
    int selected = -1;
    for (std::size_t i = 0; i < b.size(); ++i) {
      Q s = b[i].exact(point), norm;
      for (int a = 0; a < 3; ++a)
        norm += b[i].q[a] * b[i].q[a];
      if (s > 0 && s * s > rr * norm) {
        selected = i;
        break;
      }
    }
    if (selected < 0)
      return {};
    result.push_back(selected);
  }
  return result;
}
class Model {
public:
  std::shared_ptr<Geometry> g;
  P source;
  std::vector<std::vector<I>> score;
  std::vector<std::vector<Q>> qscore;
  std::vector<Poly> shadow;
  Model(std::shared_ptr<Geometry> v, P s, const Rows &rows)
      : g(std::move(v)), source(s), shadow(parse(rows)) {
    for (double x : s)
      if (!std::isfinite(x) || x <= 0 || x >= 1)
        throw std::invalid_argument("Source outside open world");
    for (auto &p : g->bodies) {
      std::vector<I> a;
      std::vector<Q> b;
      bool outside = false;
      for (auto &f : p) {
        Q x = f.exact(s);
        a.push_back(enclosure(x));
        b.push_back(x);
        outside |= x > 0;
      }
      if (!outside)
        throw std::invalid_argument("Source in closed obstacle");
      score.push_back(a);
      qscore.push_back(b);
    }
  }
  int clip(std::size_t k, P lo, P hi, Stats &st) const {
    ++st.clips;
    I enter(0), exit(1);
    auto &p = g->bodies[k];
    bool uncertain = false;
    for (std::size_t i = 0; i < p.size(); ++i) {
      I a = score[k][i], b = p[i].range(lo, hi), d = sub(b, a);
      if (a.lo > 0 && b.lo > 0)
        return 0;
      if (d.lo <= 0 && d.hi >= 0) {
        uncertain = true;
        continue;
      } // ignore constraint => conservative superset
      I t = divi({-a.hi, -a.lo}, d);
      if (d.lo > 0)
        exit = {std::min(exit.lo, t.lo), std::min(exit.hi, t.hi)};
      else
        enter = {std::max(enter.lo, t.lo), std::max(enter.hi, t.hi)};
      if (enter.lo > exit.hi)
        return 0;
    }
    return !uncertain && enter.hi < exit.lo ? 1 : -1;
  }
  bool exact_clip(std::size_t k, P x, Stats &st) const {
    ++st.exact;
    Q enter(0), exit(1);
    auto &p = g->bodies[k];
    for (std::size_t i = 0; i < p.size(); ++i) {
      Q a = qscore[k][i], d = p[i].exact(x) - a;
      if (d == 0) {
        if (a > 0)
          return false;
        continue;
      }
      Q t = -a / d;
      if (d > 0) {
        if (t < exit)
          exit = t;
      } else if (t > enter)
        enter = t;
      if (enter > exit)
        return false;
    }
    return true;
  }
  int solid(P x, Stats &st) const {
    int v = united(g->bodies, x, x);
    if (v >= 0)
      return v;
    ++st.exact;
    for (auto &p : g->bodies) {
      bool in = true;
      for (auto &f : p)
        if (f.exact(x) > 0) {
          in = false;
          break;
        }
      if (in)
        return 1;
    }
    return 0;
  }
  int visible(P x, Stats &st) const {
    for (std::size_t k = 0; k < g->bodies.size(); ++k) {
      int v = clip(k, x, x, st);
      if (v == 1 || (v < 0 && exact_clip(k, x, st)))
        return 0;
    }
    return 1;
  }
  int projected_visible_exact(P x, Stats &st) const {
    ++st.exact;
    return exact_member(shadow, rational(x)) ? 0 : 1;
  }
  int reference_solid(P x, Stats &st) const {
    int v = reference_union(g->bodies, x, x);
    if (v >= 0)
      return v;
    ++st.exact;
    return exact_member(g->bodies, rational(x));
  }
  py::tuple first_hits(py::array_t<double, py::array::c_style> points) const {
    require_environment();
    if (points.ndim() != 2 || points.shape(1) != 3)
      throw std::invalid_argument("[N,3] required");
    py::array_t<int> bodies(points.shape(0));
    py::array_t<double> entries(points.shape(0));
    {
      py::gil_scoped_release release;
      for (py::ssize_t j = 0; j < points.shape(0); ++j) {
        P p{points.data()[3 * j], points.data()[3 * j + 1],
            points.data()[3 * j + 2]};
        valid_point(p);
        int selected = -1;
        Q best(2);
        for (std::size_t k = 0; k < g->bodies.size(); ++k) {
          Q enter(0), exit(1);
          bool hit = true;
          for (std::size_t i = 0; i < g->bodies[k].size(); ++i) {
            Q a = qscore[k][i], d = g->bodies[k][i].exact(p) - a;
            if (d == 0) {
              if (a > 0) {
                hit = false;
                break;
              }
              continue;
            }
            Q t = -a / d;
            if (d > 0) {
              if (t < exit)
                exit = t;
            } else if (t > enter)
              enter = t;
            if (enter > exit) {
              hit = false;
              break;
            }
          }
          if (hit && enter < best) {
            selected = int(k);
            best = enter;
          }
        }
        bodies.mutable_data()[j] = selected;
        entries.mutable_data()[j] = selected < 0 ? NAN : best.get_d();
      }
    }
    return py::make_tuple(bodies, entries);
  }
  std::array<int, 2> cert(P lo, P hi, Stats &st, bool projected = false,
                          bool skip_mask = false) const {
    int m = skip_mask ? 0
                      : (projected ? reference_union(g->bodies, lo, hi)
                                   : united(g->bodies, lo, hi));
    if (m == 1)
      return {0, 1};
    if (projected) {
      int v = reference_union(shadow, lo, hi);
      return {v < 0 ? -1 : 1 - v, m};
    }
    bool all_miss = true;
    for (std::size_t k = 0; k < g->bodies.size(); ++k) {
      int v = clip(k, lo, hi, st);
      if (v == 0)
        continue;
      all_miss = false;
      bool corners = true;
      for (int j = 0; j < 8 && corners; ++j) {
        P x;
        for (int a = 0; a < 3; ++a)
          x[a] = (j & (1 << a)) ? hi[a] : lo[a];
        corners = clip(k, x, x, st) == 1;
      }
      if (corners)
        return {0, m};
    }
    return {all_miss ? 1 : -1, m};
  }
  py::tuple certificate(P lo, P hi, bool projected = false) {
    require_environment();
    Stats s;
    auto v = cert(lo, hi, s, projected);
    return py::make_tuple(v[0], v[1]);
  }
  py::tuple points(py::array_t<double, py::array::c_style> p,
                   bool projected = false) {
    require_environment();
    if (p.ndim() != 2 || p.shape(1) != 3)
      throw std::invalid_argument("[N,3] required");
    py::array_t<std::int8_t> y(p.shape(0)), m(p.shape(0));
    Stats s;
    {
      py::gil_scoped_release r;
      for (py::ssize_t i = 0; i < p.shape(0); ++i) {
        P x{p.data()[3 * i], p.data()[3 * i + 1], p.data()[3 * i + 2]};
        valid_point(x);
        m.mutable_data()[i] = projected ? reference_solid(x, s) : solid(x, s);
        int v = projected ? reference_union(shadow, x, x) : -1;
        y.mutable_data()[i] =
            projected ? (v < 0 ? projected_visible_exact(x, s) : 1 - v)
                      : visible(x, s);
      }
    }
    return py::make_tuple(y, m, report(s));
  }
  py::tuple raster(std::array<int, 3> origin, int side, int n, int leaf = 8,
                   bool projected = false, bool skip_mask = false) {
    require_environment();
    if (side < 1 || side > 64 || n < 1 || n > 1024 || (n & (n - 1)) || leaf < 1)
      throw std::invalid_argument("Invalid raster extent");
    for (int x : origin)
      if (x < 0 || x + side > n)
        throw std::invalid_argument("Invalid origin");
    py::array_t<std::int8_t> yy({side, side, side}), mm({side, side, side});
    auto *y = yy.mutable_data();
    auto *m = mm.mutable_data();
    Stats st;
    struct Node {
      std::array<int, 3> o, l;
    };
    std::vector<Node> pending{{origin, {side, side, side}}};
    {
      py::gil_scoped_release r;
      while (!pending.empty()) {
        auto node = pending.back();
        pending.pop_back();
        P lo, hi;
        for (int a = 0; a < 3; ++a) {
          lo[a] = (node.o[a] + .5) / n;
          hi[a] = (node.o[a] + node.l[a] - .5) / n;
        }
        auto c = cert(lo, hi, st, projected, skip_mask);
        ++st.nodes;
        bool fill = c[0] >= 0 && c[1] >= 0;
        if (fill)
          st.covered += node.l[0] * node.l[1] * node.l[2];
        if (fill || *std::max_element(node.l.begin(), node.l.end()) <= leaf) {
          for (int z = 0; z < node.l[2]; ++z)
            for (int v = 0; v < node.l[1]; ++v)
              for (int x = 0; x < node.l[0]; ++x) {
                P p{(node.o[0] + x + .5) / n, (node.o[1] + v + .5) / n,
                    (node.o[2] + z + .5) / n};
                std::size_t k = ((node.o[2] + z - origin[2]) * side +
                                 node.o[1] + v - origin[1]) *
                                    side +
                                node.o[0] + x - origin[0];
                int a = c[0], b = c[1];
                if (a < 0) {
                  ++st.rays;
                  a = projected ? reference_union(shadow, p, p) : -1;
                  a = a < 0 ? (projected ? projected_visible_exact(p, st)
                                         : visible(p, st))
                            : 1 - a;
                }
                if (b < 0) {
                  ++st.points;
                  b = projected ? reference_solid(p, st) : solid(p, st);
                }
                y[k] = a;
                m[k] = b;
              }
        } else
          for (int z = 1; z >= 0; --z)
            for (int v = 1; v >= 0; --v)
              for (int x = 1; x >= 0; --x) {
                int bits[] = {x, v, z};
                Node child = node;
                for (int a = 0; a < 3; ++a) {
                  int cut = node.l[a] / 2;
                  if (node.l[a] <= leaf) {
                    if (bits[a]) {
                      child.l[0] = 0;
                      break;
                    }
                  } else {
                    child.o[a] += bits[a] * cut;
                    child.l[a] = bits[a] ? node.l[a] - cut : cut;
                  }
                }
                if (child.l[0])
                  pending.push_back(child);
              }
      }
    }
    return py::make_tuple(yy, mm, report(st));
  }
};
PYBIND11_MODULE(_convex, m) {
  m.def("step", &step);
  m.def("interval", [](double a, double b, int op) {
    I v = op == 0   ? add(I(a), I(b))
          : op == 1 ? mul(I(a), I(b))
                    : divi(I(a), I(b));
    return py::make_tuple(v.lo, v.hi);
  });
  py::class_<Geometry, std::shared_ptr<Geometry>>(m, "Geometry",
                                                  py::module_local())
      .def(py::init<const Rows &>())
      .def("proof", &Geometry::proof)
      .def("contains", &Geometry::contains)
      .def("membership", &Geometry::membership)
      .def("box_class", &Geometry::box_class)
      .def("clearance_planes", &Geometry::clearance_planes);
  py::class_<Model>(m, "Model", py::module_local())
      .def(py::init<std::shared_ptr<Geometry>, P, const Rows &>())
      .def("first_hits", &Model::first_hits)
      .def("points", &Model::points, py::arg("points"),
           py::arg("projected") = false)
      .def("certificate", &Model::certificate, py::arg("lo"), py::arg("hi"),
           py::arg("projected") = false)
      .def("raster", &Model::raster, py::arg("origin"), py::arg("side"),
           py::arg("n"), py::arg("leaf") = 8, py::arg("projected") = false,
           py::arg("skip_mask") = false);
}
