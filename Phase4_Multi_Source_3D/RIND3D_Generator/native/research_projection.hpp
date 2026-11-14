// Independent rational shadow projection. No parametric segment clipping here.
// Inputs are exact rational halfspaces; UNKNOWN interval signs use GMP.
#include <cfenv>
#include <functional>
#include <xmmintrin.h>
namespace projection_research {
struct Interval {
  double lo, hi;
};
double down(double x) { return std::nextafter(x, -INFINITY); }
double up(double x) { return std::nextafter(x, INFINITY); }
Interval enclose(const Q &q) {
  double x = q.get_d(), lo = x, hi = x;
  if (!std::isfinite(x))
    throw std::invalid_argument("Nonfinite coefficient");
  while (Q(lo) > q)
    lo = down(lo);
  while (Q(hi) < q)
    hi = up(hi);
  return {lo, hi};
}
void environment() {
  if (std::fegetround() != FE_TONEAREST ||
      (_mm_getcsr() & ((1u << 15) | (1u << 6))))
    throw std::runtime_error("Unsupported rounding or FTZ/DAZ environment");
}
struct Work {
  std::uint64_t nodes = 0, planes = 0, fast = 0, tight = 0, exact = 0,
                exact_planes = 0, covered = 0, rays = 0, points = 0,
                max_depth = 0;
};
py::dict report(const Work &w) {
  py::dict d;
  d["region_nodes"] = w.nodes;
  d["interval_plane_evaluations"] = w.planes;
  d["analytic_error_enclosures"] = w.fast;
  d["tight_interval_enclosures"] = w.tight;
  d["gmp_point_queries"] = w.exact;
  d["gmp_plane_evaluations"] = w.exact_planes;
  d["certificate_covered_voxels"] = w.covered;
  d["point_rays"] = w.rays;
  d["solid_points"] = w.points;
  d["max_subdivision_depth"] = w.max_depth;
  return d;
}
struct Plane {
  std::array<Q, 4> q;
  std::array<Interval, 4> v;
  std::array<double, 4> approximate;
  std::array<int, 3> sign;
  bool normalized = true;
  explicit Plane(const std::vector<std::string> &row) {
    if (row.size() != 4)
      throw std::invalid_argument("Four exact coefficients required");
    for (int k = 0; k < 4; ++k) {
      q[k] = Q(row[k]);
      q[k].canonicalize();
      v[k] = enclose(q[k]);
      approximate[k] = q[k].get_d();
      normalized &= abs(q[k]) <= 1;
      if (k < 3)
        sign[k] = sgn(q[k]);
    }
  }
  Interval range(const P &lo, const P &hi, Work &w) const {
    ++w.planes;
    if (normalized) {
      // PROVEN absolute forward-error enclosure, not a geometric tolerance.
      // |q_k|<=1, xyz in [0,1], binary64 RNE, no FTZ/DAZ or contraction:
      // coefficient truncation <=4*2^-52; 3 products <=3*2^-53;
      // 3 sums and final enclosure expansion <=4*2^-50 (all abs<8).
      // Their sum < 2^-46, including subnormal rounding. Thus even the RN
      // expansion by 2^-46 encloses the exact rational extrema. Ambiguous signs
      // still use the original tight nextafter bounds, then GMP at points.
      double l = approximate[3], h = approximate[3];
      for (int k = 2; k >= 0; --k)
        if (sign[k]) {
          l += approximate[k] * (sign[k] > 0 ? lo[k] : hi[k]);
          h += approximate[k] * (sign[k] > 0 ? hi[k] : lo[k]);
        }
      Interval coarse{l - 0x1p-46, h + 0x1p-46};
      if (coarse.lo > 0 || coarse.hi < 0) {
        ++w.fast;
        return coarse;
      }
    }
    ++w.tight;
    double a = v[3].lo, b = v[3].hi;
    // All canonical query coordinates are nonnegative. Known coefficient sign
    // reduces four generic products to the two actual extremal products.
    for (int k = 2; k >= 0; --k)
      if (sign[k]) {
        double l = v[k].lo * (sign[k] > 0 ? lo[k] : hi[k]);
        double h = v[k].hi * (sign[k] > 0 ? hi[k] : lo[k]);
        a = down(a + down(l));
        b = up(b + up(h));
        if (!std::isfinite(a) || !std::isfinite(b))
          return {-INFINITY, INFINITY};
      }
    return {a, b};
  }
  Q exact(const VQ &p, Work &w) const {
    ++w.exact_planes;
    Q r = q[3];
    for (int k = 2; k >= 0; --k)
      r += q[k] * p[k];
    return r;
  }
};
using Poly = std::vector<Plane>;
std::vector<Poly> parse(const Rows &rows) {
  std::vector<Poly> out;
  for (const auto &body : rows) {
    if (body.empty())
      throw std::invalid_argument("Empty projection body");
    out.emplace_back();
    for (const auto &r : body)
      out.back().emplace_back(r);
  }
  return out;
}
void box(const P &lo, const P &hi) {
  for (int k = 0; k < 3; ++k)
    if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || lo[k] < 0 ||
        hi[k] > 1 || lo[k] > hi[k])
      throw std::invalid_argument("Projection query outside closed world");
}
int classify(const std::vector<Poly> &bodies, const P &lo, const P &hi,
             Work &w) {
  bool unknown = false;
  for (const auto &body : bodies) {
    bool inside = true, outside = false;
    for (const auto &f : body) {
      auto v = f.range(lo, hi, w);
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
bool member(const std::vector<Poly> &bodies, const P &p, Work &w) {
  ++w.exact;
  VQ x{Q(p[0]), Q(p[1]), Q(p[2])};
  for (const auto &body : bodies) {
    bool in = true;
    for (const auto &f : body)
      if (f.exact(x, w) > 0) {
        in = false;
        break;
      }
    if (in)
      return true;
  }
  return false;
}
int point(const std::vector<Poly> &bodies, const P &p, Work &w) {
  int c = classify(bodies, p, p, w);
  return c < 0 ? int(member(bodies, p, w)) : c;
}
struct Model {
  std::vector<Poly> solids, shadows;
  Model(const Rows &a, const Rows &b) : solids(parse(a)), shadows(parse(b)) {
    if (solids.empty() || solids.size() != shadows.size())
      throw std::invalid_argument(
          "Paired nonempty geometry and shadows required");
  }
  std::array<int, 2> cert(P lo, P hi, Work &w,
                          std::array<int, 2> known = {-1, -1}) const {
    int m = known[1] < 0 ? classify(solids, lo, hi, w) : known[1];
    if (m == 1)
      return {0, 1};
    int y = known[0];
    if (y < 0) {
      int s = classify(shadows, lo, hi, w);
      y = s < 0 ? -1 : 1 - s;
    }
    return {y, m};
  }
  py::tuple certificate(P lo, P hi) const {
    environment();
    box(lo, hi);
    Work w;
    auto c = cert(lo, hi, w);
    return py::make_tuple(c[0], c[1]);
  }
  py::tuple points(py::array_t<double, py::array::c_style> p) const {
    environment();
    if (p.ndim() != 2 || p.shape(1) != 3)
      throw std::invalid_argument("[N,3] required");
    py::array_t<std::int8_t> y(p.shape(0)), m(p.shape(0));
    Work w;
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < p.shape(0); ++i) {
        P x{p.data()[3 * i], p.data()[3 * i + 1], p.data()[3 * i + 2]};
        box(x, x);
        int a = point(solids, x, w);
        int b = a ? 0 : 1 - point(shadows, x, w);
        m.mutable_data()[i] = a;
        y.mutable_data()[i] = b;
      }
    }
    return py::make_tuple(y, m, report(w));
  }
  using Bits = std::array<std::string, 2>;
  Bits raster(std::array<int, 3> origin, int side, int n, int leaf,
              Work &w) const {
    if (side < 1 || side > 64 || n < 1 || n > 1024 || (n & (n - 1)) ||
        leaf < 1 || leaf > 64)
      throw std::invalid_argument("Invalid raster extent");
    for (int v : origin)
      if (v < 0 || v + side > n)
        throw std::invalid_argument("Invalid raster origin");
    Bits bits{std::string(32768, 0), std::string(32768, 0)};
    struct Node {
      std::array<int, 3> o, l;
      int depth;
      std::array<int, 2> known;
    };
    std::vector<Node> stack{{origin, {side, side, side}, 0, {-1, -1}}};
    while (!stack.empty()) {
      Node t = stack.back();
      stack.pop_back();
      P lo, hi;
      for (int k = 0; k < 3; ++k) {
        lo[k] = (t.o[k] + .5) / n;
        hi[k] = (t.o[k] + t.l[k] - .5) / n;
      }
      ++w.nodes;
      w.max_depth = std::max(w.max_depth, std::uint64_t(t.depth));
      auto c = cert(lo, hi, w, t.known);
      bool fill = c[0] >= 0 && c[1] >= 0;
      if (fill)
        w.covered += t.l[0] * t.l[1] * t.l[2];
      if (fill || *std::max_element(t.l.begin(), t.l.end()) <= leaf) {
        for (int z = 0; z < t.l[2]; ++z)
          for (int y = 0; y < t.l[1]; ++y)
            for (int x = 0; x < t.l[0]; ++x) {
              P p{(t.o[0] + x + .5) / n, (t.o[1] + y + .5) / n,
                  (t.o[2] + z + .5) / n};
              int m = c[1], a = c[0];
              if (m < 0) {
                ++w.points;
                m = point(solids, p, w);
              }
              if (m)
                a = 0;
              else if (a < 0) {
                ++w.rays;
                a = 1 - point(shadows, p, w);
              }
              std::size_t i =
                  ((t.o[2] + z - origin[2]) * 64 + t.o[1] + y - origin[1]) *
                      64 +
                  t.o[0] + x - origin[0];
              if (a)
                bits[0][i / 8] |= char(1u << (i % 8));
              if (m)
                bits[1][i / 8] |= char(1u << (i % 8));
            }
      } else
        for (int mask = 7; mask >= 0; --mask) {
          Node child = t;
          child.depth++;
          child.known = c;
          bool valid = true;
          for (int k = 0; k < 3; ++k) {
            int half = t.l[k] / 2;
            if (t.l[k] <= leaf) {
              if (mask & (1 << k)) {
                valid = false;
                break;
              }
            } else {
              child.o[k] += (mask & (1 << k)) ? half : 0;
              child.l[k] = (mask & (1 << k)) ? t.l[k] - half : half;
            }
          }
          if (valid)
            stack.push_back(child);
        }
    }
    return bits;
  }
  py::tuple batch(const std::vector<std::array<int, 3>> &origins, int side,
                  int n, int leaf) const {
    environment();
    Work w;
    std::vector<Bits> fields;
    {
      py::gil_scoped_release release;
      for (auto &o : origins)
        fields.push_back(raster(o, side, n, leaf, w));
    }
    py::list output;
    for (auto &v : fields)
      output.append(py::make_tuple(py::bytes(v[0]), py::bytes(v[1])));
    return py::make_tuple(output, report(w));
  }
};
void bind(py::module_ &m) {
  m.def("projection_interval",
        [](const std::vector<std::string> &row, P lo, P hi) {
          environment();
          box(lo, hi);
          Work w;
          auto r = Plane(row).range(lo, hi, w);
          return py::make_tuple(r.lo, r.hi);
        });
  py::class_<Model>(m, "Projection")
      .def(py::init<const Rows &, const Rows &>())
      .def("points", &Model::points)
      .def("certificate", &Model::certificate)
      .def("batch", &Model::batch, py::arg("origins"), py::arg("side"),
           py::arg("resolution"), py::arg("leaf_side") = 4);
}
} // namespace projection_research
