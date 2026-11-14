// Exact first-hit and local-only occlusion analysis; NOT a physical verifier.
namespace nonlocal_research {
using Plane = std::array<Q, 4>;
using Poly = std::vector<Plane>;
Q eval(const Plane &f, const VQ &p) {
  Q v = f[3];
  for (int a = 0; a < 3; ++a)
    v += f[a] * p[a];
  return v;
}
bool clip(const Poly &body, const VQ &p, const VQ &x, Q &lo, Q &hi) {
  lo = 0;
  hi = 1;
  for (const auto &f : body) {
    Q a = eval(f, p), d = eval(f, x) - a;
    if (d == 0) {
      if (a > 0)
        return false;
      continue;
    }
    Q t = -a / d;
    if (d > 0) {
      if (t < hi)
        hi = t;
    } else if (t > lo)
      lo = t;
    if (lo > hi)
      return false;
  }
  return true;
}
struct Model {
  std::vector<Poly> bodies;
  VQ source;
  Model(const Rows &rows, P p) {
    for (int a = 0; a < 3; ++a) {
      if (!std::isfinite(p[a]) || p[a] <= 0 || p[a] >= 1)
        throw std::invalid_argument("Interior source required");
      source[a] = Q(p[a]);
    }
    for (auto &b : rows) {
      bodies.emplace_back();
      bool outside = false;
      for (auto &r : b) {
        if (r.size() != 4)
          throw std::invalid_argument("Four coefficients required");
        Plane f;
        for (int k = 0; k < 4; ++k) {
          f[k] = Q(r[k]);
          f[k].canonicalize();
        }
        outside |= eval(f, source) > 0;
        bodies.back().push_back(f);
      }
      if (!outside)
        throw std::invalid_argument("Source in closed geometry");
    }
  }
  py::list analyze(py::array_t<double, py::array::c_style> points,
                   std::array<int, 3> origin, int side, int n) {
    if (points.ndim() != 2 || points.shape(1) != 3 || side < 1 || n < 1)
      throw std::invalid_argument("Invalid analysis window");
    VQ lower, upper;
    for (int a = 0; a < 3; ++a) {
      if (origin[a] < 0 || origin[a] + side > n)
        throw std::invalid_argument("Window bounds");
      lower[a] = Q(origin[a], n);
      upper[a] = Q(origin[a] + side, n);
      lower[a].canonicalize();
      upper[a].canonicalize();
    }
    struct Result {
      std::string first, local;
      Q t;
      VQ hit;
      std::vector<int> ids;
    };
    std::vector<Result> results;
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < points.shape(0); ++i) {
        VQ x;
        for (int a = 0; a < 3; ++a) {
          double d = points.data()[3 * i + a];
          if (!std::isfinite(d) || d < 0 || d > 1)
            throw std::invalid_argument("Receiver bounds");
          x[a] = Q(d);
          if (x[a] < lower[a] || x[a] >= upper[a])
            throw std::invalid_argument(
                "Receiver outside half-open observation");
        }
        for (const auto &b : bodies) {
          bool inside = true;
          for (const auto &f : b)
            if (eval(f, x) > 0) {
              inside = false;
              break;
            }
          if (inside)
            throw std::invalid_argument("Nonlocal samples must be FREE");
        }
        // Segment interval inside the CLOSED observation; boundaries reported
        // apart.
        Q wl = 0, wh = 1;
        for (int a = 0; a < 3; ++a) {
          Q d = x[a] - source[a];
          if (d == 0) {
            if (source[a] < lower[a] || source[a] > upper[a])
              throw std::runtime_error("Receiver/segment inconsistency");
          } else {
            Q a0 = (lower[a] - source[a]) / d, a1 = (upper[a] - source[a]) / d;
            if (a0 > a1)
              std::swap(a0, a1);
            if (a0 > wl)
              wl = a0;
            if (a1 < wh)
              wh = a1;
          }
        }
        Result r{"LIT", "none", Q(2), {}, {}};
        bool local = false, interior = false;
        for (std::size_t k = 0; k < bodies.size(); ++k) {
          Q lo, hi;
          if (!clip(bodies[k], source, x, lo, hi))
            continue;
          if (lo < r.t) {
            r.t = lo;
            r.ids = {int(k)};
          } else if (lo == r.t)
            r.ids.push_back(int(k));
          Q a = std::max(lo, wl), b = std::min(hi, wh);
          if (a <= b) {
            local = true;
            Q mid = (a + b) / 2;
            bool in = true;
            for (int j = 0; j < 3; ++j) {
              Q v = source[j] + mid * (x[j] - source[j]);
              in &= v > lower[j] && v < upper[j];
            }
            interior |= in;
          }
        }
        if (!r.ids.empty()) {
          bool in = true, boundary = false, out = false;
          for (int a = 0; a < 3; ++a) {
            r.hit[a] = source[a] + r.t * (x[a] - source[a]);
            in &= r.hit[a] > lower[a] && r.hit[a] < upper[a];
            out |= r.hit[a] < lower[a] || r.hit[a] > upper[a];
            boundary |= r.hit[a] == lower[a] || r.hit[a] == upper[a];
          }
          r.first = in         ? "INSIDE"
                    : out      ? "OUTSIDE"
                    : boundary ? "BOUNDARY"
                               : "UNRESOLVED";
          r.local = interior ? "inside" : local ? "boundary_only" : "none";
        }
        results.push_back(r);
      }
    }
    py::list out;
    for (auto &r : results) {
      py::dict d;
      d["first_location"] = r.first;
      d["local_blocking"] = r.local;
      d["external_necessary"] = r.first == "OUTSIDE" && r.local == "none";
      d["first_components"] = r.ids;
      if (!r.ids.empty()) {
        d["t_exact"] = r.t.get_str();
        std::vector<std::string> h;
        for (auto &q : r.hit)
          h.push_back(q.get_str());
        d["hit_exact"] = h;
        d["t_float"] = r.t.get_d();
      }
      out.append(d);
    }
    return out;
  }
};
void bind(py::module_ &m) {
  py::class_<Model>(m, "Nonlocal")
      .def(py::init<const Rows &, P>())
      .def("analyze", &Model::analyze);
}
} // namespace nonlocal_research
