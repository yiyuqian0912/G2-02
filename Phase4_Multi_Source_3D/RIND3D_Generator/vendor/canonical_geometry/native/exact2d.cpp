#include <CGAL/Arr_conic_traits_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/CORE_algebraic_number_traits.h>
#include <CGAL/Cartesian.h>
#include <CGAL/General_polygon_2.h>
#include <CGAL/General_polygon_set_2.h>
#include <CGAL/Gps_traits_2.h>
#include <CGAL/convex_hull_2.h>
#include <CGAL/iterator.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

using Nt = CGAL::CORE_algebraic_number_traits;
using Rat = Nt::Rational;
using Alg = Nt::Algebraic;
using Rat_kernel = CGAL::Cartesian<Rat>;
using Alg_kernel = CGAL::Cartesian<Alg>;
using Conic_traits = CGAL::Arr_conic_traits_2<Rat_kernel, Alg_kernel, Nt>;
using Polygon = CGAL::General_polygon_2<Conic_traits>;
using Traits = CGAL::Gps_traits_2<Conic_traits, Polygon>;
using Solid = CGAL::General_polygon_set_2<Traits>;
using Curve = Traits::Curve_2;
using X_curve = Traits::X_monotone_curve_2;
using Point = Traits::Point_2;
using Rat_point = Rat_kernel::Point_2;

static Rat exact(double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("nonfinite binary64 coordinate");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const bool negative = (bits >> 63) != 0;
  const std::uint64_t exp_bits = (bits >> 52) & 0x7ff;
  const std::uint64_t fraction = bits & ((std::uint64_t(1) << 52) - 1);
  if (exp_bits == 0 && fraction == 0)
    return Rat(0);
  std::uint64_t mantissa = fraction;
  int exponent = -1074;
  if (exp_bits != 0) {
    mantissa |= std::uint64_t(1) << 52;
    exponent = int(exp_bits) - 1023 - 52;
  }
  CORE::BigInt numerator(mantissa);
  CORE::BigInt denominator(1);
  if (exponent >= 0)
    numerator <<= exponent;
  else
    denominator <<= -exponent;
  if (negative)
    numerator = -numerator;
  return Rat(numerator, denominator);
}

static Point point(double x, double y) {
  return Point(Alg(exact(x)), Alg(exact(y)));
}

static void append(Polygon &polygon, const Curve &curve) {
  Conic_traits traits;
  traits.make_x_monotone_2_object()(
      curve,
      CGAL::dispatch_or_drop_output<X_curve>(std::back_inserter(polygon)));
}

static Polygon polygon_from_points(std::vector<Point> points) {
  if (points.size() < 3)
    throw std::invalid_argument("polygon needs at least three vertices");
  Polygon polygon;
  Conic_traits traits;
  auto make = traits.construct_curve_2_object();
  for (std::size_t i = 0; i < points.size(); ++i)
    append(polygon, make(points[i], points[(i + 1) % points.size()]));
  return polygon;
}

static Solid rectangle(double x0, double y0, double x1, double y1) {
  if (x0 >= x1 || y0 >= y1)
    throw std::invalid_argument("rectangle must have positive area");
  return Solid(polygon_from_points({point(x0, y0), point(x1, y0),
                                    point(x1, y1), point(x0, y1)}));
}

static bool valid_domain_point(double x, double y) {
  return std::isfinite(x) && std::isfinite(y) && x >= 0 && x <= 1 &&
         y >= 0 && y <= 1;
}

// Generator helper: CGAL's exact rational hull of the finally stored
// binary64 proposal coordinates. The output order is rotated to the smallest
// source index, making the resulting canonical ring reproducible.
static std::vector<std::size_t> convex_hull_indices(
    const std::vector<std::array<double, 2>> &coordinates) {
  std::vector<Rat_point> points;
  points.reserve(coordinates.size());
  for (const auto &coordinate : coordinates)
    points.emplace_back(exact(coordinate[0]), exact(coordinate[1]));
  std::vector<Rat_point> hull;
  CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(hull));
  if (hull.size() < 3)
    throw std::invalid_argument("convex hull is not full-dimensional");
  std::vector<std::size_t> indices;
  for (const auto &vertex : hull) {
    auto found = std::find(points.begin(), points.end(), vertex);
    if (found == points.end())
      throw std::logic_error("CGAL hull vertex did not match an input");
    indices.push_back(std::size_t(std::distance(points.begin(), found)));
  }
  auto first = std::min_element(indices.begin(), indices.end());
  std::rotate(indices.begin(), first, indices.end());
  return indices;
}

struct ProofPoint {
  Point exact_point;
  explicit ProofPoint(Point value) : exact_point(std::move(value)) {}
  double x_approx() const { return exact_point.x().doubleValue(); }
  double y_approx() const { return exact_point.y().doubleValue(); }
};

class ExactGeometry2D {
public:
  ExactGeometry2D() : domain_(rectangle(0, 0, 1, 1)) {}

  std::size_t add_box(const std::vector<double> &center,
                      const std::vector<double> &matrix) {
    check_affine(center, matrix);
    auto a = affine(center, matrix);
    std::vector<Point> vertices;
    for (auto local : std::array<std::array<int, 2>, 4>{{
             {{-1, -1}}, {{1, -1}}, {{1, 1}}, {{-1, 1}}}})
      vertices.push_back(world(a, Rat(local[0]), Rat(local[1])));
    if (a.det < 0)
      std::reverse(vertices.begin(), vertices.end());
    return add_solid(Solid(polygon_from_points(std::move(vertices))));
  }

  std::size_t add_ball(const std::vector<double> &center,
                       const std::vector<double> &matrix) {
    check_affine(center, matrix);
    auto a = affine(center, matrix);
    Rat i00 = a.m[3] / a.det;
    Rat i01 = -a.m[1] / a.det;
    Rat i10 = -a.m[2] / a.det;
    Rat i11 = a.m[0] / a.det;
    Rat q00 = i00 * i00 + i10 * i10;
    Rat q01 = i00 * i01 + i10 * i11;
    Rat q11 = i01 * i01 + i11 * i11;
    Rat cx = a.c[0];
    Rat cy = a.c[1];
    Conic_traits traits;
    auto make = traits.construct_curve_2_object();
    Polygon ellipse;
    // The negative quadratic coefficients yield a CCW outer contour with
    // CGAL's full-conic constructor, as in its conic traits example.
    append(ellipse, make(-q00, -q11, -2 * q01,
                         2 * (q00 * cx + q01 * cy),
                         2 * (q01 * cx + q11 * cy),
                         1 - q00 * cx * cx - 2 * q01 * cx * cy - q11 * cy * cy));
    return add_solid(Solid(ellipse));
  }

  std::size_t add_polygon(
      const std::vector<double> &center, const std::vector<double> &matrix,
      const std::vector<std::array<double, 2>> &vertices,
      const std::vector<std::uint64_t> &offsets,
      const std::vector<std::uint8_t> &roles,
      const std::vector<std::uint32_t> &owners) {
    check_affine(center, matrix);
    if (offsets.size() != roles.size() + 1 || roles.size() != owners.size() ||
        offsets.empty() || offsets.front() != 0 ||
        offsets.back() != vertices.size())
      throw std::invalid_argument("polygon ring offsets or roles are invalid");
    auto a = affine(center, matrix);
    std::vector<Polygon> rings;
    rings.reserve(roles.size());
    for (std::size_t r = 0; r < roles.size(); ++r) {
      std::vector<Point> points;
      for (std::size_t v = offsets[r]; v < offsets[r + 1]; ++v)
        points.push_back(world(a, exact(vertices[v][0]), exact(vertices[v][1])));
      if (a.det < 0)
        std::reverse(points.begin(), points.end());
      rings.push_back(polygon_from_points(std::move(points)));
    }
    Solid result;
    for (std::size_t r = 0; r < roles.size(); ++r) {
      if (roles[r] == 0)
        result.join(Solid(rings[r]));
      else if (roles[r] != 1 || owners[r] >= roles.size() || roles[owners[r]] != 0)
        throw std::invalid_argument("invalid polygon hole owner");
    }
    for (std::size_t r = 0; r < roles.size(); ++r)
      if (roles[r] == 1)
        result.difference(Solid(rings[r]));
    return add_solid(std::move(result));
  }

  std::size_t add_halfspaces(
      const std::vector<double> &center, const std::vector<double> &matrix,
      const std::vector<std::array<double, 2>> &normal,
      const std::vector<double> &offset) {
    check_affine(center, matrix);
    if (normal.size() != offset.size())
      throw std::invalid_argument("halfspace plane count mismatch");
    struct Plane { Rat x, y, b; };
    std::vector<Plane> planes;
    for (std::size_t i = 0; i < normal.size(); ++i) {
      Plane plane{exact(normal[i][0]), exact(normal[i][1]), exact(offset[i])};
      if (plane.x == 0 && plane.y == 0) {
        if (plane.b < 0)
          throw std::invalid_argument("empty halfspace intersection");
        continue;
      }
      planes.push_back(plane);
    }
    if (planes.empty())
      throw std::invalid_argument("unbounded halfspace intersection");
    for (const auto &plane : planes) {
      for (int sign : {-1, 1}) {
        Rat vx = sign * -plane.y;
        Rat vy = sign * plane.x;
        if (std::all_of(planes.begin(), planes.end(), [&](const Plane &other) {
              return other.x * vx + other.y * vy <= 0;
            }))
          throw std::invalid_argument("unbounded halfspace intersection");
      }
    }
    std::vector<Rat_point> feasible;
    for (std::size_t i = 0; i < planes.size(); ++i) {
      for (std::size_t j = i + 1; j < planes.size(); ++j) {
        Rat det = planes[i].x * planes[j].y - planes[j].x * planes[i].y;
        if (det == 0)
          continue;
        Rat x = (planes[i].b * planes[j].y - planes[j].b * planes[i].y) / det;
        Rat y = (planes[i].x * planes[j].b - planes[j].x * planes[i].b) / det;
        if (std::all_of(planes.begin(), planes.end(), [&](const Plane &plane) {
              return plane.x * x + plane.y * y <= plane.b;
            }))
          feasible.emplace_back(x, y);
      }
    }
    std::vector<Rat_point> hull;
    CGAL::convex_hull_2(feasible.begin(), feasible.end(), std::back_inserter(hull));
    if (hull.size() < 3)
      throw std::invalid_argument("halfspace intersection is not full-dimensional");
    auto a = affine(center, matrix);
    std::vector<Point> world_vertices;
    for (const auto &v : hull)
      world_vertices.push_back(world(a, v.x(), v.y()));
    if (a.det < 0)
      std::reverse(world_vertices.begin(), world_vertices.end());
    return add_solid(Solid(polygon_from_points(std::move(world_vertices))));
  }

  std::size_t add_csg(std::uint8_t opcode,
                      const std::vector<std::size_t> &children) {
    if ((opcode == 2 && children.size() != 2) ||
        ((opcode == 0 || opcode == 1) && children.size() < 2) || opcode > 2)
      throw std::invalid_argument("invalid CSG opcode or arity");
    for (auto child : children)
      if (child >= solids_.size())
        throw std::invalid_argument("CSG child must precede parent");
    Solid result(solids_[children.front()]);
    for (std::size_t i = 1; i < children.size(); ++i) {
      if (opcode == 0)
        result.join(solids_[children[i]]);
      else if (opcode == 1)
        result.intersection(solids_[children[i]]);
      else
        result.difference(solids_[children[i]]);
    }
    return add_solid(std::move(result));
  }

  void set_root(int index) {
    if (index < -1 || (index >= 0 && std::size_t(index) >= solids_.size()))
      throw std::invalid_argument("invalid geometry root index");
    result_.clear();
    if (index >= 0)
      result_.join(solids_[std::size_t(index)]);
    result_.intersection(domain_);
    Solid exterior(domain_);
    exterior.complement();
    relative_interior_.clear();
    relative_interior_.join(result_);
    relative_interior_.join(exterior);
    root_ready_ = true;
  }

  bool is_empty() const {
    require_root();
    return result_.is_empty();
  }

  bool is_full() const {
    require_root();
    Solid missing(domain_);
    missing.difference(result_);
    return missing.is_empty();
  }

  std::pair<std::size_t, std::size_t> arrangement_counts() const {
    require_root();
    const auto &arr = result_.arrangement();
    return {arr.number_of_vertices(), arr.number_of_halfedges()};
  }

  int classify(double x, double y) const {
    require_root();
    if (!valid_domain_point(x, y))
      throw std::invalid_argument("point outside query domain");
    Point p = point(x, y);
    auto side = result_.oriented_side(p);
    if (side == CGAL::ON_NEGATIVE_SIDE)
      return 0;
    if (relative_interior_.oriented_side(p) == CGAL::ON_POSITIVE_SIDE)
      return 1;
    return 2;
  }

  bool segment_hit(double ax, double ay, double bx, double by) const {
    require_root();
    if (!valid_domain_point(ax, ay) || !valid_domain_point(bx, by))
      throw std::invalid_argument("segment endpoint outside query domain");
    Point a = point(ax, ay);
    Point b = point(bx, by);
    if (is_member(a) || is_member(b))
      return true;
    if (ax == bx && ay == by)
      return false;
    Conic_traits traits;
    X_curve segment;
    auto make = traits.construct_curve_2_object();
    traits.make_x_monotone_2_object()(
        make(a, b), CGAL::dispatch_or_drop_output<X_curve>(&segment));
    for (auto edge = result_.arrangement().halfedges_begin();
         edge != result_.arrangement().halfedges_end(); ++edge) {
      if (traits.do_intersect_2_object()(segment, edge->curve(), true))
        return true;
    }
    return false;
  }

  // 0 = LE, 1 = GT. Both outcomes are proved from exact arrangement curves.
  std::pair<int, std::shared_ptr<ProofPoint>> clearance(
      double px, double py, double radius) const {
    require_root();
    if (!valid_domain_point(px, py) || !std::isfinite(radius) || radius < 0)
      throw std::invalid_argument("invalid clearance request");
    Point p = point(px, py);
    if (is_member(p))
      return {0, std::make_shared<ProofPoint>(p)};
    if (result_.is_empty() || radius == 0)
      return {1, nullptr};

    Rat x = exact(px);
    Rat y = exact(py);
    Rat r = exact(radius);
    Rat r2 = r * r;
    for (auto vertex = result_.arrangement().vertices_begin();
         vertex != result_.arrangement().vertices_end(); ++vertex) {
      Point q = vertex->point();
      if (is_member(q) && within_disk(q, p, r2))
        return {0, std::make_shared<ProofPoint>(q)};
    }

    Conic_traits traits;
    auto circle = traits.construct_curve_2_object()(
        Rat(-1), Rat(-1), Rat(0), 2 * x, 2 * y, r2 - x * x - y * y);
    std::vector<X_curve> arcs;
    traits.make_x_monotone_2_object()(
        circle, CGAL::dispatch_or_drop_output<X_curve>(std::back_inserter(arcs)));
    using Intersection = std::variant<std::pair<Point, unsigned int>, X_curve>;
    for (auto edge = result_.arrangement().halfedges_begin();
         edge != result_.arrangement().halfedges_end(); ++edge) {
      for (const auto &arc : arcs) {
        std::vector<Intersection> intersections;
        traits.intersect_2_object()(edge->curve(), arc,
                                    std::back_inserter(intersections));
        for (const auto &intersection : intersections) {
          Point q = std::holds_alternative<X_curve>(intersection)
                        ? std::get<X_curve>(intersection).source()
                        : std::get<std::pair<Point, unsigned int>>(intersection).first;
          if (is_member(q) && within_disk(q, p, r2))
            return {0, std::make_shared<ProofPoint>(q)};
        }
      }
    }
    return {1, nullptr};
  }

  bool verify_clearance_witness(double px, double py, double radius,
                                const ProofPoint &witness) const {
    if (!valid_domain_point(px, py) || !std::isfinite(radius) || radius < 0)
      return false;
    Rat r = exact(radius);
    return is_member(witness.exact_point) &&
           within_disk(witness.exact_point, point(px, py), r * r);
  }

  // 0 FREE, 1 SOLID, 2 MIXED. A degenerate box is a point or segment.
  int classify_box(double x0, double y0, double x1, double y1) const {
    require_root();
    if (!valid_domain_point(x0, y0) || !valid_domain_point(x1, y1) ||
        x0 > x1 || y0 > y1)
      throw std::invalid_argument("invalid box bounds");
    if (x0 == x1 && y0 == y1)
      return classify(x0, y0) == 0 ? 0 : 1;
    if (x0 == x1 || y0 == y1) {
      if (!segment_hit(x0, y0, x1, y1))
        return 0;
      return segment_contained(point(x0, y0), point(x1, y1)) ? 1 : 2;
    }
    Solid query = rectangle(x0, y0, x1, y1);
    Solid difference(query);
    difference.difference(result_);
    if (difference.is_empty())
      return 1;
    Point corners[] = {point(x0, y0), point(x1, y0),
                       point(x1, y1), point(x0, y1)};
    for (const auto &corner : corners)
      if (is_member(corner))
        return 2;
    for (auto vertex = result_.arrangement().vertices_begin();
         vertex != result_.arrangement().vertices_end(); ++vertex) {
      const Point &q = vertex->point();
      if (q.x() >= Alg(exact(x0)) && q.x() <= Alg(exact(x1)) &&
          q.y() >= Alg(exact(y0)) && q.y() <= Alg(exact(y1)) && is_member(q))
        return 2;
    }
    if (segment_hit(x0, y0, x1, y0) ||
        segment_hit(x1, y0, x1, y1) ||
        segment_hit(x1, y1, x0, y1) ||
        segment_hit(x0, y1, x0, y0))
      return 2;
    return 0;
  }

private:
  struct Affine {
    std::array<Rat, 2> c;
    std::array<Rat, 4> m;
    Rat det;
  };

  static Affine affine(const std::vector<double> &center,
                       const std::vector<double> &matrix) {
    Affine a{{exact(center[0]), exact(center[1])},
             {exact(matrix[0]), exact(matrix[1]), exact(matrix[2]),
              exact(matrix[3])}, Rat(0)};
    a.det = a.m[0] * a.m[3] - a.m[1] * a.m[2];
    if (a.det == 0)
      throw std::invalid_argument("singular affine matrix");
    return a;
  }

  static void check_affine(const std::vector<double> &center,
                           const std::vector<double> &matrix) {
    if (center.size() != 2 || matrix.size() != 4)
      throw std::invalid_argument("affine center/matrix shape mismatch");
  }

  static Point world(const Affine &a, const Rat &vx, const Rat &vy) {
    return Point(Alg(a.c[0] + a.m[0] * vx + a.m[1] * vy),
                 Alg(a.c[1] + a.m[2] * vx + a.m[3] * vy));
  }

  static bool within_disk(const Point &q, const Point &p, const Rat &r2) {
    Alg dx = q.x() - p.x();
    Alg dy = q.y() - p.y();
    return dx * dx + dy * dy <= Alg(r2);
  }

  bool is_member(const Point &q) const {
    return result_.oriented_side(q) != CGAL::ON_NEGATIVE_SIDE;
  }

  bool segment_contained(const Point &a, const Point &b) const {
    if (!is_member(a) || !is_member(b))
      return false;
    Conic_traits traits;
    X_curve segment;
    traits.make_x_monotone_2_object()(
        traits.construct_curve_2_object()(a, b),
        CGAL::dispatch_or_drop_output<X_curve>(&segment));
    using Intersection = std::variant<std::pair<Point, unsigned int>, X_curve>;
    std::vector<Point> events{a, b};
    for (auto edge = result_.arrangement().halfedges_begin();
         edge != result_.arrangement().halfedges_end(); ++edge) {
      std::vector<Intersection> intersections;
      traits.intersect_2_object()(segment, edge->curve(),
                                  std::back_inserter(intersections));
      for (const auto &intersection : intersections) {
        if (std::holds_alternative<X_curve>(intersection)) {
          const auto &arc = std::get<X_curve>(intersection);
          events.push_back(arc.source());
          events.push_back(arc.target());
        } else {
          events.push_back(std::get<std::pair<Point, unsigned int>>(intersection).first);
        }
      }
    }
    bool use_x = a.x() != b.x();
    std::sort(events.begin(), events.end(), [&](const Point &left, const Point &right) {
      return use_x ? left.x() < right.x() : left.y() < right.y();
    });
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (!is_member(events[i]))
        return false;
      if (i > 0 && (use_x ? events[i - 1].x() != events[i].x()
                          : events[i - 1].y() != events[i].y())) {
        Point midpoint((events[i - 1].x() + events[i].x()) / 2,
                       (events[i - 1].y() + events[i].y()) / 2);
        if (!is_member(midpoint))
          return false;
      }
    }
    return true;
  }

  std::size_t add_solid(Solid value) {
    solids_.push_back(std::move(value));
    return solids_.size() - 1;
  }

  void require_root() const {
    if (!root_ready_)
      throw std::logic_error("set_root must be called before queries");
  }

  std::vector<Solid> solids_;
  Solid domain_;
  Solid result_;
  Solid relative_interior_;
  bool root_ready_ = false;
};

PYBIND11_MODULE(_exact2d, module) {
  module.doc() = "Exact algebraic 2D structure geometry for RIND";
  py::class_<ProofPoint, std::shared_ptr<ProofPoint>>(module, "ProofPoint")
      .def_property_readonly("x_approx", &ProofPoint::x_approx)
      .def_property_readonly("y_approx", &ProofPoint::y_approx);
  py::class_<ExactGeometry2D>(module, "ExactGeometry2D")
      .def(py::init<>())
      .def("add_box", &ExactGeometry2D::add_box)
      .def("add_ball", &ExactGeometry2D::add_ball)
      .def("add_polygon", &ExactGeometry2D::add_polygon)
      .def("add_halfspaces", &ExactGeometry2D::add_halfspaces)
      .def("add_csg", &ExactGeometry2D::add_csg)
      .def("set_root", &ExactGeometry2D::set_root)
      .def("is_empty", &ExactGeometry2D::is_empty)
      .def("is_full", &ExactGeometry2D::is_full)
      .def("arrangement_counts", &ExactGeometry2D::arrangement_counts)
      .def("classify", &ExactGeometry2D::classify)
      .def("segment_hit", &ExactGeometry2D::segment_hit)
      .def("clearance", &ExactGeometry2D::clearance)
      .def("verify_clearance_witness", &ExactGeometry2D::verify_clearance_witness)
      .def("classify_box", &ExactGeometry2D::classify_box);
  module.attr("backend_version") = CGAL_VERSION_STR;
  module.def("convex_hull_indices", &convex_hull_indices);
}
