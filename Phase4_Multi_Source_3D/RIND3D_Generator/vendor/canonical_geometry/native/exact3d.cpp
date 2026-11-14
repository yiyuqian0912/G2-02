#include <CGAL/Extended_homogeneous.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Gmpz.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Object.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/convex_hull_3.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <utility>
#include <vector>

namespace py = pybind11;

using Integer = CGAL::Gmpz;
using Rational = CGAL::Gmpq;
using Kernel = CGAL::Extended_homogeneous<Integer>;
using Point = Kernel::Point_3;
using Plane = Kernel::Plane_3;
using Nef = CGAL::Nef_polyhedron_3<Kernel>;
using Mesh = CGAL::Surface_mesh<Point>;
using Polyhedron = CGAL::Polyhedron_3<Kernel>;
using Vector = std::array<Rational, 3>;

static Rational exact(double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("nonfinite binary64 coordinate");
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const bool negative = (bits >> 63) != 0;
  const unsigned exp_bits = unsigned((bits >> 52) & 0x7ff);
  const std::uint64_t fraction = bits & ((std::uint64_t(1) << 52) - 1);
  if (exp_bits == 0 && fraction == 0)
    return Rational(0);
  std::uint64_t mantissa = fraction;
  int exponent = -1074;
  if (exp_bits != 0) {
    mantissa |= std::uint64_t(1) << 52;
    exponent = int(exp_bits) - 1023 - 52;
  }
  Integer numerator(static_cast<unsigned long>(mantissa));
  Integer denominator(1);
  if (exponent >= 0)
    numerator <<= exponent;
  else
    denominator <<= -exponent;
  if (negative)
    numerator = -numerator;
  return Rational(numerator, denominator);
}

static Integer lcm(const Integer &a, const Integer &b) {
  return (a / CGAL::gcd(a, b)) * b;
}

static Point point(const Vector &v) {
  Integer denominator(1);
  for (const auto &value : v)
    denominator = lcm(denominator, value.denominator());
  return Point(v[0].numerator() * (denominator / v[0].denominator()),
               v[1].numerator() * (denominator / v[1].denominator()),
               v[2].numerator() * (denominator / v[2].denominator()),
               denominator);
}

static Point point(double x, double y, double z) {
  return point({exact(x), exact(y), exact(z)});
}

static Plane plane(const std::array<Rational, 4> &coefficient) {
  Integer denominator(1);
  for (const auto &value : coefficient)
    denominator = lcm(denominator, value.denominator());
  return Plane(coefficient[0].numerator() * (denominator / coefficient[0].denominator()),
               coefficient[1].numerator() * (denominator / coefficient[1].denominator()),
               coefficient[2].numerator() * (denominator / coefficient[2].denominator()),
               coefficient[3].numerator() * (denominator / coefficient[3].denominator()));
}

static Vector subtract(const Vector &a, const Vector &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static Vector add(const Vector &a, const Vector &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

static Vector scale(const Vector &a, const Rational &factor) {
  return {a[0] * factor, a[1] * factor, a[2] * factor};
}

static Integer finite_integer(const Kernel::RT &coordinate) {
  if (coordinate.degree() != 0)
    throw std::runtime_error("non-finite projective coordinate");
  return coordinate[0];
}

static Vector coordinates(const Point &p) {
  const Integer denominator = finite_integer(p.hw());
  if (denominator == 0)
    throw std::runtime_error("point at infinity has no Cartesian coordinates");
  return {Rational(finite_integer(p.hx()), denominator),
          Rational(finite_integer(p.hy()), denominator),
          Rational(finite_integer(p.hz()), denominator)};
}

static Vector cross(const Vector &a, const Vector &b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

static Rational dot(const Vector &a, const Vector &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

struct Affine {
  Vector center;
  std::array<Rational, 9> matrix;
  Rational determinant;
};

static Affine affine(const std::vector<double> &center,
                     const std::vector<double> &matrix) {
  if (center.size() != 3 || matrix.size() != 9)
    throw std::invalid_argument("3D affine center/matrix shape mismatch");
  Affine a;
  for (int i = 0; i < 3; ++i)
    a.center[i] = exact(center[i]);
  for (int i = 0; i < 9; ++i)
    a.matrix[i] = exact(matrix[i]);
  const auto &m = a.matrix;
  a.determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                  m[1] * (m[3] * m[8] - m[5] * m[6]) +
                  m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (a.determinant == 0)
    throw std::invalid_argument("singular affine transform");
  return a;
}

static Vector world(const Affine &a, const Vector &local) {
  Vector result = a.center;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      result[row] += a.matrix[3 * row + column] * local[column];
  return result;
}

static std::array<Rational, 9> inverse(const Affine &a) {
  const auto &m = a.matrix;
  return {
      (m[4] * m[8] - m[5] * m[7]) / a.determinant,
      (m[2] * m[7] - m[1] * m[8]) / a.determinant,
      (m[1] * m[5] - m[2] * m[4]) / a.determinant,
      (m[5] * m[6] - m[3] * m[8]) / a.determinant,
      (m[0] * m[8] - m[2] * m[6]) / a.determinant,
      (m[2] * m[3] - m[0] * m[5]) / a.determinant,
      (m[3] * m[7] - m[4] * m[6]) / a.determinant,
      (m[1] * m[6] - m[0] * m[7]) / a.determinant,
      (m[0] * m[4] - m[1] * m[3]) / a.determinant,
  };
}

static Nef halfspace(const std::array<Rational, 4> &coefficients) {
  return Nef(plane(coefficients), Nef::INCLUDED);
}

static Nef range_box(const Vector &lower, const Vector &upper,
                     bool regularize) {
  Nef result(Nef::COMPLETE);
  for (int axis = 0; axis < 3; ++axis) {
    std::array<Rational, 4> lower_plane{}, upper_plane{};
    lower_plane[axis] = -1;
    lower_plane[3] = lower[axis];
    upper_plane[axis] = 1;
    upper_plane[3] = -upper[axis];
    result *= halfspace(lower_plane);
    result *= halfspace(upper_plane);
  }
  return regularize ? result.regularization() : result;
}

static bool member(const Nef &solid, const Point &point_value) {
  auto object = solid.locate(point_value);
  Nef::Vertex_const_handle vertex;
  Nef::Halfedge_const_handle edge;
  Nef::Halffacet_const_handle facet;
  Nef::Volume_const_handle volume;
  if (CGAL::assign(vertex, object)) return vertex->mark();
  if (CGAL::assign(edge, object)) return edge->mark();
  if (CGAL::assign(facet, object)) return facet->mark();
  if (CGAL::assign(volume, object)) return volume->mark();
  throw std::runtime_error("Nef point location returned no cell");
}

static bool domain_point(const std::vector<double> &p) {
  if (p.size() != 3)
    return false;
  for (double value : p)
    if (!std::isfinite(value) || value < 0 || value > 1)
      return false;
  return true;
}

struct ProofPoint3D {
  Point exact_point;
  explicit ProofPoint3D(Point value) : exact_point(std::move(value)) {}
  std::array<double, 3> approximate() const {
    auto v = coordinates(exact_point);
    return {CGAL::to_double(v[0]), CGAL::to_double(v[1]), CGAL::to_double(v[2])};
  }
  std::array<std::string, 4> homogeneous() const {
    std::array<std::string, 4> result;
    const Integer entries[] = {finite_integer(exact_point.hx()),
                               finite_integer(exact_point.hy()),
                               finite_integer(exact_point.hz()),
                               finite_integer(exact_point.hw())};
    for (int i = 0; i < 4; ++i) {
      std::ostringstream text;
      text << entries[i];
      result[i] = text.str();
    }
    return result;
  }
};

static std::pair<Vector, Rational> closest_triangle(
    const Vector &p, const Vector &a, const Vector &b, const Vector &c) {
  std::vector<Vector> candidates{a, b, c};
  for (const auto &edge : std::array<std::pair<Vector, Vector>, 3>{{
           {a, b}, {b, c}, {c, a}}}) {
    Vector direction = subtract(edge.second, edge.first);
    Rational length2 = dot(direction, direction);
    if (length2 == 0)
      throw std::invalid_argument("zero-length mesh edge");
    Rational parameter = dot(subtract(p, edge.first), direction) / length2;
    if (parameter < 0) parameter = 0;
    if (parameter > 1) parameter = 1;
    candidates.push_back(add(edge.first, scale(direction, parameter)));
  }
  Vector u = subtract(b, a), v = subtract(c, a), w = subtract(p, a);
  Rational g00 = dot(u, u), g01 = dot(u, v), g11 = dot(v, v);
  Rational determinant = g00 * g11 - g01 * g01;
  if (determinant == 0)
    throw std::invalid_argument("zero-area mesh triangle");
  Rational projected_u = (dot(w, u) * g11 - dot(w, v) * g01) / determinant;
  Rational projected_v = (dot(w, v) * g00 - dot(w, u) * g01) / determinant;
  if (projected_u >= 0 && projected_v >= 0 && projected_u + projected_v <= 1)
    candidates.push_back(add(a, add(scale(u, projected_u), scale(v, projected_v))));
  Vector best = candidates.front();
  Rational best_distance = dot(subtract(best, p), subtract(best, p));
  for (const auto &candidate : candidates) {
    Rational distance = dot(subtract(candidate, p), subtract(candidate, p));
    if (distance < best_distance) {
      best = candidate;
      best_distance = distance;
    }
  }
  return {best, best_distance};
}

class ExactGeometry3D {
public:
  ExactGeometry3D()
      : domain_(range_box({Rational(0), Rational(0), Rational(0)},
                          {Rational(1), Rational(1), Rational(1)}, true)) {}

  std::size_t add_box(const std::vector<double> &center,
                      const std::vector<double> &matrix) {
    auto a = affine(center, matrix);
    Nef result(Nef::COMPLETE);
    for (int axis = 0; axis < 3; ++axis) {
      int other0 = (axis + 1) % 3;
      int other1 = (axis + 2) % 3;
      for (int sign : {-1, 1}) {
        Vector local{Rational(-1), Rational(-1), Rational(-1)};
        local[axis] = sign;
        Vector p0 = world(a, local);
        local[other0] = 1;
        Vector p1 = world(a, local);
        local[other0] = -1;
        local[other1] = 1;
        Vector p2 = world(a, local);
        Vector normal = cross(subtract(p1, p0), subtract(p2, p0));
        Rational offset = -dot(normal, p0);
        if (dot(normal, a.center) + offset > 0) {
          for (auto &value : normal) value = -value;
          offset = -offset;
        }
        result *= halfspace({normal[0], normal[1], normal[2], offset});
      }
    }
    solids_.push_back(result.regularization());
    return solids_.size() - 1;
  }

  std::size_t add_halfspaces(
      const std::vector<double> &center, const std::vector<double> &matrix,
      const std::vector<std::array<double, 3>> &normals,
      const std::vector<double> &offsets) {
    if (normals.size() != offsets.size())
      throw std::invalid_argument("halfspace normal/offset count mismatch");
    auto a = affine(center, matrix);
    auto inv = inverse(a);
    Nef result(Nef::COMPLETE);
    std::vector<Vector> active_normals;
    for (std::size_t i = 0; i < normals.size(); ++i) {
      Vector n{exact(normals[i][0]), exact(normals[i][1]), exact(normals[i][2])};
      Rational b = exact(offsets[i]);
      if (n[0] == 0 && n[1] == 0 && n[2] == 0) {
        if (b < 0)
          throw std::invalid_argument("empty zero-normal halfspace");
        continue;
      }
      active_normals.push_back(n);
      Vector world_normal{};
      for (int column = 0; column < 3; ++column)
        for (int row = 0; row < 3; ++row)
          world_normal[column] += n[row] * inv[3 * row + column];
      Rational d = -b - dot(world_normal, a.center);
      result *= halfspace({world_normal[0], world_normal[1], world_normal[2], d});
    }
    // A rank-three recession cone is nontrivial exactly when one of the
    // pairwise normal-cross-product rays (in either direction) satisfies all
    // homogeneous inequalities. Rank below three leaves a lineality vector.
    bool rank_three = false;
    for (std::size_t i = 0; i < active_normals.size(); ++i) {
      for (std::size_t j = i + 1; j < active_normals.size(); ++j) {
        const Vector ray = cross(active_normals[i], active_normals[j]);
        if (ray[0] == 0 && ray[1] == 0 && ray[2] == 0) continue;
        for (const auto &normal : active_normals)
          if (dot(normal, ray) != 0) rank_three = true;
        bool positive_recession = true, negative_recession = true;
        for (const auto &normal : active_normals) {
          const Rational projection = dot(normal, ray);
          if (projection > 0) positive_recession = false;
          if (projection < 0) negative_recession = false;
        }
        if (positive_recession || negative_recession)
          throw std::invalid_argument("unbounded halfspace intersection");
      }
    }
    if (!rank_three)
      throw std::invalid_argument("unbounded halfspace intersection with lineality");
    if (result.regularization().is_empty())
      throw std::invalid_argument("halfspace intersection lacks volume");
    solids_.push_back(result.regularization());
    return solids_.size() - 1;
  }

  std::size_t add_mesh(
      const std::vector<double> &center, const std::vector<double> &matrix,
      const std::vector<std::array<double, 3>> &vertices,
      const std::vector<std::array<std::uint32_t, 3>> &triangles,
      std::vector<std::uint64_t> shell_offsets,
      std::vector<std::uint8_t> shell_roles,
      std::vector<std::uint32_t> shell_owners) {
    auto a = affine(center, matrix);
    if (shell_offsets.empty()) {
      shell_offsets = {0, triangles.size()};
      shell_roles = {0};
      shell_owners = {std::numeric_limits<std::uint32_t>::max()};
    }
    const std::size_t shell_count = shell_roles.size();
    if (shell_count == 0 || shell_offsets.size() != shell_count + 1 ||
        shell_owners.size() != shell_count || shell_offsets.front() != 0 ||
        shell_offsets.back() != triangles.size())
      throw std::invalid_argument("invalid mesh shell partition");
    for (std::size_t s = 0; s < shell_count; ++s) {
      if (shell_offsets[s] >= shell_offsets[s + 1] || shell_roles[s] > 1)
        throw std::invalid_argument("invalid mesh shell role or range");
      if ((shell_roles[s] == 0 &&
           shell_owners[s] != std::numeric_limits<std::uint32_t>::max()) ||
          (shell_roles[s] == 1 &&
           (shell_owners[s] >= shell_count || shell_roles[shell_owners[s]] != 0)))
        throw std::invalid_argument("invalid mesh cavity owner");
    }
    std::vector<Vector> world_vertices;
    world_vertices.reserve(vertices.size());
    for (const auto &vertex : vertices) {
      Vector value = world(a, {exact(vertex[0]), exact(vertex[1]),
                               exact(vertex[2])});
      for (const auto &prior : world_vertices)
        if (value == prior)
          throw std::invalid_argument("duplicate mesh vertex position");
      world_vertices.push_back(std::move(value));
    }
    std::vector<Nef> shell_solids;
    shell_solids.reserve(shell_count);
    for (std::size_t s = 0; s < shell_count; ++s) {
      Mesh mesh;
      std::vector<Mesh::Vertex_index> indices(vertices.size(), Mesh::null_vertex());
      Rational volume_times_six(0);
      const bool cavity = shell_roles[s] == 1;
      const bool flip_for_world = (a.determinant < 0) != cavity;
      for (std::uint64_t k = shell_offsets[s]; k < shell_offsets[s + 1]; ++k) {
        const auto &triangle = triangles[k];
        if (triangle[0] >= indices.size() || triangle[1] >= indices.size() ||
            triangle[2] >= indices.size())
          throw std::invalid_argument("mesh index out of bounds");
        const auto &a0 = world_vertices[triangle[0]];
        const auto &b0 = world_vertices[triangle[1]];
        const auto &c0 = world_vertices[triangle[2]];
        if (triangle[0] == triangle[1] || triangle[1] == triangle[2] ||
            triangle[2] == triangle[0] ||
            cross(subtract(b0, a0), subtract(c0, a0)) == Vector{})
          throw std::invalid_argument("degenerate mesh triangle");
        volume_times_six += dot(a0, cross(b0, c0)) *
                            (a.determinant > 0 ? Rational(1) : Rational(-1));
        for (auto index : triangle)
          if (indices[index] == Mesh::null_vertex())
            indices[index] = mesh.add_vertex(point(world_vertices[index]));
        auto face = flip_for_world
                        ? mesh.add_face(indices[triangle[0]], indices[triangle[2]],
                                        indices[triangle[1]])
                        : mesh.add_face(indices[triangle[0]], indices[triangle[1]],
                                        indices[triangle[2]]);
        if (face == Mesh::null_face())
          throw std::invalid_argument("mesh face insertion failed");
      }
      if (!CGAL::is_closed(mesh) || !CGAL::is_valid_polygon_mesh(mesh))
        throw std::invalid_argument("mesh shell is not a valid closed surface");
      auto component_map = mesh.add_property_map<Mesh::Face_index, std::size_t>(
          "f:component", 0).first;
      if (CGAL::Polygon_mesh_processing::connected_components(mesh, component_map) != 1)
        throw std::invalid_argument("mesh shell must be connected");
      for (auto vertex : mesh.vertices())
        if (CGAL::Polygon_mesh_processing::is_non_manifold_vertex(vertex, mesh))
          throw std::invalid_argument("non-manifold mesh vertex");
      if ((cavity && volume_times_six >= 0) ||
          (!cavity && volume_times_six <= 0))
        throw std::invalid_argument("mesh shell orientation disagrees with role");
      if (CGAL::Polygon_mesh_processing::does_self_intersect(mesh))
        throw std::invalid_argument("self-intersecting mesh shell");
      shell_solids.emplace_back(Nef(mesh).regularization());
      if (shell_solids.back().is_empty())
        throw std::invalid_argument("mesh shell encloses no volume");
    }
    for (std::size_t i = 0; i < shell_count; ++i) {
      for (std::size_t j = i + 1; j < shell_count; ++j) {
        if (shell_roles[i] == 0 && shell_roles[j] == 0 &&
            !(shell_solids[i] * shell_solids[j]).is_empty())
          throw std::invalid_argument("outer mesh shells intersect");
        if (shell_roles[i] == 1 && shell_roles[j] == 1 &&
            !(shell_solids[i] * shell_solids[j]).is_empty())
          throw std::invalid_argument("cavity mesh shells intersect");
      }
      if (shell_roles[i] == 1 &&
          !(shell_solids[i] - shell_solids[shell_owners[i]].interior()).is_empty())
        throw std::invalid_argument("mesh cavity is not strictly inside owner");
    }
    Nef result(Nef::EMPTY);
    for (std::size_t s = 0; s < shell_count; ++s)
      if (shell_roles[s] == 0)
        result = (result + shell_solids[s]).regularization();
    for (std::size_t s = 0; s < shell_count; ++s)
      if (shell_roles[s] == 1)
        result = (result - shell_solids[s]).regularization();
    solids_.push_back(result.regularization());
    return solids_.size() - 1;
  }

  std::size_t add_csg(std::uint8_t opcode,
                      const std::vector<std::size_t> &children) {
    if ((opcode == 2 && children.size() != 2) ||
        ((opcode == 0 || opcode == 1) && children.size() < 2) || opcode > 2)
      throw std::invalid_argument("invalid CSG opcode or arity");
    for (auto child : children)
      if (child >= solids_.size())
        throw std::invalid_argument("CSG child must precede parent");
    Nef result(solids_[children.front()]);
    for (std::size_t i = 1; i < children.size(); ++i) {
      if (opcode == 0) result += solids_[children[i]];
      else if (opcode == 1) result *= solids_[children[i]];
      else result -= solids_[children[i]];
      result = result.regularization();
    }
    solids_.push_back(std::move(result));
    return solids_.size() - 1;
  }

  void set_root(int index) {
    if (index < -1 || (index >= 0 && std::size_t(index) >= solids_.size()))
      throw std::invalid_argument("invalid root reference");
    result_ = (index < 0 ? Nef(Nef::EMPTY) : solids_[std::size_t(index)]);
    result_ = (result_ * domain_).regularization();
    relative_interior_ = (result_ + !domain_).interior();
    root_ready_ = true;
  }

  bool is_empty() const { require_root(); return result_.is_empty(); }
  bool is_full() const {
    require_root();
    return (domain_ - result_).is_empty();
  }

  int classify(const std::vector<double> &p) const {
    require_root();
    if (!domain_point(p))
      throw std::invalid_argument("invalid point request");
    if (result_.is_empty()) return 0;
    Point q = point(p[0], p[1], p[2]);
    if (!member(result_, q)) return 0;
    return member(relative_interior_, q) ? 1 : 2;
  }

  bool segment_hit(const std::vector<double> &a,
                   const std::vector<double> &b) const {
    require_root();
    if (!domain_point(a) || !domain_point(b))
      throw std::invalid_argument("invalid segment request");
    if (result_.is_empty()) return false;
    Point pa = point(a[0], a[1], a[2]);
    Point pb = point(b[0], b[1], b[2]);
    if (pa == pb)
      return member(result_, pa);
    Point points[] = {pa, pb};
    using Range = std::pair<Point *, Point *>;
    std::vector<Range> polyline{{points, points + 2}};
    Nef segment(polyline.begin(), polyline.end(), Nef::Polylines_tag());
    return !(result_ * segment).is_empty();
  }

  int classify_box(const std::vector<double> &lower,
                   const std::vector<double> &upper) const {
    require_root();
    if (!domain_point(lower) || !domain_point(upper))
      throw std::invalid_argument("invalid query box");
    if (result_.is_empty()) return 0;
    Vector lo{}, hi{};
    for (int i = 0; i < 3; ++i) {
      if (lower[i] > upper[i])
        throw std::invalid_argument("reversed query bounds");
      lo[i] = exact(lower[i]);
      hi[i] = exact(upper[i]);
    }
    Nef query = range_box(lo, hi, false);
    if ((query * result_).is_empty()) return 0;
    if ((query - result_).is_empty()) return 1;
    return 2;
  }

  std::size_t facet_count() const {
    require_root();
    if (!result_.is_simple())
      throw std::runtime_error("final Nef polyhedron is not simple");
    Polyhedron surface;
    result_.convert_to_polyhedron(surface);
    return surface.size_of_facets();
  }

  std::pair<int, std::shared_ptr<ProofPoint3D>> clearance(
      const std::vector<double> &p, double radius) const {
    require_root();
    if (!domain_point(p) || !std::isfinite(radius) || radius < 0)
      throw std::invalid_argument("invalid clearance request");
    if (result_.is_empty())
      return {1, nullptr};
    Point input = point(p[0], p[1], p[2]);
    if (member(result_, input))
      return {0, std::make_shared<ProofPoint3D>(input)};
    if (radius == 0)
      return {1, nullptr};
    if (!result_.is_simple())
      throw std::runtime_error("final polyhedron is not simple");
    Polyhedron surface;
    result_.convert_to_polyhedron(surface);
    Vector query{exact(p[0]), exact(p[1]), exact(p[2])};
    Rational limit = exact(radius) * exact(radius);
    for (auto facet = surface.facets_begin(); facet != surface.facets_end(); ++facet) {
      auto edge = facet->facet_begin();
      if (edge->next()->next()->next() != edge)
        throw std::runtime_error("nontriangular final polyhedron facet");
      Vector a = coordinates(edge->vertex()->point());
      Vector b = coordinates(edge->next()->vertex()->point());
      Vector c = coordinates(edge->next()->next()->vertex()->point());
      auto [nearest, distance] = closest_triangle(query, a, b, c);
      if (distance <= limit) {
        Point witness = point(nearest);
        if (!member(result_, witness))
          throw std::runtime_error("nearest triangle point is not in final solid");
        return {0, std::make_shared<ProofPoint3D>(witness)};
      }
    }
    return {1, nullptr};
  }

  bool verify_clearance_witness(const std::vector<double> &p, double radius,
                                const ProofPoint3D &witness) const {
    require_root();
    if (!domain_point(p) || !std::isfinite(radius) || radius < 0)
      return false;
    Vector query{exact(p[0]), exact(p[1]), exact(p[2])};
    Vector q = coordinates(witness.exact_point);
    Rational r = exact(radius);
    return member(result_, witness.exact_point) &&
           dot(subtract(q, query), subtract(q, query)) <= r * r;
  }

  bool verify_homogeneous_witness(
      const std::vector<double> &p, double radius,
      const std::array<std::string, 4> &homogeneous) const {
    std::array<Integer, 4> integers;
    for (int i = 0; i < 4; ++i)
      integers[i] = Integer(homogeneous[i].c_str());
    if (integers[3] <= 0)
      return false;
    return verify_clearance_witness(
        p, radius,
        ProofPoint3D(Point(integers[0], integers[1], integers[2], integers[3])));
  }

private:
  void require_root() const {
    if (!root_ready_)
      throw std::logic_error("set_root must precede queries");
  }
  std::vector<Nef> solids_;
  Nef domain_;
  Nef result_;
  Nef relative_interior_;
  bool root_ready_ = false;
};

static std::pair<std::vector<std::array<double, 3>>,
                 std::vector<std::array<std::uint32_t, 3>>>
convex_hull_mesh(const std::vector<std::array<double, 3>> &samples) {
  std::vector<Point> unique_points;
  std::vector<std::size_t> original_index;
  unique_points.reserve(samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    Point candidate = point(samples[i][0], samples[i][1], samples[i][2]);
    if (std::find(unique_points.begin(), unique_points.end(), candidate) ==
        unique_points.end()) {
      unique_points.push_back(candidate);
      original_index.push_back(i);
    }
  }
  if (unique_points.size() < 4)
    throw std::invalid_argument("convex hull requires four distinct points");
  Polyhedron hull;
  CGAL::convex_hull_3(unique_points.begin(), unique_points.end(), hull);
  if (hull.size_of_vertices() < 4 || hull.size_of_facets() < 4)
    throw std::invalid_argument("convex hull has no 3D interior");
  std::vector<std::array<std::uint32_t, 3>> raw_triangles;
  for (auto facet = hull.facets_begin(); facet != hull.facets_end(); ++facet) {
    std::vector<std::uint32_t> loop;
    auto first = facet->facet_begin();
    auto edge = first;
    do {
      auto found = std::find(unique_points.begin(), unique_points.end(),
                             edge->vertex()->point());
      if (found == unique_points.end())
        throw std::runtime_error("hull vertex not found in input");
      loop.push_back(static_cast<std::uint32_t>(original_index[
          std::size_t(found - unique_points.begin())]));
      ++edge;
    } while (edge != first);
    bool changed = true;
    while (changed && loop.size() > 3) {
      changed = false;
      for (std::size_t i = 0; i < loop.size(); ++i) {
        const auto &a = unique_points[std::size_t(
            std::find(original_index.begin(), original_index.end(),
                      loop[(i + loop.size() - 1) % loop.size()]) - original_index.begin())];
        const auto &b = unique_points[std::size_t(
            std::find(original_index.begin(), original_index.end(),
                      loop[i]) - original_index.begin())];
        const auto &c = unique_points[std::size_t(
            std::find(original_index.begin(), original_index.end(),
                      loop[(i + 1) % loop.size()]) - original_index.begin())];
        if (cross(subtract(coordinates(b), coordinates(a)),
                  subtract(coordinates(c), coordinates(b))) == Vector{}) {
          loop.erase(loop.begin() + i);
          changed = true;
          break;
        }
      }
    }
    if (loop.size() < 3)
      throw std::runtime_error("degenerate convex hull facet");
    std::rotate(loop.begin(), std::min_element(loop.begin(), loop.end()), loop.end());
    for (std::size_t i = 1; i + 1 < loop.size(); ++i) {
      std::array<std::uint32_t, 3> triangle{loop[0], loop[i], loop[i + 1]};
      const auto smallest = std::min_element(triangle.begin(), triangle.end());
      std::rotate(triangle.begin(), smallest, triangle.end());
      raw_triangles.push_back(triangle);
    }
  }
  std::sort(raw_triangles.begin(), raw_triangles.end());
  raw_triangles.erase(std::unique(raw_triangles.begin(), raw_triangles.end()),
                      raw_triangles.end());
  std::vector<std::uint32_t> used;
  for (const auto &triangle : raw_triangles)
    used.insert(used.end(), triangle.begin(), triangle.end());
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  std::vector<std::array<double, 3>> vertices;
  vertices.reserve(used.size());
  for (auto index : used)
    vertices.push_back(samples[index]);
  std::vector<std::array<std::uint32_t, 3>> triangles;
  triangles.reserve(raw_triangles.size());
  for (const auto &triangle : raw_triangles) {
    std::array<std::uint32_t, 3> remapped{};
    for (int i = 0; i < 3; ++i)
      remapped[i] = static_cast<std::uint32_t>(
          std::lower_bound(used.begin(), used.end(), triangle[i]) - used.begin());
    triangles.push_back(remapped);
  }
  return {vertices, triangles};
}

PYBIND11_MODULE(_exact3d, module) {
  module.doc() = "Exact 3D polyhedral structure geometry for RIND";
  py::class_<ProofPoint3D, std::shared_ptr<ProofPoint3D>>(module, "ProofPoint3D")
      .def_property_readonly("approximate", &ProofPoint3D::approximate)
      .def_property_readonly("homogeneous", &ProofPoint3D::homogeneous);
  py::class_<ExactGeometry3D>(module, "ExactGeometry3D")
      .def(py::init<>())
      .def("add_box", &ExactGeometry3D::add_box)
      .def("add_halfspaces", &ExactGeometry3D::add_halfspaces)
      .def("add_mesh", &ExactGeometry3D::add_mesh,
           py::arg("center"), py::arg("matrix"), py::arg("vertices"),
           py::arg("triangles"),
           py::arg("shell_offsets") = std::vector<std::uint64_t>{},
           py::arg("shell_roles") = std::vector<std::uint8_t>{},
           py::arg("shell_owners") = std::vector<std::uint32_t>{})
      .def("add_csg", &ExactGeometry3D::add_csg)
      .def("set_root", &ExactGeometry3D::set_root)
      .def("is_empty", &ExactGeometry3D::is_empty)
      .def("is_full", &ExactGeometry3D::is_full)
      .def("classify", &ExactGeometry3D::classify)
      .def("segment_hit", &ExactGeometry3D::segment_hit)
      .def("classify_box", &ExactGeometry3D::classify_box)
      .def("facet_count", &ExactGeometry3D::facet_count)
      .def("clearance", &ExactGeometry3D::clearance)
      .def("verify_clearance_witness", &ExactGeometry3D::verify_clearance_witness)
      .def("verify_homogeneous_witness", &ExactGeometry3D::verify_homogeneous_witness);
  module.attr("backend_version") = CGAL_VERSION_STR;
  module.def("convex_hull_mesh", &convex_hull_mesh);
}
