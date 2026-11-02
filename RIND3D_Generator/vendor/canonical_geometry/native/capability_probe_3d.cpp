#include <CGAL/Exact_integer.h>
#include <CGAL/Extended_homogeneous.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Object.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Integer = CGAL::Exact_integer;
using Kernel = CGAL::Extended_homogeneous<Integer>;
using Point = Kernel::Point_3;
using Plane = Kernel::Plane_3;
using Nef = CGAL::Nef_polyhedron_3<Kernel>;

static Nef cube(int x0, int y0, int z0, int x1, int y1, int z1) {
  Nef value(Plane(-1, 0, 0, x0), Nef::INCLUDED);
  value *= Nef(Plane(1, 0, 0, -x1), Nef::INCLUDED);
  value *= Nef(Plane(0, -1, 0, y0), Nef::INCLUDED);
  value *= Nef(Plane(0, 1, 0, -y1), Nef::INCLUDED);
  value *= Nef(Plane(0, 0, -1, z0), Nef::INCLUDED);
  value *= Nef(Plane(0, 0, 1, -z1), Nef::INCLUDED);
  return value.regularization();
}

static bool member(const Nef &solid, const Point &point) {
  auto object = solid.locate(point);
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

int main() {
  try {
    Nef domain = cube(0, 0, 0, 4, 4, 4);
    Nef left = cube(0, 1, 1, 2, 3, 3);
    Nef right = cube(2, 1, 1, 4, 3, 3);
    Nef joined = (left + right).regularization();
    Nef touching = (left * right).regularization();
    if (!touching.is_empty())
      throw std::runtime_error("regularized touching intersection is not empty");
    Point center(2, 2, 2);
    Nef relative_interior = (joined + !domain).interior();
    if (!member(joined, center) || !member(relative_interior, center))
      throw std::runtime_error("shared interface failed relative interior check");
    if (!member((domain + !domain).interior(), Point(0, 0, 0)))
      throw std::runtime_error("full domain corner is not relative interior");

    Point endpoints[] = {Point(0, 2, 2), Point(4, 2, 2)};
    using Range = std::pair<Point *, Point *>;
    std::vector<Range> polyline{{endpoints, endpoints + 2}};
    Nef segment(polyline.begin(), polyline.end(), Nef::Polylines_tag());
    if ((joined * segment).is_empty())
      throw std::runtime_error("polyline segment missed closed solid");
    Nef outside = cube(-2, -2, -2, -1, -1, -1);
    if (!(outside * joined).is_empty())
      throw std::runtime_error("disjoint cube intersection was nonempty");
    if (!(joined - domain).is_empty())
      throw std::runtime_error("solid escaped the domain");
    std::cout << "CGAL Nef_3 regularized CSG: PASS\n"
              << "CGAL Nef_3 relative point classification: PASS\n"
              << "CGAL Nef_3 lower-dimensional segment intersection: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "RIND 3D capability probe failed: " << error.what() << '\n';
    return 1;
  }
}
