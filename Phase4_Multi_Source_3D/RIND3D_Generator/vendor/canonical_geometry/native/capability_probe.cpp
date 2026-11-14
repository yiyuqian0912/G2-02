#include <CGAL/config.h>
#include <CGAL/iterator.h>

#ifndef CGAL_USE_CORE
#error "The RIND conic backend requires CGAL Core exact algebraic numbers."
#endif

#include <CGAL/Arr_conic_traits_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/CORE_algebraic_number_traits.h>
#include <CGAL/Cartesian.h>
#include <CGAL/General_polygon_2.h>
#include <CGAL/Gps_traits_2.h>

#include <iostream>
#include <iterator>
#include <list>
#include <stdexcept>

using Number_traits = CGAL::CORE_algebraic_number_traits;
using Rational = Number_traits::Rational;
using Algebraic = Number_traits::Algebraic;
using Rational_kernel = CGAL::Cartesian<Rational>;
using Algebraic_kernel = CGAL::Cartesian<Algebraic>;
using Conic_traits =
    CGAL::Arr_conic_traits_2<Rational_kernel, Algebraic_kernel,
                             Number_traits>;
using Polygon = CGAL::General_polygon_2<Conic_traits>;
using Polygon_traits = CGAL::Gps_traits_2<Conic_traits, Polygon>;
using Polygon_with_holes = Polygon_traits::General_polygon_with_holes_2;
using Curve = Polygon_traits::Curve_2;
using X_monotone_curve = Polygon_traits::X_monotone_curve_2;
using Point = Polygon_traits::Point_2;

static void append_arc(Polygon &polygon, const Curve &arc) {
  Conic_traits traits;
  traits.make_x_monotone_2_object()(
      arc, CGAL::dispatch_or_drop_output<X_monotone_curve>(
               std::back_inserter(polygon)));
}

int main() {
  try {
    Conic_traits traits;
    auto make_curve = traits.construct_curve_2_object();

    // Rational conics with algebraic intersection points. This is the
    // CGAL conic traits adapter path required by the 2D backend.
    Polygon lens;
    append_arc(lens, make_curve(1, 0, 0, 0, 2, -4,
                                CGAL::COUNTERCLOCKWISE, Point(2, 0),
                                Point(-2, 0)));
    append_arc(lens, make_curve(1, 0, 0, 0, -2, -4,
                                CGAL::COUNTERCLOCKWISE, Point(-2, 0),
                                Point(2, 0)));

    Polygon ellipse;
    append_arc(ellipse, make_curve(-1, -9, 0, 0, 0, 9));

    std::list<Polygon_with_holes> overlap;
    CGAL::intersection(lens, ellipse, std::back_inserter(overlap));
    if (overlap.empty())
      throw std::runtime_error("conic intersection unexpectedly empty");

    std::list<Polygon_with_holes> removed;
    CGAL::difference(ellipse, ellipse, std::back_inserter(removed));
    if (!removed.empty())
      throw std::runtime_error("regularized self-difference is not empty");

    std::cout << "CGAL exact conic intersection: PASS\n"
              << "CGAL regularized conic difference: PASS\n"
              << "CGAL version: " << CGAL_VERSION_STR << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "RIND capability probe failed: " << error.what() << '\n';
    return 1;
  }
}

