#include "regions.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace b2;
using namespace b2r;
namespace {
py::dict output(Result &r, int n) {
  auto &f = r.field;
  py::dict out;
  for (auto pair :
       {std::pair{"response", &f.response}, std::pair{"solid", &f.solid},
        std::pair{"valid", &f.valid}}) {
    py::array_t<std::uint8_t> a({n, n, n});
    std::copy(pair.second->begin(), pair.second->end(), a.mutable_data());
    out[pair.first] = a;
  }
  out["complete"] = f.complete;
  out["reason"] = f.reason;
  out["wall_seconds"] = f.seconds;
  out["work"] = f.stats.n;
  out["native_setup_seconds"] = r.setup_seconds;
  out["cover_seconds"] = r.cover_seconds;
  out["region_seconds"] = r.region_seconds;
  out["fallback_seconds"] = r.fallback_seconds;
  out["assembly_seconds"] = r.assembly_seconds;
  if (!r.evaluation_path.empty()) {
    py::array_t<std::uint8_t> path({n, n, n});
    std::copy(r.evaluation_path.begin(), r.evaluation_path.end(),
              path.mutable_data());
    out["evaluation_path"] = path;
    for (auto pair : {std::pair{"ray_nanoseconds", &r.ray_nanoseconds},
                      std::pair{"point_nanoseconds", &r.point_nanoseconds}}) {
      py::array_t<std::uint64_t> a({n, n, n});
      std::copy(pair.second->begin(), pair.second->end(), a.mutable_data());
      out[pair.first] = a;
    }
  }
  return out;
}
} // namespace
PYBIND11_MODULE(_b2_regions, m) {
  m.attr("ready_capable") = false;
  m.def("geometry_screen", &geometry_screen);
  py::class_<Options>(m, "Options", py::module_local())
      .def(py::init<>())
      .def_readwrite("max_nodes", &Options::max_nodes)
      .def_readwrite("max_depth", &Options::max_depth)
      .def_readwrite("precision", &Options::precision)
      .def_readwrite("query_seconds", &Options::query_seconds)
      .def_readwrite("taylor", &Options::taylor)
      .def_readwrite("taylor_depth", &Options::taylor_depth)
      .def_readwrite("tail_bounds", &Options::tail_bounds)
      .def_readwrite("filters", &Options::filters)
      .def_readwrite("exact_depth", &Options::exact_depth);
  py::class_<Settings>(m, "Settings")
      .def(py::init<>())
      .def_readwrite("mode", &Settings::mode)
      .def_readwrite("cover_depth", &Settings::cover_depth)
      .def_readwrite("cover_nodes", &Settings::cover_nodes)
      .def_readwrite("shadow_boxes", &Settings::shadow_boxes)
      .def_readwrite("parameter_nodes", &Settings::parameter_nodes)
      .def_readwrite("parameter_depth", &Settings::parameter_depth)
      .def_readwrite("cover_seconds", &Settings::cover_seconds)
      .def_readwrite("fast_exp", &Settings::fast_exp)
      .def_readwrite("volume_after_cache", &Settings::volume_after_cache)
      .def_readwrite("trace", &Settings::trace);
  py::class_<Engine>(m, "Engine")
      .def(py::init([](std::vector<P> c, std::vector<Matrix> a,
                       std::vector<double> w, double t, bool exponential,
                       bool independent) {
        return Engine(Inputs{c, a, w, t, exponential}, independent);
      }))
      .def("region",
           [](const Engine &e, P source, P lo, P hi, const Options &o,
              const Settings &cfg) {
             Stats s;
             std::pair<int, int> d;
             {
               py::gil_scoped_release release;
               d = e.region(source, lo, hi, o, cfg, s);
             }
             return py::make_tuple(d, s.n);
           })
      .def("field",
           [](const Engine &e, P source, int n, double seconds,
              const Options &o, const Settings &cfg) {
             Result r;
             {
               py::gil_scoped_release release;
               r = e.field(source, n, seconds, o, cfg);
             }
             return output(r, n);
           })
      .def("tile",
           [](const Engine &e, P source, int grid, std::array<int, 3> origin,
              int side, double seconds, const Options &o, const Settings &cfg) {
             Result r;
             {
               py::gil_scoped_release release;
               r = e.tile(source, grid, origin, side, seconds, o, cfg);
             }
             auto out = output(r, side);
             out["world_resolution"] = grid;
             out["origin_xyz"] = origin;
             return out;
           });
  m.def("exponent", [](double x, bool fast) {
    require_environment();
    Stats s;
    I v = certified_exp(x, fast, s);
    return py::make_tuple(v.lo, v.hi, s.n);
  });
  m.def("exponents",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> arg,
           bool fast) {
          require_environment();
          auto x = arg.unchecked<1>();
          if (x.shape(0) > 1000000)
            throw std::invalid_argument("Bounded exp batch required");
          py::array_t<double> bounds({x.shape(0), py::ssize_t(2)});
          auto a = bounds.mutable_unchecked<2>();
          Stats s;
          {
            py::gil_scoped_release release;
            for (py::ssize_t i = 0; i < x.shape(0); ++i) {
              I v = certified_exp(x(i), fast, s);
              a(i, 0) = v.lo;
              a(i, 1) = v.hi;
            }
          }
          return py::make_tuple(bounds, s.n);
        });
  m.def("exp_classes",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> arg,
           py::array_t<double, py::array::c_style | py::array::forcecast>
               thresholds,
           bool fast, bool fallback) {
          require_environment();
          auto x = arg.unchecked<1>(), tau = thresholds.unchecked<1>();
          if (x.shape(0) != tau.shape(0) || x.shape(0) > 1000000)
            throw std::invalid_argument("Invalid bounded classification batch");
          std::vector<int> out(x.shape(0));
          Stats s;
          {
            py::gil_scoped_release release;
            MP lo(256), hi(256);
            for (py::ssize_t i = 0; i < x.shape(0); ++i) {
              if (!(tau(i) > 0) || !std::isfinite(tau(i)))
                throw std::invalid_argument(
                    "Finite positive threshold required");
              I v = certified_exp(x(i), fast, s);
              int d = compare(v, I(tau(i)));
              if (d < 0 && fallback) {
                s.add("high_precision_fallbacks");
                mpfr_set_d(lo.v, x(i), MPFR_RNDN);
                mpfr_exp(lo.v, lo.v, MPFR_RNDD);
                mpfr_set_d(hi.v, x(i), MPFR_RNDN);
                mpfr_exp(hi.v, hi.v, MPFR_RNDU);
                s.add("high_precision_exp_endpoints", 2);
                d = mpfr_cmp_d(lo.v, tau(i)) >= 0  ? 1
                    : mpfr_cmp_d(hi.v, tau(i)) < 0 ? 0
                                                   : -1;
              }
              out[i] = d;
            }
          }
          return py::make_tuple(out, s.n);
        });
}
