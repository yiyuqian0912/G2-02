#include "angular.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace b2a;
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
  out["angular_seconds"] = r.angular_seconds;
  out["adaptive_seconds"] = r.adaptive_seconds;
  out["fallback_seconds"] = r.fallback_seconds;
  out["assembly_seconds"] = r.assembly_seconds;
  out["angular_setup_seconds"] = r.angular_setup_seconds;
  out["angular_leaf_seconds"] = r.angular_leaf_seconds;
  out["point_cover_seconds"] = r.point_cover_seconds;
  if (!r.evaluation_path.empty()) {
    for (auto pair :
         {std::pair{"evaluation_path", &r.evaluation_path},
          std::pair{"visibility_evidence", &r.visibility_evidence}}) {
      py::array_t<std::uint8_t> a({n, n, n});
      std::copy(pair.second->begin(), pair.second->end(), a.mutable_data());
      out[pair.first] = a;
    }
    for (auto pair : {std::pair{"ray_nanoseconds", &r.ray_nanoseconds},
                      std::pair{"point_nanoseconds", &r.point_nanoseconds}}) {
      py::array_t<std::uint64_t> a({n, n, n});
      std::copy(pair.second->begin(), pair.second->end(), a.mutable_data());
      out[pair.first] = a;
    }
    out["angular_tiles"] = r.angular_tiles;
  }
  return out;
}
} // namespace
PYBIND11_MODULE(_b2_angular, m) {
  m.attr("ready_capable") = false;
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
      .def_readwrite("method", &Settings::method)
      .def_readwrite("cover_depth", &Settings::cover_depth)
      .def_readwrite("cover_nodes", &Settings::cover_nodes)
      .def_readwrite("shadow_boxes", &Settings::shadow_boxes)
      .def_readwrite("adaptive_depth", &Settings::adaptive_depth)
      .def_readwrite("adaptive_nodes", &Settings::adaptive_nodes)
      .def_readwrite("adaptive_batch", &Settings::adaptive_batch)
      .def_readwrite("adaptive_seconds", &Settings::adaptive_seconds)
      .def_readwrite("angular_depth", &Settings::angular_depth)
      .def_readwrite("angular_nodes", &Settings::angular_nodes)
      .def_readwrite("angular_references", &Settings::angular_references)
      .def_readwrite("angular_boxes", &Settings::angular_boxes)
      .def_readwrite("prefix_builds", &Settings::prefix_builds)
      .def_readwrite("cover_seconds", &Settings::cover_seconds)
      .def_readwrite("empty_prefix", &Settings::empty_prefix)
      .def_readwrite("trace", &Settings::trace);
  py::class_<Engine>(m, "Engine")
      .def(py::init([](std::vector<P> c, std::vector<Matrix> a,
                       std::vector<double> w, double t, bool exp, bool ref) {
        return Engine(Inputs{c, a, w, t, exp}, ref);
      }))
      .def("tile",
           [](const Engine &e, P s, int grid, std::array<int, 3> origin,
              int size, double seconds, const Options &o, const Settings &cfg) {
             Result r;
             {
               py::gil_scoped_release release;
               r = e.tile(s, grid, origin, size, seconds, o, cfg);
             }
             auto out = output(r, size);
             out["world_resolution"] = grid;
             out["origin_xyz"] = origin;
             return out;
           });
  m.def("chart", [](P s, P lo, P hi) {
    require_environment();
    if (!finite(s) || !good({lo, hi}))
      throw std::invalid_argument("Invalid chart input");
    auto q = chart(s, {lo, hi});
    return py::make_tuple(q.face, q.u0, q.u1, q.v0, q.v1, q.near, q.far);
  });
  // Geometry-only diagnostic: callers must independently establish Q subset K
  // before these abstract boxes can provide physical evidence for B2.
  m.def(
      "screen",
      [](P source, P lo, P hi, std::vector<P> inner_lo, std::vector<P> inner_hi,
         bool ref, int depth, int nodes, bool fusion) {
        if (inner_lo.size() != inner_hi.size() || inner_lo.size() > 100000)
          throw std::invalid_argument("Bounded paired boxes required");
        Stats stats;
        AngularOptions cfg;
        cfg.depth = depth;
        cfg.nodes = nodes;
        cfg.fusion = fusion;
        AngularMap map(source, ref, cfg, stats);
        for (std::size_t i = 0; i < inner_lo.size(); ++i)
          map.add_inner({inner_lo[i], inner_hi[i]});
        auto d = map.query({lo, hi});
        py::dict r;
        r["visibility"] = d.visibility;
        r["first_hit_upper"] = d.first_hit_upper;
        r["distinct_blockers"] = d.distinct_blockers;
        r["work"] = stats.n;
        r["tiles"] = map.snapshot();
        return r;
      },
      py::arg("source"), py::arg("lo"), py::arg("hi"), py::arg("inner_lo"),
      py::arg("inner_hi"), py::arg("independent") = false,
      py::arg("depth") = 10, py::arg("nodes") = 32766,
      py::arg("fusion") = true);
}
