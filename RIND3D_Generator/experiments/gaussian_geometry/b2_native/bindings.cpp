#include "common.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
namespace py = pybind11;
using namespace b2;
namespace {
py::dict decision(const Decision &d) {
  py::dict r;
  r["status"] = d.status;
  r["reason"] = d.reason;
  r["nodes"] = d.nodes;
  r["depth"] = d.depth;
  return r;
}
template <class Engine> void bind_engine(py::module_ &m, const char *name) {
  py::class_<Engine>(m, name)
      .def(py::init([](std::vector<P> c, std::vector<Matrix> a,
                       std::vector<double> w, double t,
                       bool exp) { return Engine(Inputs{c, a, w, t, exp}); }),
           py::arg("centers"), py::arg("matrices"), py::arg("weights"),
           py::arg("threshold"), py::arg("exponential") = false)
      .def("points",
           [](const Engine &e,
              py::array_t<double, py::array::c_style | py::array::forcecast> p,
              const Options &o) {
             o.validate();
             require_environment();
             auto a = p.unchecked<2>();
             if (a.shape(1) != 3)
               throw std::invalid_argument("Expected N x 3 points");
             std::vector<Decision> d(a.shape(0));
             Stats stats;
             {
               py::gil_scoped_release release;
               for (py::ssize_t i = 0; i < a.shape(0); i++)
                 d[i] = e.point({a(i, 0), a(i, 1), a(i, 2)}, o, stats);
             }
             py::list out;
             for (auto &v : d)
               out.append(decision(v));
             return py::make_tuple(out, stats.n);
           })
      .def("rays",
           [](const Engine &e, P source,
              py::array_t<double, py::array::c_style | py::array::forcecast> p,
              const Options &o) {
             o.validate();
             require_environment();
             auto a = p.unchecked<2>();
             if (a.shape(1) != 3)
               throw std::invalid_argument("Expected N x 3 receivers");
             std::vector<Decision> d(a.shape(0));
             Stats stats;
             {
               py::gil_scoped_release release;
               for (py::ssize_t i = 0; i < a.shape(0); i++)
                 d[i] = e.ray(source, {a(i, 0), a(i, 1), a(i, 2)}, o, stats);
             }
             py::list out;
             for (auto &v : d)
               out.append(decision(v));
             return py::make_tuple(out, stats.n);
           })
      .def("box",
           [](const Engine &e, P lo, P hi, const Options &o) {
             o.validate();
             require_environment();
             Stats s;
             Decision d;
             {
               py::gil_scoped_release release;
               d = e.box(lo, hi, o, s);
             }
             return py::make_tuple(decision(d), s.n);
           })
      .def("certificate",
           [](const Engine &e, P source, P lo, P hi, const Options &o) {
             o.validate();
             require_environment();
             Stats s;
             std::pair<int, int> d;
             {
               py::gil_scoped_release release;
               d = e.certificate(source, lo, hi, o, s);
             }
             return py::make_tuple(d, s.n);
           })
      .def(
          "field",
          [](const Engine &e, P source, int n, double seconds, const Options &o,
             bool scalar) {
            o.validate();
            require_environment();
            Field f;
            {
              py::gil_scoped_release release;
              f = e.field(source, n, seconds, o, scalar);
            }
            py::dict out;
            for (auto pair :
                 {std::pair{"response", &f.response},
                  std::pair{"solid", &f.solid}, std::pair{"valid", &f.valid}}) {
              py::array_t<std::uint8_t> a({n, n, n});
              std::copy(pair.second->begin(), pair.second->end(),
                        a.mutable_data());
              out[pair.first] = a;
            }
            out["complete"] = f.complete;
            out["reason"] = f.reason;
            out["wall_seconds"] = f.seconds;
            out["work"] = f.stats.n;
            out["resolution"] = n;
            return out;
          },
          py::arg("source"), py::arg("resolution"), py::arg("seconds"),
          py::arg("options"), py::arg("scalar") = false);
}
} // namespace
PYBIND11_MODULE(_b2_native, m) {
  m.doc() = "EXPERIMENTAL B2: directed certified filters and independent "
            "natural MPFR reference; no READY authority";
  m.attr("ready_capable") = false;
  m.attr("mpfr_version") = mpfr_get_version();
  py::class_<Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("coefficients", &Options::coefficients)
      .def_readwrite("filters", &Options::filters)
      .def_readwrite("groups", &Options::groups)
      .def_readwrite("regions", &Options::regions)
      .def_readwrite("inherited_regions", &Options::inherited_regions)
      .def_readwrite("tail_bounds", &Options::tail_bounds)
      .def_readwrite("taylor", &Options::taylor)
      .def_readwrite("taylor_depth", &Options::taylor_depth)
      .def_readwrite("exact_depth", &Options::exact_depth)
      .def_readwrite("fallback", &Options::fallback)
      .def_readwrite("max_nodes", &Options::max_nodes)
      .def_readwrite("max_depth", &Options::max_depth)
      .def_readwrite("precision", &Options::precision)
      .def_readwrite("query_seconds", &Options::query_seconds);
  bind_engine<Primary>(m, "Primary");
  bind_engine<Independent>(m, "Independent");
}
