#include "max.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace b2max;
namespace {
Inputs input(std::vector<P>c,std::vector<Matrix>m,std::vector<double>w,double t,bool e){return {c,m,w,t,e};}
}
PYBIND11_MODULE(_b2_max_cert,m) {
  m.attr("ready_capable")=false;m.attr("mpfr_version")=mpfr_get_version();
  py::class_<Settings>(m,"Settings").def(py::init<>())
    .def_readwrite("method",&Settings::method).def_readwrite("max_nodes",&Settings::max_nodes)
    .def_readwrite("max_depth",&Settings::max_depth).def_readwrite("seconds",&Settings::seconds).def_readwrite("fallback",&Settings::fallback)
    .def_readwrite("endpoint_probes",&Settings::endpoint_probes);
  py::class_<Leaf>(m,"Leaf").def(py::init<>())
    .def_readwrite("lo",&Leaf::lo).def_readwrite("hi",&Leaf::hi).def_readwrite("upper",&Leaf::upper)
    .def_readwrite("op",&Leaf::op).def_readwrite("jlo",&Leaf::jlo).def_readwrite("jhi",&Leaf::jhi)
    .def_readwrite("empty",&Leaf::empty).def_readwrite("inner_op",&Leaf::inner_op);
  py::class_<Solver>(m,"Solver")
    .def(py::init([](std::vector<P>c,std::vector<Matrix>a,std::vector<double>w,double t,bool e){return Solver(input(c,a,w,t,e));}))
    .def("solve",[](const Solver&s,P source,P receiver,const Settings&o,bool proof){
      Result r;{py::gil_scoped_release release;r=s.solve(source,receiver,o,proof);}
      py::dict d;d["status"]=r.decision.status;d["reason"]=r.decision.reason;d["nodes"]=r.decision.nodes;d["depth"]=r.decision.depth;
      d["seconds"]=r.seconds;d["preparation_seconds"]=r.preparation_seconds;d["fallback_used"]=r.fallback_used;d["work"]=r.stats.n;
      py::list leaves;for(const auto &l:r.leaves)leaves.append(py::cast(l));d["leaves"]=leaves;return d;
    },py::arg("source"),py::arg("receiver"),py::arg("settings"),py::arg("proof")=false)
    .def("bounds",&Solver::bounds);
  py::class_<Checker>(m,"Checker")
    .def(py::init([](std::vector<P>c,std::vector<Matrix>a,std::vector<double>w,double t,bool e){return Checker(input(c,a,w,t,e));}))
    .def("check",[](const Checker&c,P s,P x,const std::vector<Leaf>&leaves,int bits,double seconds){
      CheckResult r;{py::gil_scoped_release release;r=c.check(s,x,leaves,bits,seconds);}
      py::dict d;d["accepted"]=r.accepted;d["reason"]=r.reason;d["seconds"]=r.seconds;d["preparation_seconds"]=r.preparation_seconds;d["work"]=r.stats.n;return d;
    },py::arg("source"),py::arg("receiver"),py::arg("leaves"),py::arg("precision")=128,py::arg("seconds")=2.);
}
