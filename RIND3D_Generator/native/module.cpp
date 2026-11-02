#include "pack.hpp"
#include "filter.hpp"
#include "functional.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <rindray/rindray.hpp>
#include <cmath>
namespace py=pybind11;
using Array=py::array_t<double,py::array::c_style>;
static std::vector<r3::Primitive> primitives(const py::list& rows){
 std::vector<r3::Primitive> p;for(auto row:rows){auto t=py::cast<py::tuple>(row);r3::Primitive x{py::cast<int>(t[0]),py::cast<std::array<double,3>>(t[1]),py::cast<std::array<double,3>>(t[2])};
 if(x.kind<0||x.kind>1)throw std::invalid_argument("Unknown filter primitive");
 for(int a=0;a<3;++a)if(!std::isfinite(x.center[a])||!std::isfinite(x.half[a])||x.half[a]<=0)throw std::invalid_argument("Invalid primitive");p.push_back(x);}return p;
}
static void points(Array a){if(a.ndim()!=2||a.shape(1)!=3)throw std::invalid_argument("Expected contiguous float64 [N,3]");for(auto v:std::span(a.data(),a.size()))if(!std::isfinite(v)||v<0||v>1)throw std::invalid_argument("Point outside normalized world");}
PYBIND11_MODULE(_native,m){
 m.doc()="RINDPack encoder and conservative interval filters; exact oracle fallback is external";
 py::class_<r3::Writer>(m,"PackWriter")
 .def(py::init([](std::string p,std::array<std::uint32_t,3>s,py::bytes i,py::bytes n,bool resume,int level){return new r3::Writer(p,s,py::cast<std::string>(i),py::cast<std::string>(n),resume,level);} ))
 .def("write",[](r3::Writer&w,std::size_t i,py::bytes b){auto s=py::cast<std::string>(b);py::gil_scoped_release r;w.write(i,s);})
 .def("committed",&r3::Writer::committed).def("checkpoint",&r3::Writer::checkpoint).def("finish",&r3::Writer::finish);
 m.def("encode_brick",[](py::bytes b,int level){auto e=r3::encode(py::cast<std::string>(b),level);return py::make_tuple(e.codec,py::bytes(e.payload),e.checksum,e.ones);},py::arg("raw"),py::arg("level")=3);
 py::class_<r3::Filter>(m,"IntervalFilter")
 .def(py::init([](py::list p){return r3::Filter(primitives(p));}))
 .def("contains",[](r3::Filter&f,Array p){points(p);py::array_t<std::int8_t>out(p.shape(0));{py::gil_scoped_release r;for(py::ssize_t i=0;i<p.shape(0);++i)out.mutable_data()[i]=f.contains({p.data()[3*i],p.data()[3*i+1],p.data()[3*i+2]});}return out;})
 .def("visibility",[](r3::Filter&f,Array s,Array p){points(s);points(p);auto n=std::max(s.shape(0),p.shape(0));if((s.shape(0)!=1&&s.shape(0)!=n)||(p.shape(0)!=1&&p.shape(0)!=n))throw std::invalid_argument("Paired or singleton source/receiver arrays required");py::array_t<std::int8_t>out(n);{py::gil_scoped_release r;for(py::ssize_t i=0;i<n;++i){auto a=s.data()+3*(s.shape(0)==1?0:i),b=p.data()+3*(p.shape(0)==1?0:i);out.mutable_data()[i]=f.visibility({a[0],a[1],a[2]},{b[0],b[1],b[2]});}}return out;})
 .def("certificate",&r3::Filter::certificate);
 m.def("rindray_reference",[](py::list rows,Array source,Array p,unsigned threads){
  points(source);points(p);if(source.shape(0)!=1)throw std::invalid_argument("One source required");
  rindray::Scene scene("reference",threads);auto parts=primitives(rows);int id=0;
  for(auto&v:parts){std::array<float,3>c,h;for(int a=0;a<3;++a){c[a]=v.center[a];h[a]=2*v.half[a];}if(v.kind==0)scene.add_box(c,h,id++);else{if(v.half[0]!=v.half[1]||v.half[1]!=v.half[2])throw std::invalid_argument("RINDRay does not support general ellipsoids");scene.add_sphere(c,v.half[0],id++);}}
  scene.commit();std::vector<float>q(p.size()),s(3);for(py::ssize_t i=0;i<p.size();++i)q[i]=p.data()[i];for(int a=0;a<3;++a)s[a]=source.data()[a];
  py::array_t<bool>out(p.shape(0));std::array<std::int32_t,1>kind{0};
  {py::gil_scoped_release r;scene.visibility_into(s,kind,q,std::span(out.mutable_data(),out.size()),{0,0,0,1,1,1});}return out;
 },py::arg("primitives"),py::arg("source"),py::arg("points"),py::arg("threads")=1);
 m.def("rindray_build_id",&rindray::build_id);
 m.def("embree_available",&rindray::embree_available);
 py::class_<r3::CertifiedSpace>(m,"CertifiedSpace")
 .def(py::init<std::vector<std::int8_t>,std::vector<std::int32_t>>())
 .def("contains",[](const r3::CertifiedSpace& f,Array p){points(p);py::array_t<std::int8_t>out(p.shape(0));{py::gil_scoped_release r;for(py::ssize_t i=0;i<p.shape(0);++i)out.mutable_data()[i]=f.contains({p.data()[3*i],p.data()[3*i+1],p.data()[3*i+2]});}return out;})
 .def("visibility",[](const r3::CertifiedSpace&f,std::array<double,3>s,Array p){points(p);py::array_t<std::int8_t>out(p.shape(0));std::uint64_t nodes=0;{py::gil_scoped_release r;for(py::ssize_t i=0;i<p.shape(0);++i)out.mutable_data()[i]=f.visibility(s,{p.data()[3*i],p.data()[3*i+1],p.data()[3*i+2]},nodes);}return py::make_tuple(out,nodes);})
 .def("certificate",[](const r3::CertifiedSpace&f,r3::Point s,r3::Point lo,r3::Point hi){std::uint64_t nodes=0;int value;{py::gil_scoped_release r;value=f.certificate(s,lo,hi,nodes);}return py::make_tuple(value,nodes);});
 py::class_<r3::CertifiedGrid>(m,"CertifiedGrid")
 .def(py::init([](unsigned n,py::array_t<double,py::array::c_style>coefficients,py::array_t<double,py::array::c_style>basis,std::array<double,2>bias,int constant){
  return r3::CertifiedGrid(n,std::vector<double>(coefficients.data(),coefficients.data()+coefficients.size()),std::vector<double>(basis.data(),basis.data()+basis.size()),bias,constant);}))
 .def("contains",[](const r3::CertifiedGrid&f,Array p){points(p);py::array_t<std::int8_t>out(p.shape(0));{py::gil_scoped_release r;for(py::ssize_t i=0;i<p.shape(0);++i)out.mutable_data()[i]=f.contains({p.data()[3*i],p.data()[3*i+1],p.data()[3*i+2]});}return out;});
}
