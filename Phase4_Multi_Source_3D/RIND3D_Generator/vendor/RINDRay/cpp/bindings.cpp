#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>
#include <rindray/rindray.hpp>
#include "vulkan.hpp"

namespace nb=nanobind;
using namespace nb::literals;
template<typename T> using Input=nb::ndarray<nb::numpy,const T,nb::c_contig,nb::device::cpu>;
template<typename T> using Output=nb::ndarray<nb::numpy,T,nb::c_contig,nb::device::cpu>;
template<typename T> auto span(Input<T> a) { return std::span<const T>(a.data(),a.size()); }
template<typename T> auto span(Output<T> a) { return std::span<T>(a.data(),a.size()); }

NB_MODULE(_core, module) {
    module.def("embree_available", &rindray::embree_available);
    module.def("embree_version", &rindray::embree_version);
    module.def("build_id", &rindray::build_id);
    module.def("vulkan_compiled",&rindray::vulkan_compiled);
    module.def("vulkan_capabilities",[] {
        rindray::VulkanCapabilities c;
        {nb::gil_scoped_release release;c=rindray::vulkan_capabilities();}
        nb::dict out;out["compiled"]=c.compiled;out["available"]=c.available;out["name"]=c.name;out["error"]=c.error;
        out["api_version"]=c.api_version;out["driver_version"]=c.driver_version;out["ray_query"]=c.ray_query;
        out["device_memory_bytes"]=c.device_memory_bytes;
        out["acceleration_structure"]=c.acceleration_structure;out["buffer_device_address"]=c.buffer_device_address;out["compute_queue"]=c.compute_queue;return out;
    });
    nb::class_<rindray::VulkanDevice>(module,"VulkanDevice");
    module.def("create_vulkan_device",&rindray::create_vulkan_device,nb::call_guard<nb::gil_scoped_release>());
    nb::class_<rindray::Scene>(module,"Scene")
        .def(nb::init<std::string,unsigned,std::size_t,std::shared_ptr<rindray::VulkanDevice>>(),
             "backend"_a="reference","num_threads"_a=1,"memory_budget"_a=1024ull*1024*1024,"device"_a=nb::none())
        .def("add_box",&rindray::Scene::add_box)
        .def("add_sphere",&rindray::Scene::add_sphere)
        .def("add_instance",&rindray::Scene::add_instance,nb::call_guard<nb::gil_scoped_release>())
        .def("add_mesh",[](rindray::Scene& scene,Input<float> v,Input<std::uint32_t> f,std::int32_t id) {
            nb::gil_scoped_release release; scene.add_mesh(span(v),span(f),id);
        },"vertices"_a.noconvert(),"indices"_a.noconvert(),"geometry_id"_a)
        .def("commit",&rindray::Scene::commit,nb::call_guard<nb::gil_scoped_release>())
        .def("bounds",&rindray::Scene::bounds,nb::call_guard<nb::gil_scoped_release>())
        .def("configure",&rindray::Scene::configure,nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("geometry_counts",&rindray::Scene::geometry_counts)
        .def_prop_ro("backend",&rindray::Scene::backend)
        .def_prop_ro("num_threads",&rindray::Scene::num_threads)
        .def("gpu_metrics",&rindray::Scene::gpu_metrics,nb::call_guard<nb::gil_scoped_release>())
        .def("trim_gpu_workspace",&rindray::Scene::trim_gpu_workspace,nb::call_guard<nb::gil_scoped_release>())
        .def("trace_into",[](const rindray::Scene& scene,Input<float> o,Input<float> d,Input<float> lo,Input<float> hi,
                             Output<bool> hit,Output<float> t,Output<std::int32_t> id,Output<float> normal,std::size_t scratch) {
            nb::gil_scoped_release release;
            scene.trace_into({span(o),span(d),span(lo),span(hi)},{span(hit),span(t),span(id),span(normal)},scratch);
        },"origins"_a.noconvert(),"directions"_a.noconvert(),"tmin"_a.noconvert(),"tmax"_a.noconvert(),
          "hit"_a.noconvert(),"t"_a.noconvert(),"geometry_id"_a.noconvert(),"normal"_a.noconvert(),"scratch_budget"_a=std::numeric_limits<std::size_t>::max())
        .def("occluded_into",[](const rindray::Scene& scene,Input<float> o,Input<float> d,Input<float> lo,Input<float> hi,Output<bool> out,std::size_t scratch) {
            nb::gil_scoped_release release; scene.occluded_into({span(o),span(d),span(lo),span(hi)},span(out),scratch);
        },"origins"_a.noconvert(),"directions"_a.noconvert(),"tmin"_a.noconvert(),"tmax"_a.noconvert(),"out"_a.noconvert(),"scratch_budget"_a=std::numeric_limits<std::size_t>::max())
        .def("contains_into",[](const rindray::Scene& scene,Input<float> points,Output<bool> out) {
            nb::gil_scoped_release release; scene.contains_into(span(points),span(out));
        },"points"_a.noconvert(),"out"_a.noconvert())
        .def("visibility_into",[](const rindray::Scene& scene,Input<float> sources,Input<std::int32_t> kinds,Input<float> points,
                                  Output<bool> out,std::array<float,6> bounds,float epsilon,std::size_t scratch) {
            nb::gil_scoped_release release; scene.visibility_into(span(sources),span(kinds),span(points),span(out),bounds,epsilon,scratch);
        },"sources"_a.noconvert(),"kinds"_a.noconvert(),"points"_a.noconvert(),"out"_a.noconvert(),"bounds"_a,"epsilon_multiplier"_a=4,"scratch_budget"_a=std::numeric_limits<std::size_t>::max())
        .def("validate_visibility",[](const rindray::Scene& scene,Input<float> sources,Input<std::int32_t> kinds,
                                      Input<float> points,std::array<float,6> bounds,float epsilon) {
            nb::gil_scoped_release release; scene.validate_visibility(span(sources),span(kinds),span(points),bounds,epsilon);
        },"sources"_a.noconvert(),"kinds"_a.noconvert(),"points"_a.noconvert(),"bounds"_a,"epsilon_multiplier"_a=4);
}
