"""Generate an auditable scalar-only hook into an UNCHANGED region traversal.

No new spatial/angular acceleration is introduced. All region, membership,
geometry, contact and canonical independent code comes from its frozen source.
Only the residual primary ray call is selectable. Proof checking remains an
additional experiment, never a replacement for the original independent field.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def source_identity():
    here=Path(__file__).resolve().parent;regions=here.parent/'b2_regions';native=here.parent/'b2_native'
    paths=[*regions.glob('*.cpp'),*regions.glob('*.hpp'),*native.glob('*.cpp'),*native.glob('*.hpp'),here/'field_bridge.py',here/'max.hpp',here/'solver.cpp',here/'checker.cpp']
    return {str(p.relative_to(here.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def build(output):
    here=Path(__file__).resolve().parent;regions=here.parent/'b2_regions';native=here.parent/'b2_native';src=output/'source';src.mkdir(parents=True,exist_ok=True);patch=[]
    hook='''
    if(independent || cfg.maximum_method==0)
      return independent ? b->ray(source,x,o,s):a->ray(source,x,o,s);
    b2max::Settings settings;settings.method=cfg.maximum_method;
    settings.max_nodes=o.max_nodes;settings.max_depth=o.max_depth;
    settings.seconds=o.query_seconds;settings.fallback=true;
    settings.endpoint_probes=cfg.initial_endpoint_probes;
    auto result=maximum->solve(source,x,settings,cfg.check_miss_proof);
    s.add("maximum_calls");s.add("maximum_nanoseconds",std::uint64_t(result.seconds*1e9));
    s.add("maximum_prepare_nanoseconds",std::uint64_t(result.preparation_seconds*1e9));
    for(auto [key,value]:result.stats.n) {
      if(key=="max_depth"||key=="active_nodes")s.maximum("maximum_"+key,value);
      else s.add("maximum_"+key,value);
    }
    if(result.fallback_used)s.add("maximum_old_fallback_calls");
    if(cfg.check_miss_proof && result.decision.status==0) {
      if(result.leaves.empty())s.add("MISS_without_new_partition_old_certified_fallback");
      else {
        auto checked=checker->check(source,x,result.leaves,cfg.checker_precision,o.query_seconds);
        s.add("MISS_proof_checks");s.add("checker_nanoseconds",std::uint64_t(checked.seconds*1e9));
        s.add("native_proof_payload_bytes",result.leaves.size()*sizeof(b2max::Leaf));
        s.maximum("peak_proof_intervals",result.leaves.size());
        for(auto [key,value]:checked.stats.n)s.add(key,value);
        if(!checked.accepted){s.add("MISS_proof_rejections");return {-1,"experimental_checker_rejected:"+checked.reason};}
      }
    }
    return result.decision;
'''
    for name in ('regions.hpp','regions.cpp','bindings.cpp'):
        old=(regions/name).read_text();text=old
        if name=='regions.hpp':text=text.replace('  int mode =','  int maximum_method=3,checker_precision=256;\n  bool check_miss_proof=false,initial_endpoint_probes=true;\n  int mode =')
        elif name=='regions.cpp':
            text='#include "max.hpp"\n'+text
            text=text.replace('  std::unique_ptr<Primary> a;','  std::unique_ptr<b2max::Solver> maximum;\n  std::unique_ptr<b2max::Checker> checker;\n  std::unique_ptr<Primary> a;')
            text=text.replace('      a = std::make_unique<Primary>(i);','      a = std::make_unique<Primary>(i);\n      maximum=std::make_unique<b2max::Solver>(i);\n      checker=std::make_unique<b2max::Checker>(i);')
            oldray='  Decision ray(P source, P x, const Options &o, Stats &s) const {\n    return independent ? b->ray(source, x, o, s) : a->ray(source, x, o, s);\n  }'
            assert text.count(oldray)==1;text=text.replace(oldray,'  Decision ray(P source,P x,const Options &o,Stats &s,const Settings &cfg)const {'+hook+'\n  }')
            assert text.count('p->ray(source, v, o, f.stats)')==1
            text=text.replace('p->ray(source, v, o, f.stats)','p->ray(source, v, o, f.stats,cfg)')
        else:
            text=text.replace('PYBIND11_MODULE(_b2_regions, m)','PYBIND11_MODULE(_b2_max_field, m)')
            text=text.replace('py::class_<Settings>(m, "Settings")','py::class_<Settings>(m, "Settings",py::module_local())')
            text=text.replace('py::class_<Engine>(m, "Engine")','py::class_<Engine>(m, "Engine",py::module_local())')
            text=text.replace('      .def_readwrite("mode", &Settings::mode)','      .def_readwrite("maximum_method", &Settings::maximum_method)\n      .def_readwrite("checker_precision", &Settings::checker_precision)\n      .def_readwrite("check_miss_proof", &Settings::check_miss_proof)\n      .def_readwrite("initial_endpoint_probes", &Settings::initial_endpoint_probes)\n      .def_readwrite("mode", &Settings::mode)')
        (src/name).write_text(text);patch.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=name,tofile='isolated-hook/'+name))
    (output/'scalar-hook.patch').write_text(''.join(patch))
    cmake='''cmake_minimum_required(VERSION 3.23)
project(b2_max_field LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)
find_library(GMPXX_LIBRARY gmpxx REQUIRED)
find_library(MPFR_LIBRARY mpfr REQUIRED)
'''
    cmake+=f'pybind11_add_module(_b2_max_field regions.cpp bindings.cpp "{here}/solver.cpp" "{here}/checker.cpp" "{native}/primary.cpp" "{native}/reference.cpp")\n'
    cmake+=f'target_include_directories(_b2_max_field PRIVATE "{native}" "{here}" "{regions}")\n'
    cmake+='target_link_libraries(_b2_max_field PRIVATE ${MPFR_LIBRARY} ${GMPXX_LIBRARY} ${GMP_LIBRARY})\ntarget_compile_options(_b2_max_field PRIVATE -Wall -Wextra -fno-fast-math -ffp-contract=off)\n'
    (src/'CMakeLists.txt').write_text(cmake)
    pycmake=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip();b=output/'build'
    subprocess.run(['cmake','-S',str(src.resolve()),'-B',str(b.resolve()),f'-DPython_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pycmake}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(b),'--parallel','2'],check=True)
    binary=list(b.glob('_b2_max_field*.so'));assert len(binary)==1
    (output/'identity.json').write_text(json.dumps({'sources':source_identity(),'binary_SHA256':hashlib.sha256(binary[0].read_bytes()).hexdigest(),
                                                  'hook_patch_SHA256':hashlib.sha256((output/'scalar-hook.patch').read_bytes()).hexdigest(),'only_residual_primary_ray_changed':True},indent=2)+'\n')


def extension(directory):
    binary=list((directory/'build').glob('_b2_max_field*.so'));identity=json.loads((directory/'identity.json').read_text());assert len(binary)==1
    assert identity['sources']==source_identity() and identity['binary_SHA256']==hashlib.sha256(binary[0].read_bytes()).hexdigest(),'Field-hook identity changed; rebuild separately'
    name='_b2_max_field'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,binary[0]);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
        m.__identity=identity
    assert sys.modules[name].__identity==identity,'Restart Python before changing a loaded library'
    return sys.modules[name]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1/field-hook-v3'));a=p.parse_args();build(a.output)
