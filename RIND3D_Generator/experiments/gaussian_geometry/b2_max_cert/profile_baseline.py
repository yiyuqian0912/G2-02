"""Build a timing-only copy of the unchanged optimized native baseline.

Generated instrumented sources and visit traces remain outside Git. The saved
source diff is auditable; this module never edits the original B2 implementation.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def build(output):
    original=Path(__file__).resolve().parent.parent/'b2_native'
    src=output/'source';src.mkdir(parents=True,exist_ok=True)
    hashes={};patch=[]
    header='''#pragma once
#include "common.hpp"
namespace b2 {
inline thread_local std::vector<std::array<double,4>> profile_visits;
struct ProfileTimer {
  Stats &s; const char *key; Clock::time_point start=Clock::now();
  ProfileTimer(Stats &s,const char *key):s(s),key(key){}
  ~ProfileTimer(){s.add(key,std::uint64_t(elapsed(start)*1e9));}
};
inline void profile_visit(double l,double h,int status,int kind){
 if(profile_visits.size()<100000)profile_visits.push_back({l,h,double(status),double(kind)});
}
}
'''
    (src/'profile.hpp').write_text(header)
    for path in sorted(original.iterdir()):
        if path.suffix not in ('.cpp','.hpp') and path.name!='CMakeLists.txt':continue
        text=old=path.read_text();hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name=='reference.cpp':
            text='#include "profile.hpp"\n'+text
            text=text.replace('    std::vector<Line> out;', '    ProfileTimer timer(s,"profile_line_preparation_ns");\n    std::vector<Line> out;')
            text=text.replace('    Scratch z;\n    for (std::size_t i = 0; i < kernels.size(); i++) {\n      mpfr_set_zero(z.ql.v, 0);', '    ProfileTimer timer(s,"profile_natural_bounds_ns");\n    Scratch z;\n    for (std::size_t i = 0; i < kernels.size(); i++) {\n      mpfr_set_zero(z.ql.v, 0);')
            # Exactly one natural() terminal; spatial() uses a different loop.
            needle='    return decide(z);\n  }\n  void derivatives'
            assert text.count(needle)==1
            text=text.replace(needle,'    int status=decide(z); profile_visit(low,high,status,0);\n    return status;\n  }\n  void derivatives')
            text=text.replace('    mpfr_set_zero(second.v, 0);','    ProfileTimer timer(s,"profile_derivative_preparation_ns");\n    mpfr_set_zero(second.v, 0);')
            text=text.replace('    Scratch z;\n    MP dl(bits)', '    ProfileTimer timer(s,"profile_taylor_bounds_ns");\n    Scratch z;\n    MP dl(bits)')
            text=text.replace('      s.add("reference_taylor_interval_free");','      profile_visit(low,high,0,1);\n      s.add("reference_taylor_interval_free");')
            text=text.replace('    return -1;\n  }\n  Decision analytic_ray','    profile_visit(low,high,-1,1);\n    return -1;\n  }\n  Decision analytic_ray')
        if path.name=='bindings.cpp':
            text='#include "profile.hpp"\n'+text
            text=text.replace('py::class_<Engine>(m, name)', 'py::class_<Engine>(m, name, py::module_local())')
            text=text.replace('py::class_<Options>(m, "Options")', 'py::class_<Options>(m, "Options", py::module_local())')
            text=text.replace('PYBIND11_MODULE(_b2_native, m)', 'PYBIND11_MODULE(_b2_profile, m)')
            # Clear visits immediately before an explicit batch, export separately.
            text=text.replace('             std::vector<Decision> d(a.shape(0));','             profile_visits.clear();\n             std::vector<Decision> d(a.shape(0));')
            text=text.replace('  bind_engine<Primary>(m, "Primary");','  m.def("visits",[](){return profile_visits;});\n  bind_engine<Primary>(m, "Primary");')
        if path.name=='CMakeLists.txt':text=text.replace('_b2_native','_b2_profile')
        (src/path.name).write_text(text)
        patch.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=path.name,tofile='instrumented/'+path.name))
    (output/'instrumentation.patch').write_text(''.join(patch))
    cmake=subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'],text=True).strip()
    builddir=output/'build'
    subprocess.run(['cmake','-S',str(src.resolve()),'-B',str(builddir.resolve()),f'-DPython_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={cmake}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(builddir),'--parallel','2'],check=True)
    (output/'identity.json').write_text(json.dumps({'original_source_SHA256':hashes,'instrumentation_SHA256':hashlib.sha256(''.join(patch).encode()).hexdigest(),'physical_algorithm_changed':False},indent=2)+'\n')


def extension(output):
    paths=list((output/'build').glob('_b2_profile*.so'));assert len(paths)==1
    spec=importlib.util.spec_from_file_location('_b2_profile',paths[0]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/gaussian-b2-max-cert-v1/profile-baseline-v2'));a=p.parse_args();build(a.output)
