"""Build a residual-ray-only experiment from fingerprinted B2 source files."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ..b2_angular.preservation import sha
from ..study import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT = Path("outputs/gaussian-b2-independent-v1")


def source_files():
    for path in (ROOT / "b2_native", ROOT / "b2_regions"):
        yield from (p for p in sorted(path.iterdir()) if p.suffix in (".cpp", ".hpp", ".py") or p.name == "CMakeLists.txt")

    yield from (HERE / name for name in ("build.py", "profile.hpp", "local_search.hpp"))


def fingerprint():
    return {str(p.relative_to(ROOT)): sha(p) for p in source_files()}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Original B2 hook no longer matches: {old}")
    return text.replace(old, new, 1)


def reference(text, profile):
    def edit(old, new):
        nonlocal text
        text = replace_once(text, old, new)

    edit("  Bound() : l(bits), h(bits) {}", "  explicit Bound(int p = bits) : l(p), h(p) {}")
    edit("  Bound derivative_offset, curvature;\n};", """  Bound derivative_offset, curvature;
  explicit Line(int p = bits)
      : offset{Bound(p), Bound(p), Bound(p)},
        slope{Bound(p), Bound(p), Bound(p)} {}
};""")
    edit("  Scratch()\n      : ql(bits), qh(bits), lo(bits), hi(bits), x(bits), y(bits), t(bits),\n"
         "        suml(bits), sumh(bits), dl(bits), dh(bits), gl(bits), gh(bits) {};",
         "  explicit Scratch(int p = bits)\n"
         "      : ql(p), qh(p), lo(p), hi(p), x(p), y(p), t(p), suml(p), sumh(p),\n"
         "        dl(p), dh(p), gl(p), gh(p) {}")
    edit("std::vector<Line> lines(P source, P receiver, Stats &s) const {",
         "std::vector<Line> lines(P source, P receiver, Stats &s, int p = bits) const {")
    edit("    Scratch z;\n    for (auto &k : kernels) {\n      Line line;",
         "    Scratch z(p);\n    for (auto &k : kernels) {\n      Line line(p);")
    edit("bool tails) const {\n    Scratch z;\n    for (std::size_t i = 0; i < kernels.size(); i++) {",
         """bool tails, int p = bits, std::vector<Bound> *terms = nullptr,
              Bound *sum = nullptr) const {
    Scratch z(p);
    for (std::size_t i = 0; i < kernels.size(); i++) {""")
    edit("      accumulate(z, kernels[i].w, s, tails);\n    }\n    return decide(z);",
         """      accumulate(z, kernels[i].w, s, tails);
      if (terms) {
        mpfr_set((*terms)[i].l.v, z.gl.v, MPFR_RNDD);
        mpfr_set((*terms)[i].h.v, z.gh.v, MPFR_RNDU);
      }
    }
    if (sum) {
      mpfr_set(sum->l.v, z.suml.v, MPFR_RNDD);
      mpfr_set(sum->h.v, z.sumh.v, MPFR_RNDU);
    }
    return decide(z);""")
    if profile:
        text = '#include "profile.hpp"\n' + text
        for rounding in ("RNDD", "RNDU"):
            edit(f"  mpfr_exp(z.x.v, z.x.v, MPFR_{rounding});",
                 f"  profile_exp(z.x.v, z.x.v, MPFR_{rounding});")
        edit("    std::vector<Line> out;", '    ProfileTimer timer{s, "profile_line_ns"};\n    std::vector<Line> out;')
        edit("    Scratch z(p);\n    for (std::size_t i = 0; i < kernels.size(); i++) {",
             """    ProfileTimer timer{s, low == high ? "profile_midpoint_ns" : "profile_natural_ns"};
    timer.full_exp = !tails;
    if (profile_ray_active && !tails) s.add("profile_full_exp_calls");
    Scratch z(p);
    for (std::size_t i = 0; i < kernels.size(); i++) {""")
        edit("    return decide(z);\n  }\n  void derivatives",
             """    int state = decide(z);
    timer.failed = state < 0;
    return state;
  }
  void derivatives""")
        edit("    mpfr_set_zero(second.v, 0);", '    ProfileTimer timer{s, "profile_exact_coefficients_ns"};\n    mpfr_set_zero(second.v, 0);')
        edit("    Scratch z;\n    MP dl(bits)", '    ProfileTimer timer{s, "profile_taylor_ns"};\n    Scratch z;\n    MP dl(bits)')
        edit('    return -1;\n  }\n  Decision analytic_ray', '    timer.failed = true;\n    return -1;\n  }\n  Decision analytic_ray')
        edit('  s.add("reference_ray_queries");', '  ProfileScope scope(s);\n  ProfileTimer timer{s, "profile_ray_ns"};\n  s.add("reference_ray_queries");')
        edit("    queue.push_back({n.l, mid, n.d + 1});",
             '    s.add("profile_ray_splits");\n    queue.push_back({n.l, mid, n.d + 1});')
    return text + '\n#include "local_search.hpp"\n'


def build(output, profile=False):
    if (output / "identity.json").exists():
        raise ValueError("Choose a new build directory to preserve the measured build")
    src = output / "source"
    src.mkdir(parents=True, exist_ok=True)
    patch = []

    def write(original, text):
        old = original.read_text()
        (src / original.name).write_text(text)
        patch.extend(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                         fromfile=str(original.relative_to(ROOT)), tofile=f"generated/{original.name}"))

    native, regions = ROOT / "b2_native", ROOT / "b2_regions"
    for name in ("primary.cpp", "field.hpp", "interval.hpp"):
        shutil.copyfile(native / name, src / name)
    shutil.copyfile(regions / "exp.hpp", src / "exp.hpp")
    for name in ("profile.hpp", "local_search.hpp"):
        shutil.copyfile(HERE / name, src / name)
    p = native / "common.hpp"
    common = replace_once(p.read_text(), "struct Stats {", """struct RayStudy {
  int method = 0; // 0 original, 1 precision, 2 curvature, 3 combined.
  int filter_bits = 64, filter_depth = 3, curvature_depth = 4;
  bool stationary = false;
  void validate() const;
};
struct Stats {""")
    start = common.index("class Independent {")
    common = common[:start] + replace_once(common[start:],
        "  Decision ray(P, P, const Options &, Stats &) const;",
        """  Decision ray(P, P, const Options &, Stats &) const;
  Decision study_ray(P, P, const Options &, Stats &, const RayStudy &) const;""")
    write(p, common)
    p = native / "reference.cpp"
    write(p, reference(p.read_text(), profile))
    p = regions / "regions.hpp"
    header = replace_once(p.read_text(), "struct Settings {", "struct Settings {\n  RayStudy study;")
    write(p, replace_once(header, "  Engine(const Inputs &, bool independent);",
         "  Engine(const Inputs &, bool independent);\n"
         "  Decision scalar(P, P, const Options &, const Settings &, Stats &) const;"))
    p = regions / "regions.cpp"
    cpp = replace_once(p.read_text(),
        """  Decision ray(P source, P x, const Options &o, Stats &s) const {
    return independent ? b->ray(source, x, o, s) : a->ray(source, x, o, s);
  }""",
        """  Decision ray(P source, P x, const Options &o, Stats &s, const RayStudy &study) const {
    if (!independent) return a->ray(source, x, o, s);
    if (study.method == 0) return b->ray(source, x, o, s);
    return b->study_ray(source, x, o, s, study);
  }""")
    cpp = replace_once(cpp, "p->ray(source, v, o, f.stats)", "p->ray(source, v, o, f.stats, cfg.study)")
    cpp = replace_once(cpp, "void Settings::validate() const {", "void Settings::validate() const {\n  study.validate();")
    write(p, cpp + """
namespace b2r {
Decision Engine::scalar(P source, P receiver, const Options &o,
                        const Settings &cfg, Stats &stats) const {
  o.validate();
  cfg.study.validate();
  require_environment();
  return p->ray(source, receiver, o, stats, cfg.study);
}
} // namespace b2r
""")
    p = regions / "bindings.cpp"
    bindings = replace_once(p.read_text(), "PYBIND11_MODULE(_b2_regions, m)", "PYBIND11_MODULE(_b2_independent_study, m)")
    bindings = replace_once(bindings, '  py::class_<Settings>(m, "Settings")',
        '''  py::class_<RayStudy>(m, "RayStudy", py::module_local())
      .def(py::init<>())
      .def_readwrite("method", &RayStudy::method)
      .def_readwrite("filter_bits", &RayStudy::filter_bits)
      .def_readwrite("filter_depth", &RayStudy::filter_depth)
      .def_readwrite("curvature_depth", &RayStudy::curvature_depth)
      .def_readwrite("stationary", &RayStudy::stationary);
  py::class_<Settings>(m, "Settings", py::module_local())''')
    bindings = replace_once(bindings, '      .def_readwrite("mode", &Settings::mode)',
        '      .def_readwrite("study", &Settings::study)\n      .def_readwrite("mode", &Settings::mode)')
    bindings = replace_once(bindings, 'py::class_<Engine>(m, "Engine")', 'py::class_<Engine>(m, "Engine", py::module_local())')
    bindings = replace_once(bindings, '      .def("region",', '''      .def("ray",
           [](const Engine &engine, P source, P receiver, const Options &options, const Settings &cfg) {
             Stats stats;
             auto start = Clock::now();
             auto d = engine.scalar(source, receiver, options, cfg, stats);
             double seconds = elapsed(start);
             py::dict out;
             out["status"] = d.status;
             out["reason"] = d.reason;
             out["nodes"] = d.nodes;
             out["depth"] = d.depth;
             out["native_seconds"] = seconds;
             out["work"] = stats.n;
             return out;
           })
      .def("region",''')
    write(p, bindings)
    p = regions / "CMakeLists.txt"
    cmake = p.read_text().replace("_b2_regions", "_b2_independent_study")
    cmake = cmake.replace("../b2_native/", "").replace("PRIVATE ../b2_native)", "PRIVATE .)")
    write(p, cmake)
    (output / "generated.patch").write_text("".join(patch))
    builddir = output / "build"
    pybind = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    subprocess.run(["cmake", "-S", str(src), "-B", str(builddir), f"-DPython_EXECUTABLE={sys.executable}",
                    f"-Dpybind11_DIR={pybind}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(builddir), "--parallel", "2"], check=True)
    (binary,) = builddir.glob("_b2_independent_study*.so")
    atomic_json(output / "identity.json", {
        "sources": fingerprint(),
        "binary_SHA256": sha(binary),
        "generated_patch_SHA256": sha(output / "generated.patch"),
        "profile": profile,
    })


def extension(directory):
    name = "_b2_independent_study"
    try:
        (binary,) = (directory / "build").glob(name + "*.so")
        identity = json.loads((directory / "identity.json").read_text())
    except (FileNotFoundError, ValueError) as e:
        raise ImportError("Build the isolated B2 study first") from e
    if identity["sources"] != fingerprint() or identity["binary_SHA256"] != sha(binary):
        raise ImportError("B2 study identity differs from its sources or binary; rebuild in a new directory")
    if name in sys.modules:
        if sys.modules[name].study_identity != identity:
            raise ImportError("Restart the interpreter to switch study builds")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.study_identity = identity
    sys.modules[name] = module
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT / "native")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    build(args.output, args.profile)
