"""Build a versioned derivative of the fingerprinted independent B2 study."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from ..b2_angular.preservation import sha
from ..b2_independent.build import replace_once
from ..study import atomic_json

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "configs/b2-precision-v1.json"
OUT = Path("outputs/gaussian-b2-precision-v1")
D_BUILD = Path("outputs/gaussian-b2-independent-v1/native-frozen")


def load_baseline(baseline):
    expected = json.loads(CONFIG.read_text())["baseline_generated_sources"]
    source = baseline / "source"
    actual = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("Generated A/D sources differ from the frozen study")
    return {name: (source / name).read_text() for name in expected}


def compile_study(baseline, output, tag, transform, headers):
    if (output / "identity.json").exists():
        raise ValueError("Choose a new output to retain the measured binary")
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    original = load_baseline(baseline)
    generated = transform(original.copy())
    name = f"_b2_precision_{tag}"
    for filename in ("CMakeLists.txt", "bindings.cpp"):
        generated[filename] = generated[filename].replace("_b2_independent_study", name)
    for filename in headers:
        generated[filename] = (HERE / filename).read_text()
    patch = []
    for filename, value in generated.items():
        (source / filename).write_text(value)
        patch.extend(difflib.unified_diff(original.get(filename, "").splitlines(True), value.splitlines(True),
                                         fromfile=f"baseline/{filename}", tofile=f"{tag}/{filename}"))
    (output / "generated.patch").write_text("".join(patch))
    build = output / "build"
    pybind = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"], text=True).strip()
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), f"-DPython_EXECUTABLE={sys.executable}",
                    f"-Dpybind11_DIR={pybind}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
    (binary,) = build.glob(name + "*.so")
    atomic_json(output / "identity.json", {
        "module": name, "baseline_config_SHA256": sha(CONFIG),
        "source_SHA256": {p.name: sha(p) for p in sorted(source.iterdir())},
        "patch_SHA256": sha(output / "generated.patch"), "binary_SHA256": sha(binary),
    })


def extension(directory):
    identity = json.loads((directory / "identity.json").read_text())
    name = identity["module"]
    (binary,) = (directory / "build").glob(name + "*.so")
    if sha(binary) != identity["binary_SHA256"]:
        raise ImportError("Precision-study binary does not match its build identity")
    if name in sys.modules:
        module = sys.modules[name]
        if module.study_identity != identity:
            raise ImportError("Restart Python to switch a loaded precision-study binary")
        return module
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.study_identity = identity
    sys.modules[name] = module
    return module


def build_audit(baseline, output):
    def transform(files):
        text = files["reference.cpp"]
        text = '#include "precision_audit.hpp"\n' + text
        text = replace_once(text, "    std::vector<Line> out;",
                            "    PrecisionAuditSpan elapsed(audit_slot(&PrecisionAudit::lines, p));\n    std::vector<Line> out;")
        text = replace_once(text, """        affine(k, r, source, source, k.c, z);
        mpfr_set(line.offset[r].l.v, z.lo.v, MPFR_RNDD);
        mpfr_set(line.offset[r].h.v, z.hi.v, MPFR_RNDU);
        affine(k, r, receiver, receiver, source, z);
        mpfr_set(line.slope[r].l.v, z.lo.v, MPFR_RNDD);
        mpfr_set(line.slope[r].h.v, z.hi.v, MPFR_RNDU);""", """        {
          PrecisionAuditSpan elapsed(audit_slot(&PrecisionAudit::offset, p));
          affine(k, r, source, source, k.c, z);
          mpfr_set(line.offset[r].l.v, z.lo.v, MPFR_RNDD);
          mpfr_set(line.offset[r].h.v, z.hi.v, MPFR_RNDU);
        }
        {
          PrecisionAuditSpan elapsed(audit_slot(&PrecisionAudit::slope, p));
          affine(k, r, receiver, receiver, source, z);
          mpfr_set(line.slope[r].l.v, z.lo.v, MPFR_RNDD);
          mpfr_set(line.slope[r].h.v, z.hi.v, MPFR_RNDU);
        }""")
        text = replace_once(text, "      out.push_back(line);",
                            """      {
        PrecisionAuditSpan elapsed(audit_slot(&PrecisionAudit::copy, p));
        out.push_back(line);
      }
      if (active_precision_audit) ++active_precision_audit->line_kernels[p == 256];""")
        for rounding in ("RNDD", "RNDU"):
            text = replace_once(text, f"  mpfr_exp(z.x.v, z.x.v, MPFR_{rounding});",
                                f"  audit_exp(z.x.v, MPFR_{rounding});")
        files["reference.cpp"] = text

        text = files["local_search.hpp"]
        text = replace_once(text, '  s.add("reference_ray_queries");',
                            '  PrecisionAudit audit(s);\n  s.add("reference_ray_queries");')
        text = replace_once(text, '  auto line = impl->lines(source, receiver, s, precision);',
                            '  auto line = impl->lines(source, receiver, s, precision);\n  std::vector<Line> low_probe;')
        text = replace_once(text, "    low = false;",
                            "    if (precision_audit_probe) low_probe = std::move(line);\n    low = false;")
        text = replace_once(text, """    if (state >= 0) {
      queue.pop_front();""", """    ++audit.interval_decisions[!low][state + 1];
    if (!low && !low_probe.empty()) {
      Stats scratch_stats;
      active_precision_audit = nullptr;
      int answer = impl->natural(low_probe, n.lo, n.hi, scratch_stats,
                                 o.tail_bounds, study.filter_bits);
      active_precision_audit = &audit;
      ++audit.counterfactual[state + 1][answer + 1];
    }
    if (state >= 0) {
      queue.pop_front();""")
        text = replace_once(text, "      if (low) {\n        escalate();",
                            "      if (low) {\n        ++audit.revisited_at_depth_limit;\n        escalate();")
        text = replace_once(text, "    if (!prepared)\n      prepare(line, kernels, stats);", """    if (!prepared) {
      PrecisionAuditSpan elapsed(active_precision_audit ? &active_precision_audit->derivative : nullptr);
      prepare(line, kernels, stats);
    }""")
        files["local_search.hpp"] = text
        return files

    compile_study(baseline, output, "audit", transform, ["precision_audit.hpp"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=D_BUILD)
    p.add_argument("--output", type=Path, default=OUT / "audit-native")
    args = p.parse_args()
    build_audit(args.baseline, args.output)
