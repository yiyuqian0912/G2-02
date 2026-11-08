"""Generate an isolated native build from the fingerprinted B2 MISS study."""
import argparse
import hashlib
import json
from pathlib import Path

from ..b2_miss.bench import OUT as MISS_OUT
from ..b2_miss.build import transform as miss_transform
from ..b2_precision.build import D_BUILD, compile_study, replace_once

HERE = Path(__file__).resolve().parent
OUT = Path("outputs/gaussian-b2-primary-miss-v1")
FROZEN_BUILD = MISS_OUT / "propagation-native"


def frozen(files):
    generated = miss_transform(files)
    identity = json.loads((FROZEN_BUILD / "identity.json").read_text())
    actual = {}
    for name, text in generated.items():
        if name in ("CMakeLists.txt", "bindings.cpp"):
            text = text.replace("_b2_independent_study", identity["module"])
        actual[name] = hashlib.sha256(text.encode()).hexdigest()
    if actual != identity["source_SHA256"]:
        raise ValueError("The previous frozen B2 source identity differs")
    return generated


def capture_hooks(files):
    name = "common.hpp"
    text = replace_once(files[name], "  int miss_mode = 0;",
                        "  int miss_mode = 0;\n  int primary_audit = 0;")
    text = replace_once(text, "class Primary {",
                        "struct PrimaryCapture;\nstruct PrimaryRayWork;\nclass Primary {")
    start = text.index("class Primary {")
    end = text.index("struct MissProof;", start)
    files[name] = text[:start] + replace_once(text[start:end],
        "  Decision ray(P, P, const Options &, Stats &) const;",
        """  Decision ray(P, P, const Options &, Stats &, PrimaryCapture * = nullptr,
               PrimaryRayWork * = nullptr) const;""") + text[end:]

    name = "interval.hpp"
    text = replace_once(files[name], "struct Exp {", """using ExpProbe = void (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
inline thread_local ExpProbe primary_exp_probe = nullptr;
inline void primary_exp(mpfr_ptr out, mpfr_srcptr in, mpfr_rnd_t round) {
  if (primary_exp_probe)
    primary_exp_probe(out, in, round);
  else
    mpfr_exp(out, in, round);
}
struct Exp {""")
    files[name] = text.replace("mpfr_exp(x.v, x.v, MPFR_", "primary_exp(x.v, x.v, MPFR_")

    name = "primary.cpp"
    text = '#include "primary_proof.hpp"\n' + files[name]
    text = replace_once(text, "                Stats &s) const {\n    s.add(\"high_precision_calls\");",
                        '                Stats &s, double *margin = nullptr) const {\n    s.add("high_precision_calls");')
    text = text.replace("mpfr_exp(v.v, v.v, MPFR_", "primary_exp(v.v, v.v, MPFR_")
    text = replace_once(text, """    if (mpfr_cmp(h.v, tl.v) < 0)
      return 0;""", """    if (mpfr_cmp(h.v, tl.v) < 0) {
      if (margin) {
        mpfr_sub(v.v, tl.v, h.v, MPFR_RNDD);
        *margin = mpfr_get_d(v.v, MPFR_RNDD);
      }
      return 0;
    }""")
    text = replace_once(text, "Decision Primary::ray(P source, P receiver, const Options &o, Stats &s) const {",
        """Decision Primary::ray(P source, P receiver, const Options &o, Stats &s,
                      PrimaryCapture *proof, PrimaryRayWork *audit) const {""")
    text = replace_once(text, "  auto fallback = [&](Q lo, Q hi) {", "  auto fallback = [&](Q lo, Q hi, double *margin = nullptr) {")
    text = replace_once(text, "    return e.exact_sum(exact, lo, hi, o.precision, s);",
        """    int result = e.exact_sum(exact, lo, hi, o.precision, s, margin);
    if (audit && result < 0)
      ++audit->exact_inconclusive;
    return result;""")
    text = replace_once(text, "      state = compare(e.sum(qr, lo, hi, exp, s, false), e.tau);",
                        "      value = e.sum(qr, lo, hi, exp, s, false);\n      state = compare(value, e.tau);")
    text = replace_once(text, """    if (state == 0)
      continue;
    if (state == 1)""", """    if (state == 0) {
      if (proof) proof->close(node.l, node.r, node.depth, value, e.tau);
      continue;
    }
    if (audit && state < 0)
      ++audit->inconclusive;
    if (state == 1)""")
    text = replace_once(text, """      int exact_state = fallback(Q(node.l), Q(node.r));
      if (exact_state == 0) {
        s.add("early_exact_interval_free");""", """      double margin;
      int exact_state = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr);
      if (exact_state == 0) {
        if (proof) proof->close(node.l, node.r, node.depth, margin);
        s.add("early_exact_interval_free");""")
    text = replace_once(text, """        if (add(v, error).hi < e.tau.lo) {
          s.add("taylor_interval_free");""", """        I enclosure = add(v, error);
        if (enclosure.hi < e.tau.lo) {
          if (proof) proof->close(node.l, node.r, node.depth, enclosure, e.tau);
          s.add("taylor_interval_free");""")
    text = replace_once(text, """        int r = fallback(Q(node.l), Q(node.r));
        if (r == 0)
          continue;""", """        double margin;
        int r = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr);
        if (r == 0) {
          if (proof) proof->close(node.l, node.r, node.depth, margin);
          continue;
        }""")
    text = replace_once(text, "    stack.push_back({mid, node.r, node.depth + 1});",
                        "    if (audit) ++audit->splits;\n    stack.push_back({mid, node.r, node.depth + 1});")
    files[name] = replace_once(text, '  return {0, "all_closed_intervals_certified_free", nodes, depth};',
        '  if (proof) proof->complete = proof->covered == PrimaryCapture::whole;\n'
        '  return {0, "all_closed_intervals_certified_free", nodes, depth};')

    name = "regions.cpp"
    text = '#include "primary_proof.hpp"\n#include "profile.hpp"\n' + files[name]
    text = replace_once(text, '#include "miss_cache.hpp"',
                        '#include "miss_cache.hpp"\n#include "primary_audit.hpp"')
    text = replace_once(text, "  std::optional<MissCache> miss;", """  std::optional<PrimaryTileAudit> primary_audit;
  if (cfg.study.primary_audit) {
    if (p->independent || cfg.study.primary_audit < 1 || cfg.study.primary_audit > 2)
      throw std::invalid_argument("Primary audit mode must be 1 or 2 on the primary");
    primary_audit.emplace(f.stats, cfg.study.primary_audit);
  }
  std::optional<MissCache> miss;""")
    text = replace_once(text, "            blocked = miss",
        """            blocked = primary_audit
                ? primary_audit->ray(*p->a, source, v, o).status
                : miss""")
    files[name] = replace_once(text, "done:\n  f.complete", "done:\n  if (primary_audit) primary_audit->report();\n  f.complete")
    name = "bindings.cpp"
    files[name] = replace_once(files[name], '      .def_readwrite("miss_mode", &RayStudy::miss_mode)',
        '''      .def_readwrite("miss_mode", &RayStudy::miss_mode)
      .def_readwrite("primary_audit", &RayStudy::primary_audit)''')
    for name in ("primary_proof.hpp", "primary_audit.hpp"):
        files[name] = (HERE / name).read_text()
    return files


def audit_transform(files):
    return capture_hooks(frozen(files))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    compile_study(D_BUILD, a.output, "primary_audit", audit_transform, [])
