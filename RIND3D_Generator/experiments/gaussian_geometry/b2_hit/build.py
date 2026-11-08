"""Build witness capture and two fixed-witness certificates over frozen P_J."""
import argparse
import hashlib
from pathlib import Path

from ..b2_miss.bench import read
from ..b2_precision.build import D_BUILD, compile_study, replace_once
from ..b2_primary_miss.propagation import transform as primary_transform
from .freeze import NATIVE, OUT

HERE = Path(__file__).resolve().parent


def frozen(files):
    generated = primary_transform(files)
    identity = read(NATIVE / "identity.json")
    actual = {}
    for name, source in generated.items():
        if name in ("CMakeLists.txt", "bindings.cpp"):
            source = source.replace("_b2_independent_study", identity["module"])
        actual[name] = hashlib.sha256(source.encode()).hexdigest()
    if actual != identity["source_SHA256"]:
        raise ValueError("Frozen P_J native sources differ")
    return generated


def transform(files):
    files = frozen(files)
    for name in ("hit_witness.hpp", "hit_certificate.hpp", "tile_hit_audit.hpp"):
        files[name] = (HERE / name).read_text()
    name = "common.hpp"
    text = replace_once(files[name], "struct PrimaryCapture;",
                        "struct HitWitness;\nstruct HitAudit;\nstruct PrimaryCapture;")
    files[name] = replace_once(text, "               PrimaryRayWork * = nullptr) const;",
        """               PrimaryRayWork * = nullptr, HitWitness * = nullptr) const;
  HitAudit audit_hit(P, P, int, const Options &,
                     const std::vector<std::array<int, 2>> &, bool) const;""")

    name = "primary.cpp"
    text = '#include "hit_witness.hpp"\n' + files[name]
    text = replace_once(text, "Stats &s, double *margin = nullptr) const {",
                        "Stats &s, double *margin = nullptr, double *hit_lower = nullptr) const {")
    text = replace_once(text, """    if (mpfr_cmp(l.v, th.v) >= 0)
      return 1;""", """    if (mpfr_cmp(l.v, th.v) >= 0) {
      if (hit_lower)
        *hit_lower = mpfr_get_d(l.v, MPFR_RNDD);
      return 1;
    }""")
    text = replace_once(text, "PrimaryCapture *proof, PrimaryRayWork *audit) const {",
                        "PrimaryCapture *proof, PrimaryRayWork *audit, HitWitness *hit) const {")
    text = replace_once(text, """    auto q = qrange(exact_coefficients(e.kernels[0], source, receiver), 0, 1);
    return {q.first <= -2 * Q(e.input.threshold) ? 1 : 0,
            "exact_closed_quadratic"};""", """    const auto poly = exact_coefficients(e.kernels[0], source, receiver);
    const auto q = qrange(poly, 0, 1);
    const Q boundary = -2 * Q(e.input.threshold);
    if (hit && q.first <= boundary) {
      Q exact_t = 0;
      if (poly.a != 0)
        exact_t = std::clamp(Q(-poly.b / (2 * poly.a)), Q(0), Q(1));
      const double t = exact_t.get_d();
      hit->record(q.first == boundary ? "analytic_contact" : "analytic_strict",
                  Q(t) == exact_t ? t : NAN, 0, e.tau);
    }
    return {q.first <= boundary ? 1 : 0, "exact_closed_quadratic"};""")
    text = replace_once(text, "auto fallback = [&](Q lo, Q hi, double *margin = nullptr) {",
                        "auto fallback = [&](Q lo, Q hi, double *margin = nullptr, double *lower = nullptr) {")
    text = replace_once(text, "e.exact_sum(exact, lo, hi, o.precision, s, margin);",
                        "e.exact_sum(exact, lo, hi, o.precision, s, margin, lower);")
    text = replace_once(text, """        if (e.kernels[i].inner.lo >= 0 &&
            at(c[i], I(peak(c[i]))).hi <= e.kernels[i].inner.lo) {
          s.add("inner_ray_hit");
          return {1, "contained_ellipsoid_witness", nodes, depth};
        }""", """        if (e.kernels[i].inner.lo >= 0) {
          const double t = peak(c[i]);
          const double upper_q = at(c[i], I(t)).hi;
          if (upper_q <= e.kernels[i].inner.lo) {
            if (hit) {
              // w*exp(-q/2) >= tau*exp((inner-q)/2) >= tau*(1+(inner-q)/2).
              const I slack = sub(I(e.kernels[i].inner.lo), I(upper_q));
              const I bound = mul(I(e.tau.lo), add(I(1), mul(I(.5), slack)));
              hit->record("inner_ellipsoid", t, bound.lo, e.tau);
            }
            s.add("inner_ray_hit");
            return {1, "contained_ellipsoid_witness", nodes, depth};
          }
        }""")
    text = replace_once(text, """    if (state == 1)
      return {1, "entire_parameter_interval_solid", nodes, depth};""",
        """    if (state == 1) {
      if (hit) hit->record("interval", (node.l + node.r)/2, value.lo, e.tau);
      return {1, "entire_parameter_interval_solid", nodes, depth};
    }""")
    text = replace_once(text, """      double margin;
      int exact_state = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr);""",
        """      double margin, lower;
      int exact_state = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr,
                                 hit ? &lower : nullptr);""")
    text = replace_once(text, """      if (exact_state == 1)
        return {1, "early_exact_interval_solid", nodes, depth};""",
        """      if (exact_state == 1) {
        if (hit) hit->record("exact_interval", (node.l + node.r)/2, lower, e.tau);
        return {1, "early_exact_interval_solid", nodes, depth};
      }""")
    text = replace_once(text, """      if (st == 1)
        return {1, "certified_parameter_witness", nodes, depth};""",
        """      if (st == 1) {
        if (hit) hit->record("point", t, v.lo, e.tau);
        return {1, "certified_parameter_witness", nodes, depth};
      }""")
    text = replace_once(text, """      if (st < 0 && o.fallback && fallback(Q(t), Q(t)) == 1)
        return {1, "high_precision_point_witness", nodes, depth};""",
        """      double lower;
      if (st < 0 && o.fallback && fallback(Q(t), Q(t), nullptr, hit ? &lower : nullptr) == 1) {
        if (hit) hit->record("exact_point", t, lower, e.tau);
        return {1, "high_precision_point_witness", nodes, depth};
      }""")
    text = replace_once(text, """        double margin;
        int r = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr);""",
        """        double margin, lower;
        int r = fallback(Q(node.l), Q(node.r), proof ? &margin : nullptr,
                         hit ? &lower : nullptr);""")
    text = replace_once(text, """        if (r == 1)
          return {1, "high_precision_interval", nodes, depth};""",
        """        if (r == 1) {
          if (hit) hit->record("exact_interval", (node.l + node.r)/2, lower, e.tau);
          return {1, "high_precision_interval", nodes, depth};
        }""")
    files[name] = replace_once(text, '#include "primary_local.hpp"',
                               '#include "primary_local.hpp"\n#include "hit_certificate.hpp"')
    name = "regions.hpp"
    text = '#include "hit_witness.hpp"\n' + files[name]
    files[name] = replace_once(text, "  std::pair<Decision, AxialResult> primary_probe(P, int, int, Stats &);",
        """  std::pair<Decision, AxialResult> primary_probe(P, int, int, Stats &);
  HitAudit hit_audit(P, const std::vector<std::array<int, 2>> &, bool, bool);""")
    name = "regions.cpp"
    files[name] = replace_once(files[name], '#include "tile_session.hpp"',
        '#include "tile_session.hpp"\n#include "tile_hit_audit.hpp"')
    name = "bindings.cpp"
    files[name] = replace_once(files[name], '      .def("preparation_work", &TileSession::preparation_work)',
        '''      .def("preparation_work", &TileSession::preparation_work)
      .def("hit_audit", [](TileSession &session, P receiver,
                            const std::vector<std::array<int, 2>> &requests,
                            bool controls, bool capture_first) {
        HitAudit a;
        {
          py::gil_scoped_release release;
          a = session.hit_audit(receiver, requests, controls, capture_first);
        }
        py::dict row;
        row["status"] = a.decision.status;
        row["reason"] = a.decision.reason;
        row["nodes"] = a.decision.nodes;
        row["kind"] = a.witness.kind;
        row["t"] = std::isfinite(a.witness.t) ? py::object(py::float_(a.witness.t)) : py::object(py::none());
        row["lower"] = a.witness.lower;
        row["margin"] = a.witness.margin;
        row["tau_upper"] = a.tau_upper;
        row["scalar_seconds"] = a.scalar_seconds;
        row["capture_seconds"] = a.capture_seconds;
        py::list trials;
        for (const auto &r : a.trials) {
          py::dict v;
          v["direction"] = r.direction;
          v["steps"] = r.steps;
          v["gradient"] = r.gradient;
          v["direct"] = r.direct;
          v["gradient_seconds"] = r.gradient_seconds;
          v["direct_seconds"] = r.direct_seconds;
          v["gradient_exp"] = r.gradient_exp;
          v["direct_exp"] = r.direct_exp;
          if (r.controls) {
            v["cover"] = r.cover;
            v["proxy"] = r.proxy;
            v["section"] = r.section;
            v["cover_seconds"] = r.cover_seconds;
            v["proxy_seconds"] = r.proxy_seconds;
            v["section_seconds"] = r.section_seconds;
          }
          trials.append(v);
        }
        row["trials"] = trials;
        return row;
      }, py::arg("receiver"), py::arg("requests"),
         py::arg("controls") = false, py::arg("capture_first") = true)''')
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "audit-native")
    args = parser.parse_args()
    compile_study(D_BUILD, args.output, "hit_audit", transform, [])
