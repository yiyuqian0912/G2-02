"""Build primary-only HIT marks alongside unchanged J MISS planning."""
import argparse
from pathlib import Path

from ..b2_precision.build import D_BUILD, compile_study, replace_once
from .build import HERE, transform as audit_transform
from .freeze import OUT


def transform(files):
    files = audit_transform(files)
    for name in ("hit_live.hpp", "hit_cache_ops.hpp"):
        files[name] = (HERE / name).read_text()

    name = "common.hpp"
    text = replace_once(files[name], "  int primary_audit = 0;",
                        "  int primary_audit = 0;\n  int hit_mode = 0;")
    files[name] = replace_once(text, "                            const AxialPlan &, AxialResult &) const;",
        """                            const AxialPlan &, AxialResult &,
                            HitWitness * = nullptr) const;
  bool extend_hit(P, P, int, HitWitness &, int, int, bool, std::uint64_t &) const;""")
    name = "hit_witness.hpp"
    text = files[name].replace('#include "interval.hpp"', '#include "primary_proof.hpp"\n#include <stdexcept>')
    text = replace_once(text, "namespace b2 {", """namespace b2 {
inline void require_compatible_hit_label(int existing, int certified) {
  if (existing >= 0 && existing != certified)
    throw std::logic_error("Contradictory certified primary HIT and MISS labels");
}
""")
    text = replace_once(text, "  double t = NAN, lower = 0, margin = 0;",
        """  double t = NAN, lower = 0, margin = 0, tau_upper = 0;
  std::vector<PrimaryCapture::Line> lines;""")
    files[name] = replace_once(text, "    t = parameter;", "    t = parameter;\n    tau_upper = tau.hi;")
    name = "hit_certificate.hpp"
    files[name] = replace_once(files[name],
        """      const double upper_q = std::max(q.hi, q_other.hi);
      if (!std::isfinite(upper_q))
        return false;""", """      if (!std::isfinite(q_other.hi))
        return false;
      const double upper_q = std::max(q.hi, q_other.hi);""")
    name = "primary_local.hpp"
    text = replace_once(files[name], "AxialResult &result) const {",
                        "AxialResult &result, HitWitness *hit) const {")
    files[name] = replace_once(text, "  auto d = ray(source, receiver, o, stats, &proof);",
        """  auto d = ray(source, receiver, o, stats, &proof, nullptr, hit);
  if (hit && d.status == 1 && hit->margin > 0)
    hit->lines = std::move(proof.lines);""")
    name = "primary.cpp"
    files[name] = replace_once(files[name], '#include "hit_certificate.hpp"',
                        '#include "hit_certificate.hpp"\n#include "hit_live.hpp"')

    name = "miss_cache.hpp"
    text = replace_once(files[name], "private:\n  template <class Render>",
        '#include "hit_cache_ops.hpp"\n  template <class Render>')
    text = replace_once(text, "    const int seed = seeds[k];\n    seeds[k] = visited;",
        """    const int seed = seeds[k];
    const int hit_seed = hits ? hits->at(k) : -1;
    if (hit_seed >= 0)
      require_compatible_hit_label(seed >= 0 ? 0 : -1, 1);
    seeds[k] = visited;""")
    text = replace_once(text, "    std::vector<I3> coverage;\n    int decision = render(coverage);",
        """    if (hit_seed >= 0) {
      // The old J planner advances its cooldown on this scalar visit even
      // though the independent MISS proof construction cannot occur for a HIT.
      const auto dispatch = Clock::now();
      (void)plan(xyz);
      work.add("miss_dispatch_ns", std::uint64_t(elapsed(dispatch)*1e9));
      if (!hits->trace.empty())
        hits->trace[k] = hit_seed;
      work.add("hit_cache_hits");
      return 1;
    }
    std::vector<I3> coverage;
    int decision = render(coverage);""")
    text = replace_once(text, """        auto idx = index(dest);
        if (seeds[idx] == empty) {""", """        auto idx = index(dest);
        if (hits) {
          require_compatible_hit_label(hits->at(idx) >= 0 ? 1 : -1, 0);
          require_compatible_hit_label(hits->physical(idx), 0);
        }
        if (seeds[idx] == empty) {""")
    text = replace_once(text, """      if (plan.empty())
        return backend.ray(source, receiver, o, work);
      return backend.neighborhood_ray(source, receiver, grid, o, work, plan, result);""",
        """      if (!hits) {
        if (plan.empty())
          return backend.ray(source, receiver, o, work);
        return backend.neighborhood_ray(source, receiver, grid, o, work, plan, result);
      }
      HitWitness witness;
      const auto d = plan.empty()
          ? backend.ray(source, receiver, o, work, nullptr, nullptr, &witness)
          : backend.neighborhood_ray(source, receiver, grid, o, work, plan, result, &witness);
      if (d.status == 1 && witness.margin > 0)
        try_hit(backend, source, receiver, grid, xyz, witness);
      return d;""")
    files[name] = text

    name = "regions.hpp"
    files[name] = replace_once(files[name], "  std::vector<int> propagated_seed;",
                        "  std::vector<int> propagated_seed, hit_propagated_seed;")
    name = "regions.cpp"
    text = replace_once(files[name], """  if (cfg.study.miss_mode)
    miss.emplace(n, cfg.study.miss_mode, f.stats, cfg.trace, result.propagated_seed);""",
        """  if (cfg.study.miss_mode)
    miss.emplace(n, cfg.study.miss_mode, f.stats, cfg.trace, result.propagated_seed);
  if (cfg.study.hit_mode) {
    if (!miss || p->independent || cfg.study.miss_mode != 4 || primary_audit ||
        cfg.study.hit_mode < 1 || cfg.study.hit_mode > 5)
      throw std::invalid_argument("HIT propagation requires the primary J tile session");
    miss->attach_hit(cfg.study.hit_mode, f, cfg.trace, result.hit_propagated_seed);
  }""")
    files[name] = replace_once(text, "done:\n  if (primary_audit)",
                        "done:\n  if (miss) miss->check_hit_labels();\n  if (primary_audit)")
    name = "bindings.cpp"
    text = replace_once(files[name], '  out["complete"] = f.complete;',
        '''  if (!r.hit_propagated_seed.empty()) {
    py::array_t<int> seed({n, n, n});
    std::copy(r.hit_propagated_seed.begin(), r.hit_propagated_seed.end(), seed.mutable_data());
    out["hit_propagated_seed"] = seed;
  }
  out["complete"] = f.complete;''')
    text = replace_once(text, '      .def_readwrite("primary_audit", &RayStudy::primary_audit)',
        '''      .def_readwrite("primary_audit", &RayStudy::primary_audit)
      .def_readwrite("hit_mode", &RayStudy::hit_mode)''')
    files[name] = replace_once(text, '  m.attr("ready_capable") = false;',
        '''  m.attr("ready_capable") = false;
  m.def("_require_compatible_hit_label", &require_compatible_hit_label);''')
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "live-native")
    a = parser.parse_args()
    compile_study(D_BUILD, a.output, "hit_live", transform, [])
