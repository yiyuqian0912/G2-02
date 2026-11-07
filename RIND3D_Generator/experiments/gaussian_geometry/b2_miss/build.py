"""Build isolated hooks into the fingerprinted F and tile-session sources."""
import argparse
from pathlib import Path

from ..b2_precision.build import D_BUILD, compile_study, replace_once
from ..b2_precision.tile_build import transform as tile_transform
from .bench import OUT

HERE = Path(__file__).resolve().parent
PRECISION = HERE.parent / "b2_precision"


def transform(files):
    files = tile_transform(files)
    name = "common.hpp"
    files[name] = replace_once(files[name], "class Independent {", "struct MissProof;\nclass Independent {")
    files[name] = replace_once(files[name], "struct RayStudy {", "struct RayStudy {\n  int miss_mode = 0;")
    files[name] = replace_once(files[name], "                         const PreparedOffsets *) const;",
        """                         const PreparedOffsets *, MissProof * = nullptr) const;
  struct MissGradient;
  std::shared_ptr<const MissGradient> prepare_miss_gradient(Stats &) const;
  Decision neighborhood_ray(P, P, int, const Options &, Stats &, const RayStudy &,
                            const PreparedOffsets *, const MissGradient *,
                            const std::vector<int> &, int, int,
                            std::vector<std::array<int, 3>> &) const;
  std::pair<Decision, std::map<std::string, double>>
  miss_probe(P, P, int, const Options &, Stats &, const RayStudy &,
             const PreparedOffsets *, const MissGradient &,
             const std::vector<int> &, int) const;""")

    name = "reference.cpp"
    files[name] = replace_once(files[name], '#include "precision_search.hpp"',
        '#include "miss_proof.hpp"\n#include "precision_search.hpp"')
    for header in ("miss_proof.hpp", "miss_local.hpp", "miss_cache.hpp"):
        files[header] = (HERE / header).read_text()

    name = "local_search.hpp"
    files[name] = replace_once(files[name], "if (method < 0 || method > 5",
                              "if (miss_mode < 0 || miss_mode > 3 || method < 0 || method > 5")
    files[name] = replace_once(files[name], "bool stationary, Stats &stats) {",
                              "bool stationary, Stats &stats, mpfr_ptr accepted = nullptr) {")
    for key in ("study_global_taylor_free", "study_concave_free", "study_local_taylor_free"):
        needle = f'stats.add("{key}");'
        files[name] = replace_once(files[name], needle,
            "if (accepted) mpfr_set(accepted, bound.v, MPFR_RNDU);\n      " + needle)

    name = "precision_search.hpp"
    text = (PRECISION / name).read_text()
    text = replace_once(text, "const PreparedOffsets *cached) const {",
                        "const PreparedOffsets *cached, MissProof *proof) const {")
    text = replace_once(text, "    covered += whole >> n.depth;",
                        "    if (proof) proof->close(n.lo, n.hi, impl->tau.l.v);\n    covered += whole >> n.depth;")
    text = replace_once(text, """    int state = impl->natural(low, n.lo, n.hi, s, o.tail_bounds, p,
                              local_node ? &local->interval_terms : nullptr,
                              nullptr, &work.low_exp);""", """    auto eval_low = [&](double lo, double hi, bool tails,
                        std::vector<Bound> *terms = nullptr) {
      int result = impl->natural(low, lo, hi, s, tails, p, terms,
                                 proof ? &proof->enclosure : nullptr, &work.low_exp);
      if (proof) proof->observe(proof->enclosure.l.v);
      return result;
    };
    int state = eval_low(n.lo, n.hi, o.tail_bounds,
                         local_node ? &local->interval_terms : nullptr);""")
    text = replace_once(text, """      return impl->natural(high.empty() ? low : high, lo, hi, s, tails, bits,
                           terms, sum, &work.high_exp);""", """      if (proof && !sum) sum = &proof->enclosure;
      int result = impl->natural(high.empty() ? low : high, lo, hi, s, tails, bits,
                                 terms, sum, &work.high_exp);
      if (proof) proof->observe(sum->l.v);
      return result;""")
    text = replace_once(text, "int proof = local->certify(", "int certification = local->certify(")
    text = replace_once(text, "if (proof == 0)", "if (certification == 0)")
    text = replace_once(text, "if (proof == 1)", "if (certification == 1)")
    text = replace_once(text, "study.stationary, s);",
                        "study.stationary, s, proof ? proof->enclosure.h.v : nullptr);")
    text = replace_once(text, """      state = impl->natural(low, mid, mid, s, o.tail_bounds, p, nullptr,
                            nullptr, &work.low_exp);
      if (state < 0 && o.tail_bounds)
        state = impl->natural(low, mid, mid, s, false, p, nullptr, nullptr,
                              &work.low_exp);""", """      state = eval_low(mid, mid, o.tail_bounds);
      if (state < 0 && o.tail_bounds)
        state = eval_low(mid, mid, false);""")
    text = replace_once(text, """    if (covered == whole)
      return {0, "independent_partition_free", nodes, depth};""", """    if (covered == whole) {
      if (proof) {
        proof->complete = true;
        certify_local(*proof, low, impl->kernels, receiver, s);
      }
      return {0, "independent_partition_free", nodes, depth};
    }""")
    files[name] = text

    name = "regions.hpp"
    files[name] = replace_once(files[name], "  std::map<std::string, double> info() const;",
        """  std::map<std::string, double> info() const;
  std::pair<Decision, std::map<std::string, double>> miss_probe(P, const std::vector<int> &, int, Stats &);""")
    files[name] = replace_once(files[name], "struct Result {",
                              "struct Result {\n  std::vector<int> propagated_seed;")
    name = "tile_session.hpp"
    text = (PRECISION / name).read_text()
    text = replace_once(text, "  Stats preparation;", """  Stats preparation;
  std::shared_ptr<const Independent::MissGradient> miss_gradient;""")
    text = replace_once(text, "  p->clearance_seconds = elapsed(start);",
        """  p->clearance_seconds = elapsed(start);
  if (settings.study.miss_mode) {
    if (!engine.p->independent || settings.study.method != 5 ||
        settings.study.stationary || !reuse_cover)
      throw std::invalid_argument("Propagation requires a frozen independent F tile session");
    if (settings.study.miss_mode == 1)
      p->miss_gradient = engine.p->b->prepare_miss_gradient(p->preparation);
  }""")
    files[name] = text + """
std::pair<Decision, std::map<std::string, double>>
TileSession::miss_probe(P receiver, const std::vector<int> &directions,
                       int step, Stats &stats) {
  std::lock_guard guard(p->lock);
  if (!p->engine.p->independent || p->settings.study.method != 5)
    throw std::invalid_argument("MISS extraction requires an independent F session");
  require_environment();
  if (!p->miss_gradient)
    p->miss_gradient = p->engine.p->b->prepare_miss_gradient(stats);
  return p->engine.p->b->miss_probe(p->source, receiver, p->grid, p->options,
                                    stats, p->settings.study, p->offsets.get(),
                                    *p->miss_gradient, directions, step);
}
"""
    name = "regions.cpp"
    text = replace_once(files[name], '#include "tile_session.hpp"',
                        '#include "tile_session.hpp"\n#include "miss_cache.hpp"')
    text = replace_once(text, "  Result result;\n  result.setup_seconds",
        """  if (cfg.study.miss_mode && (!session || n > 64))
    throw std::invalid_argument("Propagation cache requires a session and tile side <=64");
  Result result;
  result.setup_seconds""")
    text = replace_once(text, """  auto start = Clock::now();
  std::shared_ptr<const Independent::PreparedOffsets> prepared_offsets;""",
        """  auto start = Clock::now();
  std::optional<MissCache> miss;
  if (cfg.study.miss_mode)
    miss.emplace(n, cfg.study.miss_mode, f.stats, cfg.trace, result.propagated_seed);
  std::shared_ptr<const Independent::PreparedOffsets> prepared_offsets;""")
    text = replace_once(text,
        "blocked = p->ray(source, v, o, f.stats, cfg.study, prepared_offsets.get()).status;",
        """blocked = miss
                ? miss->ray(*p->b, source, v, grid, {x, yy, z}, o, cfg.study,
                            prepared_offsets.get(), session->p->miss_gradient.get(),
                            result.propagated_seed)
                : p->ray(source, v, o, f.stats, cfg.study, prepared_offsets.get()).status;""")
    files[name] = "#include <optional>\n" + text
    name = "bindings.cpp"
    files[name] = replace_once(files[name], '      .def_readwrite("method", &RayStudy::method)',
        '''      .def_readwrite("method", &RayStudy::method)
      .def_readwrite("miss_mode", &RayStudy::miss_mode)''')
    files[name] = replace_once(files[name], '  out["complete"] = f.complete;',
        '''  if (!r.propagated_seed.empty()) {
    py::array_t<int> seed({n, n, n});
    std::copy(r.propagated_seed.begin(), r.propagated_seed.end(), seed.mutable_data());
    out["propagated_seed"] = seed;
  }
  out["complete"] = f.complete;''')
    files[name] = replace_once(files[name], '      .def("preparation_work", &TileSession::preparation_work)',
        '''      .def("preparation_work", &TileSession::preparation_work)
      .def("miss_probe", [](TileSession &session, P receiver,
                            const std::vector<int> &directions, int step) {
        Stats stats;
        auto start = Clock::now();
        auto [d, data] = session.miss_probe(receiver, directions, step, stats);
        py::dict row;
        row["status"] = d.status;
        row["reason"] = d.reason;
        row["nodes"] = d.nodes;
        row["depth"] = d.depth;
        row["seconds"] = elapsed(start);
        row["bounds"] = data;
        row["work"] = stats.n;
        return row;
      }, py::arg("receiver"), py::arg("directions") = std::vector<int>{},
         py::arg("step") = 1)''')
    return files


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUT / "audit-native")
    a = p.parse_args()
    compile_study(D_BUILD, a.output, "miss", transform, [])
