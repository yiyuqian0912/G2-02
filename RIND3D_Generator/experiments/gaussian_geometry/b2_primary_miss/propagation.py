"""Attach backend-local axial proof tests and a shared scheduling policy."""
import argparse
from pathlib import Path

from ..b2_precision.build import D_BUILD, compile_study, replace_once
from .build import HERE, audit_transform


def transform(files):
    files = audit_transform(files)
    for name in ("axial_policy.hpp", "primary_local.hpp", "adaptive_cache.hpp"):
        files[name] = (HERE / name).read_text()

    name = "common.hpp"
    text = replace_once(files[name], "struct PrimaryCapture;",
                        "struct AxialPlan;\nstruct AxialResult;\nstruct PrimaryCapture;")
    text = replace_once(text, "               PrimaryRayWork * = nullptr) const;",
        """               PrimaryRayWork * = nullptr) const;
  Decision neighborhood_ray(P, P, int, const Options &, Stats &,
                            const AxialPlan &, AxialResult &) const;""")
    files[name] = replace_once(text, "  struct MissGradient;",
        """  Decision adaptive_ray(P, P, int, const Options &, Stats &, const RayStudy &,
                        const PreparedOffsets *, const AxialPlan &, AxialResult &) const;
  struct MissGradient;""")
    name = "primary_proof.hpp"
    files[name] = replace_once(files[name], "struct PrimaryCapture {",
        """struct PrimaryCapture {
  struct Line {
    std::array<I, 3> offset, slope;
  };
  std::vector<Line> lines;
  bool retain_line = false;""")

    name = "primary.cpp"
    text = '#include "axial_policy.hpp"\n' + files[name]
    text = replace_once(text, "Poly coefficients(const Kernel &k, P s, P r) {",
        """Poly coefficients(const Kernel &k, P s, P r,
                  PrimaryCapture::Line *line = nullptr) {""")
    text = replace_once(text, "    a = add(a, square(v));\n    b = add(b, mul(I(2), mul(u, v)));",
        """    if (line) {
      line->offset[j] = u;
      line->slope[j] = v;
    }
    a = add(a, square(v));
    b = add(b, mul(I(2), mul(u, v)));""")
    text = replace_once(text, """    v.reserve(e.kernels.size());
    for (auto &k : e.kernels) {
      v.push_back(coefficients(k, source, receiver));""", """    v.reserve(e.kernels.size());
    const bool retain_line = proof && proof->retain_line;
    if (retain_line)
      proof->lines.resize(e.kernels.size());
    for (auto &k : e.kernels) {
      v.push_back(coefficients(k, source, receiver,
                               retain_line ? &proof->lines[v.size()] : nullptr));""")
    files[name] = replace_once(text, "std::pair<int, int> Primary::certificate(",
        '#include "primary_local.hpp"\nstd::pair<int, int> Primary::certificate(')

    name = "local_search.hpp"
    files[name] = replace_once(files[name], "miss_mode > 3", "miss_mode > 4")
    name = "reference.cpp"
    files[name] = '#include "axial_policy.hpp"\n' + files[name]
    name = "miss_proof.hpp"
    text = replace_once(files[name], "  bool partition_truncated = false;",
        """  const AxialPlan *adaptive = nullptr;
  AxialResult *adaptive_result = nullptr;
  bool partition_truncated = false;""")
    text = replace_once(text, "    if (!directions.empty()) {",
                        "    if (!directions.empty() || adaptive) {")
    files[name] = replace_once(text, "struct Independent::MissGradient {",
        """Decision Independent::adaptive_ray(
    P source, P receiver, int grid, const Options &o, Stats &stats,
    const RayStudy &study, const PreparedOffsets *offsets,
    const AxialPlan &plan, AxialResult &result) const {
  MissProof proof;
  proof.grid = grid;
  proof.adaptive = &plan;
  proof.adaptive_result = &result;
  proof.partition.reserve(16);
  proof.sample_capture = (stats.n["miss_capture_rays"] & 255) == 0;
  stats.add("miss_capture_rays");
  auto d = precision_ray(source, receiver, o, stats, study, offsets, &proof);
  stats.add("miss_captured_leaves", proof.leaves);
  stats.maximum("miss_partition_payload_bytes",
                proof.partition.capacity() * sizeof(MissLeaf));
  if (proof.sample_capture) {
    stats.add("miss_capture_sample_rays");
    stats.add("miss_capture_sample_leaves", proof.leaves);
    stats.add("miss_capture_sample_ns", proof.capture_ns);
  }
  return d;
}

struct Independent::MissGradient {""")
    name = "miss_local.hpp"
    text = replace_once(files[name], "if (proof.directions.empty() ||",
                        "if ((!proof.adaptive && proof.directions.empty()) ||")
    files[name] = replace_once(text, "  const double step = double(proof.step) / proof.grid;",
        """  if (proof.adaptive) {
    *proof.adaptive_result = certify_axial(
        *proof.adaptive, proof.partition.size(), proof.limiting_leaf,
        [&](std::size_t i, int direction, int step) {
          ++calls;
          return local_leaf_free(proof.partition[i], line, kernels, direction,
                                 double(step) / proof.grid, work, exps);
        }, stats);
    stats.add("miss_local_leaf_tests", calls);
    stats.add("miss_local_exp_endpoints", exps);
    stats.add("miss_local_construction_ns", std::uint64_t(elapsed(begin) * 1e9));
    return;
  }
  const double step = double(proof.step) / proof.grid;""")

    files["miss_cache.hpp"] = extend_cache(files["miss_cache.hpp"])
    name = "tile_session.hpp"
    text = replace_once(files[name],
        """    if (!engine.p->independent || settings.study.method != 5 ||
        settings.study.stationary || !reuse_cover)
      throw std::invalid_argument("Propagation requires a frozen independent F tile session");""",
        """    if (!reuse_cover || settings.study.primary_audit ||
        (engine.p->independent && (settings.study.method != 5 || settings.study.stationary)) ||
        (!engine.p->independent && settings.study.miss_mode < 2))
      throw std::invalid_argument("Propagation requires a compatible source-bound tile session");""")
    files[name] = text + """
std::pair<Decision, AxialResult> TileSession::primary_probe(
    P receiver, int direction, int step, Stats &stats) {
  std::lock_guard guard(p->lock);
  require_environment();
  if (p->engine.p->independent || direction == 0 || direction < -3 ||
      direction > 3 || (step != 1 && step != 4) || (step == 4 && direction > 0))
    throw std::invalid_argument("Invalid primary axial diagnostic");
  for (double x : receiver) {
    const double i = x * p->grid - .5;
    if (!std::isfinite(i) || i < 0 || i >= p->grid || i != std::floor(i))
      throw std::invalid_argument("Primary probe requires a voxel center");
  }
  const double target = receiver[std::abs(direction)-1] +
                        (direction > 0 ? step : -step) / double(p->grid);
  if (!(target > 0 && target < 1))
    throw std::invalid_argument("Probe target is outside the world");
  AxialPlan plan;
  if (step == 1)
    plan.short_direction = direction;
  else
    plan.long_directions[0] = direction;
  AxialResult result;
  auto d = p->engine.p->a->neighborhood_ray(p->source, receiver, p->grid,
                                            p->options, stats, plan, result);
  return {d, result};
}
"""
    name = "regions.hpp"
    files[name] = replace_once(files[name], '#include "common.hpp"',
                               '#include "common.hpp"\n#include "axial_policy.hpp"')
    files[name] = replace_once(files[name], "  std::map<std::string, double> info() const;",
        """  std::map<std::string, double> info() const;
  std::pair<Decision, AxialResult> primary_probe(P, int, int, Stats &);""")
    name = "regions.cpp"
    files[name] = replace_once(files[name], """            blocked = primary_audit
                ? primary_audit->ray(*p->a, source, v, o).status
                : miss
                ? miss->ray(*p->b, source, v, grid, {x, yy, z}, o, cfg.study,
                            prepared_offsets.get(), session->p->miss_gradient.get(),
                            result.propagated_seed)
                : p->ray(source, v, o, f.stats, cfg.study, prepared_offsets.get()).status;""",
        """            if (primary_audit)
              blocked = primary_audit->ray(*p->a, source, v, o).status;
            else if (miss && !p->independent)
              blocked = miss->primary(*p->a, source, v, grid, {x, yy, z}, o,
                                      result.propagated_seed);
            else if (miss && cfg.study.miss_mode == 4)
              blocked = miss->adaptive(*p->b, source, v, grid, {x, yy, z}, o, cfg.study,
                                       prepared_offsets.get(), result.propagated_seed);
            else if (miss)
              blocked = miss->ray(*p->b, source, v, grid, {x, yy, z}, o, cfg.study,
                                  prepared_offsets.get(), session->p->miss_gradient.get(),
                                  result.propagated_seed);
            else
              blocked = p->ray(source, v, o, f.stats, cfg.study, prepared_offsets.get()).status;""")
    name = "bindings.cpp"
    files[name] = replace_once(files[name], '      .def("preparation_work", &TileSession::preparation_work)',
        '''      .def("preparation_work", &TileSession::preparation_work)
      .def("primary_probe", [](TileSession &session, P receiver, int direction, int step) {
        Stats work;
        auto [d, result] = session.primary_probe(receiver, direction, step, work);
        py::dict row;
        row["status"] = d.status;
        row["reason"] = d.reason;
        row["nodes"] = d.nodes;
        row["certified_step"] = result.step;
        row["work"] = work.n;
        return row;
      })''')
    return files


def extend_cache(text):
    text = replace_once(text, "using I3 = std::array<int, 3>;",
                        'using I3 = std::array<int, 3>;\n#include "adaptive_cache.hpp"')
    text = replace_once(text, "  std::vector<int> seeds;",
                        "  std::vector<int> seeds;\n  std::optional<AxialHistory> history;")
    text = replace_once(text, "public:\n  MissCache", """  AxialPlan plan(I3 xyz) {
    constexpr std::array<int, 6> order{-2, -3, -1, 2, 3, 1};
    AxialPlan request;
    if (mode != 3)
      for (int d : order)
        if (new_targets(xyz, d, 1)) {
          request.short_direction = d;
          break;
        }
    if (mode == 2)
      return request;
    std::array<std::pair<int, int>, 3> ranked{};
    for (int i = 0; i < 3; ++i) {
      const int d = order[i], count = new_targets(xyz, d, 4);
      if (count >= 2) {
        if (history && !history->permit(xyz, d))
          work.add("axial_long_dispatch_suppressed");
        else
          ranked[i] = {count, d};
      }
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](auto a, auto b) { return a.first > b.first; });
    for (int i = 0; i < 2; ++i)
      request.long_directions[i] = ranked[i].second;
    return request;
  }

public:
  MissCache""")
    text = replace_once(text, "    seeds.assign(size, empty);",
        """    seeds.assign(size, empty);
    if (mode == 4)
      history.emplace(side, work);""")
    start = text.index("  int ray(const Independent")
    setup_end = text.index("    start = Clock::now();\n    const auto dirs", start)
    marks = text.index("    if (!coverage.empty()) {", setup_end)
    decl_end = text.index("    auto start = Clock::now();", start)
    head = text[start:decl_end]
    lookup = text[decl_end:setup_end]
    old_solve = text[setup_end:marks].replace(
        "    start = Clock::now();", "    auto start = Clock::now();", 1)
    old_solve = old_solve.replace("    std::vector<I3> coverage;\n", "")
    tail = text[marks:]
    text = text[:start] + """private:
  template <class Render>
  int cached(I3 xyz, std::vector<int> &trace, Render render) {
""" + lookup + """    std::vector<I3> coverage;
    int decision = render(coverage);
""" + tail
    text = replace_once(text, "    return decision.status;", "    return decision;")
    text = replace_once(text, "};\n} // namespace", """  template <class Solve>
  int axial(I3 xyz, std::vector<int> &trace, Solve solve) {
    return cached(xyz, trace, [&](std::vector<I3> &coverage) {
      const auto start = Clock::now();
      const auto request = plan(xyz);
      work.add("miss_dispatch_ns", std::uint64_t(elapsed(start) * 1e9));
      AxialResult result;
      auto d = solve(request, result);
      if (history)
        history->observe(xyz, result);
      if (result.step) {
        for (int j = 1; j <= result.step; ++j) {
          I3 offset{};
          offset[std::abs(result.direction)-1] = result.direction > 0 ? j : -j;
          coverage.push_back(offset);
        }
      }
      return d.status;
    });
  }

public:
  int primary(const Primary &backend, P source, P receiver, int grid, I3 xyz,
              const Options &o, std::vector<int> &trace) {
    return axial(xyz, trace, [&](const AxialPlan &plan, AxialResult &result) {
      if (plan.empty())
        return backend.ray(source, receiver, o, work);
      return backend.neighborhood_ray(source, receiver, grid, o, work, plan, result);
    });
  }

  int adaptive(const Independent &backend, P source, P receiver, int grid, I3 xyz,
               const Options &o, const RayStudy &study,
               const Independent::PreparedOffsets *offsets, std::vector<int> &trace) {
    return axial(xyz, trace, [&](const AxialPlan &plan, AxialResult &result) {
      if (plan.empty())
        return backend.precision_ray(source, receiver, o, work, study, offsets);
      return backend.adaptive_ray(source, receiver, grid, o, work, study, offsets,
                                  plan, result);
    });
  }

""" + head + """    return cached(xyz, trace, [&](std::vector<I3> &coverage) {
""" + "".join("  " + line if line.strip() else line for line in old_solve.splitlines(keepends=True)) + """      return decision.status;
    });
  }
};
} // namespace""")
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    compile_study(D_BUILD, args.output, "primary_local", transform, [])
