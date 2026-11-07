"""Build the interval-local precision and scoped source-preparation variants."""
import argparse
from pathlib import Path

from .build import D_BUILD, OUT, compile_study, replace_once


def transform(files):
    name = "common.hpp"
    text = files[name]
    start = text.index("class Independent {")
    files[name] = text[:start] + replace_once(text[start:],
        "  Decision study_ray(P, P, const Options &, Stats &, const RayStudy &) const;",
        """  Decision study_ray(P, P, const Options &, Stats &, const RayStudy &) const;
  struct PreparedOffsets;
  std::shared_ptr<const PreparedOffsets> prepare_offsets(P, int, Stats &) const;
  Decision precision_ray(P, P, const Options &, Stats &, const RayStudy &,
                         const PreparedOffsets *) const;""")

    name = "reference.cpp"
    text = files[name]
    text = replace_once(text, "void accumulate(Scratch &z, double weight, Stats &s, bool tails) {",
        "void accumulate(Scratch &z, double weight, Stats &s, bool tails, std::uint64_t *exp_count = nullptr) {")
    text = replace_once(text, '  s.add("reference_exp_endpoints", 2);',
        '  if (exp_count) *exp_count += 2;\n  s.add("reference_exp_endpoints", 2);')
    text = replace_once(text, "              Bound *sum = nullptr) const {",
        "              Bound *sum = nullptr, std::uint64_t *exp_count = nullptr) const {")
    text = replace_once(text, "      accumulate(z, kernels[i].w, s, tails);\n      if (terms)",
        "      accumulate(z, kernels[i].w, s, tails, exp_count);\n      if (terms)")
    files[name] = text + '\n#include "precision_search.hpp"\n'

    name = "local_search.hpp"
    text = files[name]
    text = replace_once(text, "method < 0 || method > 3", "method < 0 || method > 5")
    text = replace_once(text, "      : interval_terms(kernels), midpoint_terms(kernels) {}",
        """      : interval_terms(kernels), midpoint_terms(kernels) {}

  LocalTaylor(std::size_t kernels, int term_precision) : midpoint_terms(kernels) {
    interval_terms.reserve(kernels);
    for (std::size_t i = 0; i < kernels; ++i)
      interval_terms.emplace_back(term_precision);
  }""")
    files[name] = text

    name = "regions.cpp"
    text = files[name]
    text = replace_once(text, "const RayStudy &study) const {",
        "const RayStudy &study, const Independent::PreparedOffsets *cached = nullptr) const {")
    text = replace_once(text, "    return b->study_ray(source, x, o, s, study);",
        "    if (study.method >= 4) return b->precision_ray(source, x, o, s, study, cached);\n"
        "    return b->study_ray(source, x, o, s, study);")
    text = replace_once(text, "  Cover tree;\n  bool usecover", """  std::shared_ptr<const Independent::PreparedOffsets> prepared_offsets;
  if (p->independent && cfg.study.method == 5)
    prepared_offsets = p->b->prepare_offsets(source, cfg.study.filter_bits, f.stats);
  Cover tree;
  bool usecover""")
    text = replace_once(text, "p->ray(source, v, o, f.stats, cfg.study)",
        "p->ray(source, v, o, f.stats, cfg.study, prepared_offsets.get())")
    files[name] = text
    return files


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=D_BUILD)
    p.add_argument("--output", type=Path, default=OUT / "native")
    a = p.parse_args()
    compile_study(a.baseline, a.output, "local", transform, ["precision_search.hpp"])
