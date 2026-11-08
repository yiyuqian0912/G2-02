// Included in primary.cpp, in b2 after the primary numerical helpers.
namespace {
double density_endpoint(Exp &exp, double q, bool upper,
                        std::uint64_t &endpoints) {
  q = std::max(0., q);
  if (q >= 128) {
    if (!upper)
      return 0;
    // e > 2. The dyadic upper tail remains positive even at underflow.
    const int exponent = int(std::min(1074., q * .5));
    return std::ldexp(1., -exponent);
  }
  const auto rounding = upper ? MPFR_RNDU : MPFR_RNDD;
  mpfr_set_d(exp.x.v, q, MPFR_RNDN);
  mpfr_div_2ui(exp.x.v, exp.x.v, 1, MPFR_RNDN);
  mpfr_neg(exp.x.v, exp.x.v, MPFR_RNDN);
  mpfr_exp(exp.x.v, exp.x.v, rounding);
  ++endpoints;
  return mpfr_get_d(exp.x.v, rounding);
}

bool primary_leaf_free(const PrimaryCapture::Leaf &leaf,
                       const std::vector<PrimaryCapture::Line> &lines,
                       const std::vector<Kernel> &kernels, int direction,
                       double length, Exp &exp, std::uint64_t &endpoints) {
  const int axis = std::abs(direction) - 1;
  const I t(leaf.lo, leaf.hi);
  const double displacement = mul(I(leaf.hi), I(length)).hi;
  double sum = 0;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const auto &kernel = kernels[i];
    I q, factor;
    for (int r = 0; r < 3; ++r) {
      I b = kernel.inv[3 * r + axis];
      if (direction < 0)
        b = neg(b);
      const I z = add(add(lines[i].offset[r], mul(lines[i].slope[r], t)),
                      mul(b, I(0, displacement)));
      q = add(q, square(z));
      factor = sub(factor, mul(b, z));
    }
    if (!std::isfinite(q.lo) || !std::isfinite(q.hi) ||
        !std::isfinite(factor.hi))
      return false;
    const bool upper = factor.hi >= 0;
    const double g =
        density_endpoint(exp, upper ? q.lo : q.hi, upper, endpoints);
    const double signed_weight = step(kernel.w * factor.hi, true);
    sum = step(sum + step(signed_weight * g, true), true);
    if (!std::isfinite(sum))
      return false;
  }
  return sum <= 0 || step(displacement * sum, true) < leaf.margin;
}
} // namespace

Decision Primary::neighborhood_ray(P source, P receiver, int grid,
                                   const Options &o, Stats &stats,
                                   const AxialPlan &plan,
                                   AxialResult &result) const {
  PrimaryCapture proof;
  proof.retain_line = true;
  auto d = ray(source, receiver, o, stats, &proof);
  stats.add("primary_capture_rays");
  stats.add("primary_capture_leaves", proof.size);
  stats.maximum("primary_capture_line_payload_bytes",
                proof.lines.capacity() * sizeof(PrimaryCapture::Line));
  if (d.status != 0 || !proof.complete || !proof.usable || proof.lines.empty())
    return d;
  std::size_t limiting = 0;
  for (std::size_t i = 1; i < proof.size; ++i)
    if (proof.leaves[i].margin / proof.leaves[i].hi <
        proof.leaves[limiting].margin / proof.leaves[limiting].hi)
      limiting = i;
  Exp exp;
  std::uint64_t endpoints = 0, tests = 0;
  result = certify_axial(
      plan, proof.size, limiting,
      [&](std::size_t i, int direction, int distance) {
        ++tests;
        return primary_leaf_free(proof.leaves[i], proof.lines, impl->kernels,
                                 direction, double(distance) / grid, exp,
                                 endpoints);
      },
      stats);
  stats.add("primary_local_leaf_tests", tests);
  stats.add("primary_local_exp_endpoints", endpoints);
  return d;
}
