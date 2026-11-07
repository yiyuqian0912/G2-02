// Included in the independent reference translation unit.
namespace {
struct LocalGradientScratch {
  Scratch z{64};
  MP factor{64}, c_lo{64}, c_hi{64}, move{64}, weight{64}, total{64};
};

bool local_leaf_free(const MissLeaf &leaf, const std::vector<Line> &line,
                     const std::vector<RefKernel> &kernels, int direction,
                     double step, LocalGradientScratch &work,
                     std::uint64_t &exps) {
  auto &z = work.z;
  const int axis = std::abs(direction) - 1;
  const double signed_step = direction > 0 ? step : -step;
  mpfr_set_zero(work.total.v, 0);
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    const auto &kernel = kernels[i];
    mpfr_set_zero(z.ql.v, 0);
    mpfr_set_zero(z.qh.v, 0);
    mpfr_set_zero(work.factor.v, 0);
    for (int r = 0; r < 3; ++r) {
      const auto &l = line[i];
      mpfr_mul_d(z.lo.v, l.slope[r].l.v,
                 mpfr_sgn(l.slope[r].l.v) >= 0 ? leaf.lo : leaf.hi, MPFR_RNDD);
      mpfr_add(z.lo.v, z.lo.v, l.offset[r].l.v, MPFR_RNDD);
      mpfr_mul_d(z.hi.v, l.slope[r].h.v,
                 mpfr_sgn(l.slope[r].h.v) >= 0 ? leaf.hi : leaf.lo, MPFR_RNDU);
      mpfr_add(z.hi.v, z.hi.v, l.offset[r].h.v, MPFR_RNDU);
      const auto &b = kernel.bounds[3 * r + axis];
      mpfr_mul_d(work.move.v, direction > 0 ? b.l.v : b.h.v, signed_step,
                 MPFR_RNDD);
      mpfr_mul_d(work.move.v, work.move.v, leaf.hi, MPFR_RNDD);
      if (mpfr_sgn(work.move.v) < 0)
        mpfr_add(z.lo.v, z.lo.v, work.move.v, MPFR_RNDD);
      mpfr_mul_d(work.move.v, direction > 0 ? b.h.v : b.l.v, signed_step,
                 MPFR_RNDU);
      mpfr_mul_d(work.move.v, work.move.v, leaf.hi, MPFR_RNDU);
      if (mpfr_sgn(work.move.v) > 0)
        mpfr_add(z.hi.v, z.hi.v, work.move.v, MPFR_RNDU);

      // Upper bound on the contribution to -sign(direction) * R[:,k]·u.
      if (direction > 0) {
        mpfr_neg(work.c_lo.v, b.h.v, MPFR_RNDD);
        mpfr_neg(work.c_hi.v, b.l.v, MPFR_RNDU);
      } else {
        mpfr_set(work.c_lo.v, b.l.v, MPFR_RNDD);
        mpfr_set(work.c_hi.v, b.h.v, MPFR_RNDU);
      }
      auto a = mpfr_sgn(work.c_lo.v) >= 0 ? z.hi.v : z.lo.v;
      mpfr_mul(z.x.v, a, mpfr_sgn(a) >= 0 ? work.c_hi.v : work.c_lo.v,
               MPFR_RNDU);
      mpfr_add(work.factor.v, work.factor.v, z.x.v, MPFR_RNDU);
      add_square(z);
    }
    if (!mpfr_number_p(work.factor.v) || !mpfr_number_p(z.ql.v) ||
        !mpfr_number_p(z.qh.v))
      return false;
    const int sign = mpfr_sgn(work.factor.v);
    if (sign == 0)
      continue;
    if (mpfr_cmp_ui(z.ql.v, 128) >= 0) {
      if (sign < 0)
        continue; // The zero lower density is a safe signed upper product.
      unsigned long exponent = 1074;
      if (mpfr_cmp_ui(z.ql.v, 2148) < 0) {
        mpfr_div_2ui(z.x.v, z.ql.v, 1, MPFR_RNDD);
        exponent = mpfr_get_ui(z.x.v, MPFR_RNDD);
      }
      mpfr_set_ui(work.weight.v, 1, MPFR_RNDN);
      mpfr_div_2ui(work.weight.v, work.weight.v, exponent, MPFR_RNDU);
    } else {
      const auto rnd = sign > 0 ? MPFR_RNDU : MPFR_RNDD;
      mpfr_neg(work.weight.v, sign > 0 ? z.ql.v : z.qh.v, rnd);
      mpfr_div_2ui(work.weight.v, work.weight.v, 1, rnd);
      mpfr_exp(work.weight.v, work.weight.v, rnd);
      ++exps;
    }
    mpfr_mul_d(work.weight.v, work.weight.v, kernel.w,
               sign > 0 ? MPFR_RNDU : MPFR_RNDD);
    mpfr_mul(z.x.v, work.factor.v, work.weight.v, MPFR_RNDU);
    mpfr_add(work.total.v, work.total.v, z.x.v, MPFR_RNDU);
  }
  if (!mpfr_number_p(work.total.v))
    return false;
  if (mpfr_sgn(work.total.v) <= 0)
    return leaf.margin > 0;
  mpfr_mul_d(work.total.v, work.total.v, leaf.hi, MPFR_RNDU);
  mpfr_mul_d(work.total.v, work.total.v, step, MPFR_RNDU);
  return mpfr_cmp_d(work.total.v, leaf.margin) < 0;
}

void certify_local(MissProof &proof, const std::vector<Line> &line,
                   const std::vector<RefKernel> &kernels, P receiver,
                   Stats &stats) {
  if (proof.directions.empty() || proof.partition_truncated || !proof.usable ||
      proof.partition.empty())
    return;
  if (proof.leaves < proof.minimum_leaves) {
    stats.add("miss_local_below_work_threshold");
    return;
  }
  auto begin = Clock::now();
  LocalGradientScratch work;
  std::uint64_t calls = 0, exps = 0;
  const double step = double(proof.step) / proof.grid;
  for (int dir : proof.directions) {
    const int axis = std::abs(dir) - 1;
    const double dest = receiver[axis] + (dir > 0 ? step : -step);
    if (!(dest > 0 && dest < 1))
      continue;
    auto start = Clock::now();
    const auto exps_before = exps;
    auto check = [&](std::size_t i) {
      ++calls;
      return local_leaf_free(proof.partition[i], line, kernels, dir, step, work,
                             exps);
    };
    bool certified = check(proof.limiting_leaf);
    for (std::size_t i = 0; certified && i < proof.partition.size(); ++i)
      if (i != proof.limiting_leaf)
        certified = check(i);
    proof.local.push_back({dir, certified, elapsed(start), exps - exps_before});
    stats.add(certified ? "miss_local_certificates" : "miss_local_failures");
    if (certified && !proof.audit)
      break;
  }
  stats.add("miss_local_leaf_tests", calls);
  stats.add("miss_local_exp_endpoints", exps);
  stats.add("miss_local_construction_ns", std::uint64_t(elapsed(begin) * 1e9));
}
} // namespace
