// Included after the primary-only directed arithmetic helpers.
namespace {
bool fixed_witness_hit(const HitWitness &w,
                       const std::vector<PrimaryCapture::Line> &lines,
                       const std::vector<Kernel> &kernels, I tau, int direction,
                       double length, bool direct, Exp &exp,
                       std::uint64_t &endpoints) {
  if (!(w.margin > 0) || lines.size() != kernels.size())
    return false;
  const int axis = std::abs(direction) - 1;
  const I displacement = mul(I(w.t), I(length));
  double sum = 0;
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    const auto &k = kernels[i];
    I q, q_other, factor;
    for (int r = 0; r < 3; ++r) {
      const I b = direction > 0 ? k.inv[3*r+axis] : neg(k.inv[3*r+axis]);
      const I z = add(lines[i].offset[r], mul(I(w.t), lines[i].slope[r]));
      if (direct) {
        // The exact squared norm is convex along this axial interval.
        q = add(q, square(z));
        q_other = add(q_other, square(add(z, mul(b, displacement))));
      } else {
        const I swept = add(z, mul(b, I(0, displacement.hi)));
        q = add(q, square(swept));
        factor = add(factor, mul(b, swept));
      }
    }
    if (!std::isfinite(q.lo) || !std::isfinite(q.hi))
      return false;
    if (direct) {
      const double upper_q = std::max(q.hi, q_other.hi);
      if (!std::isfinite(upper_q))
        return false;
      const double density = density_endpoint(exp, upper_q, false, endpoints);
      if (density > 0) {
        const double term = std::max(0., step(k.w * density, false));
        sum = std::max(0., step(sum + term, false));
      }
    } else {
      if (!std::isfinite(factor.hi))
        return false;
      const bool upper = factor.hi >= 0;
      const double density = density_endpoint(exp, upper ? q.lo : q.hi, upper,
                                              endpoints);
      const double weight = step(k.w * factor.hi, true);
      sum = step(sum + step(weight * density, true), true);
    }
    if (!std::isfinite(sum))
      return false;
  }
  return direct ? sum >= tau.hi
                : sum <= 0 || step(displacement.hi * sum, true) <= w.margin;
}
} // namespace

HitAudit Primary::audit_hit(P source, P receiver, int grid, const Options &o,
                            const std::vector<std::array<int, 2>> &requests,
                            bool capture_first) const {
  HitAudit result;
  result.tau_upper = impl->tau.hi;
  PrimaryCapture captured, baseline;
  captured.retain_line = baseline.retain_line = true;
  Stats a, b;
  Decision control;
  auto with_capture = [&] {
    auto t = Clock::now();
    result.decision = ray(source, receiver, o, a, &captured, nullptr, &result.witness);
    result.capture_seconds = elapsed(t);
  };
  auto without_capture = [&] {
    auto t = Clock::now();
    control = ray(source, receiver, o, b, &baseline);
    result.scalar_seconds = elapsed(t);
  };
  if (capture_first) {
    with_capture();
    without_capture();
  } else {
    without_capture();
    with_capture();
  }
  if (control.status != result.decision.status || control.reason != result.decision.reason ||
      control.nodes != result.decision.nodes || a.n != b.n)
    throw std::logic_error("Witness capture changed the scalar decision or work");
  Exp exp;
  for (auto [direction, steps] : requests) {
    auto &r = result.trials.emplace_back();
    r.direction = direction;
    r.steps = steps;
    auto test = [&](bool direct, double &seconds, std::uint64_t &endpoints) {
      auto t = Clock::now();
      const bool ok = fixed_witness_hit(result.witness, captured.lines, impl->kernels,
                                        impl->tau, direction, double(steps)/grid,
                                        direct, exp, endpoints);
      seconds = elapsed(t);
      return ok;
    };
    if (capture_first) {
      r.gradient = test(false, r.gradient_seconds, r.gradient_exp);
      r.direct = test(true, r.direct_seconds, r.direct_exp);
    } else {
      r.direct = test(true, r.direct_seconds, r.direct_exp);
      r.gradient = test(false, r.gradient_seconds, r.gradient_exp);
    }
  }
  return result;
}
