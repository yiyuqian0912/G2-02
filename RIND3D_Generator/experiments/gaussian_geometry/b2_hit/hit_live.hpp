bool Primary::extend_hit(P source, P receiver, int grid, HitWitness &w,
                         int direction, int distance, bool direct,
                         std::uint64_t &endpoints) const {
  if (!(w.margin > 0))
    return false;
  if (w.lines.empty()) {
    // A seed outside the MISS planner's support did not retain its affine
    // coefficients. Reconstruct those enclosures, without any Gaussian exp.
    for (const auto &k : impl->kernels) {
      auto &line = w.lines.emplace_back();
      coefficients(k, source, receiver, &line);
    }
  }
  Exp exp;
  return fixed_witness_hit(w, w.lines, impl->kernels, impl->tau, direction,
                           double(distance)/grid, direct, exp, endpoints);
}
