HitAudit TileSession::hit_audit(P receiver,
                                const std::vector<std::array<int, 2>> &requests,
                                bool controls, bool capture_first) {
  std::lock_guard guard(p->lock);
  require_environment();
  if (p->engine.p->independent || requests.size() > 12)
    throw std::invalid_argument("Primary HIT audit requires a bounded primary session");
  for (double v : receiver) {
    const double i = v * p->grid - .5;
    if (!std::isfinite(i) || i < 0 || i >= p->grid || i != std::floor(i))
      throw std::invalid_argument("Expected a receiver voxel center");
  }
  for (auto [direction, steps] : requests) {
    if (direction == 0 || direction < -3 || direction > 3 ||
        (steps != 1 && steps != 4))
      throw std::invalid_argument("Expected an axial one- or four-voxel request");
    const double end = receiver[std::abs(direction)-1] +
                       double(direction < 0 ? -steps : steps)/p->grid;
    if (!(end > 0 && end < 1))
      throw std::invalid_argument("Audit target is outside the world");
  }
  auto &e = *p->engine.p;
  auto result = e.a->audit_hit(p->source, receiver, p->grid, p->options,
                               requests, capture_first);
  if (controls) {
    Stats stats;
    for (auto &r : result.trials) {
      P end = receiver;
      end[std::abs(r.direction)-1] += double(r.direction < 0 ? -r.steps : r.steps)/p->grid;
      Box box{receiver, receiver};
      for (int k = 0; k < 3; ++k) {
        box.lo[k] = std::min(box.lo[k], end[k]);
        box.hi[k] = std::max(box.hi[k], end[k]);
      }
      r.controls = true;
      auto t = Clock::now();
      r.cover = p->reuse_cover && e.shadow(p->cover, p->source, box, stats);
      r.cover_seconds = elapsed(t);
      t = Clock::now();
      r.proxy = e.old_cert(p->source, box, p->options, stats, -1, -1).first == 0;
      r.proxy_seconds = elapsed(t);
      t = Clock::now();
      r.section = e.fused(p->source, box, p->options, p->settings, stats) == 0;
      r.section_seconds = elapsed(t);
    }
  }
  return result;
}
