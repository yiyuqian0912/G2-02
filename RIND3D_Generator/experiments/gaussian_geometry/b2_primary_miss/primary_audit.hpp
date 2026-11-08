// Included in the region traversal TU, where timing covers scalar rays only.
namespace {
class PrimaryTileAudit {
  struct Totals {
    std::uint64_t queries = 0, exp = 0, high = 0, nodes = 0, failures = 0,
                  exact_failures = 0, splits = 0, proofs = 0, leaves = 0,
                  samples = 0, sampled_exp = 0, sampled_ticks = 0;
  };

  Stats &stats;
  const int mode;
  std::array<Totals, 3> by_status{};
  std::uint64_t queries = 0;
  const std::uint64_t &exp, &high;

public:
  PrimaryTileAudit(Stats &s, int mode)
      : stats(s), mode(mode), exp(s.n["exp_endpoints"]),
        high(s.n["high_precision_exp_endpoints"]) {}

  Decision ray(const Primary &backend, P source, P receiver, const Options &o) {
    const auto exp_before = exp, high_before = high;
    const bool sampled = (queries++ & 127) == 0;
    PrimaryRayWork work;
    PrimaryCapture capture;
    auto d = [&] {
      if (!sampled)
        return backend.ray(source, receiver, o, stats,
                           mode == 2 ? &capture : nullptr, &work);
      struct Sample {
        Sample() {
          profile_ray_active = true;
          profile_exp_count = profile_exp_ticks = 0;
          primary_exp_probe = profile_exp;
        }
        ~Sample() {
          primary_exp_probe = nullptr;
          profile_ray_active = false;
        }
      } sample;
      return backend.ray(source, receiver, o, stats,
                         mode == 2 ? &capture : nullptr, &work);
    }();
    auto &row = by_status[d.status < 0 ? 2 : d.status];
    ++row.queries;
    row.exp += exp - exp_before;
    row.high += high - high_before;
    row.nodes += d.nodes;
    row.failures += work.inconclusive;
    row.exact_failures += work.exact_inconclusive;
    row.splits += work.splits;
    row.proofs += capture.complete && capture.usable;
    row.leaves += capture.size;
    if (sampled) {
      ++row.samples;
      row.sampled_exp += profile_exp_count;
      row.sampled_ticks += profile_exp_ticks;
    }
    return d;
  }

  void report() {
    stats.maximum("primary_audit_invariant_cycle_counter", profile_cycle_counter);
    stats.maximum("primary_audit_capture_capacity_bytes", sizeof(PrimaryCapture));
    for (int k = 0; k < 3; ++k) {
      const char *label = k == 0 ? "MISS" : k == 1 ? "HIT" : "unresolved";
      const std::string prefix = std::string("primary_audit_") + label + "_";
      const auto &r = by_status[k];
      for (const auto &[key, count] :
           {std::pair{"rays", r.queries}, {"exp_endpoints", r.exp},
            {"high_exp_endpoints", r.high}, {"nodes", r.nodes},
            {"inconclusive_natural", r.failures},
            {"inconclusive_exact", r.exact_failures}, {"splits", r.splits},
            {"usable_proofs", r.proofs}, {"recorded_leaves", r.leaves},
            {"sampled_rays", r.samples}, {"sampled_exp_calls", r.sampled_exp},
            {"sampled_exp_ns", std::uint64_t(r.sampled_ticks * ns_per_tick)}})
        stats.add(prefix + key, count);
    }
  }
};
} // namespace
