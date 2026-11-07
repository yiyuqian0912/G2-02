// Instrumentation of the unchanged D search. A counterfactual natural bound
// is available only in a separate, untimed diagnostic pass.
#pragma once
#include "profile.hpp"
#include <cstdlib>

namespace b2 {
namespace {
struct PrecisionAudit;
inline thread_local PrecisionAudit *active_precision_audit = nullptr;
inline const bool precision_audit_probe =
    std::getenv("RIND_B2_AUDIT_PROBE64") != nullptr;

struct PrecisionAuditSpan {
  std::uint64_t *sum;
  std::uint64_t start;
  explicit PrecisionAuditSpan(std::uint64_t *sum)
      : sum(sum), start(sum ? ticks() : 0) {}
  ~PrecisionAuditSpan() {
    if (sum)
      *sum += tick_delta(start);
  }
};

struct PrecisionAudit {
  Stats &stats;
  std::array<std::uint64_t, 2> lines{}, offset{}, slope{}, copy{}, exp_time{},
      line_kernels{}, exp_count{};
  std::uint64_t derivative = 0, revisited_at_depth_limit = 0;
  std::array<std::array<std::uint64_t, 3>, 2> interval_decisions{};
  std::array<std::array<std::uint64_t, 3>, 3> counterfactual{};

  explicit PrecisionAudit(Stats &s) : stats(s) {
    active_precision_audit = this;
  }
  ~PrecisionAudit() {
    active_precision_audit = nullptr;
    for (int i = 0; i < 2; ++i) {
      std::string key = i ? "audit256_" : "audit64_";
      for (const auto &[name, time] : {std::pair{"line_ns", lines[i]},
                                       {"offset_ns", offset[i]},
                                       {"slope_ns", slope[i]},
                                       {"vector_copy_ns", copy[i]},
                                       {"exp_ns", exp_time[i]}})
        stats.add(key + name, std::uint64_t(time * ns_per_tick));
      stats.add(key + "line_kernels", line_kernels[i]);
      stats.add(key + "exp_endpoints", exp_count[i]);
      for (int j = 0; j < 3; ++j)
        stats.add(key + "natural_status_" + std::to_string(j - 1),
                  interval_decisions[i][j]);
    }
    if (precision_audit_probe)
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          stats.add("probe256_" + std::to_string(i - 1) + "_64_" +
                        std::to_string(j - 1),
                    counterfactual[i][j]);
    stats.add("audit_interval_derivative_ns",
              std::uint64_t(derivative * ns_per_tick));
    stats.add("audit_revisited_at_depth_limit", revisited_at_depth_limit);
    stats.maximum("audit_tsc_available", profile_cycle_counter);
  }
};

std::uint64_t *audit_slot(std::array<std::uint64_t, 2> PrecisionAudit::*member,
                          int precision) {
  return active_precision_audit
             ? &((active_precision_audit->*member)[precision == 256])
             : nullptr;
}

void audit_exp(mpfr_ptr value, mpfr_rnd_t direction) {
  PrecisionAudit *audit = active_precision_audit;
  int index = mpfr_get_prec(value) == 256;
  PrecisionAuditSpan span(audit ? &audit->exp_time[index] : nullptr);
  mpfr_exp(value, value, direction);
  if (audit)
    ++audit->exp_count[index];
}
} // namespace
} // namespace b2
