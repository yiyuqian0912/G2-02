#pragma once
#include "common.hpp"

namespace b2 {
struct AxialPlan {
  int short_direction = 0;
  std::array<int, 2> long_directions{};

  bool empty() const { return !short_direction && !long_directions[0]; }
};

struct AxialResult {
  int direction = 0, step = 0;
  // Axis-aligned long attempts are all negative directions.
  std::array<int, 3> long_status{};
};

// Scheduling only. The caller supplies its own certified continuous leaf test.
template <class LeafTest>
AxialResult certify_axial(const AxialPlan &plan, std::size_t leaves,
                          std::size_t limiting, LeafTest check, Stats &stats) {
  AxialResult result;
  auto attempt = [&](int direction, int step) {
    const auto start = Clock::now();
    bool safe = check(limiting, direction, step);
    for (std::size_t i = 0; safe && i < leaves; ++i)
      if (i != limiting)
        safe = check(i, direction, step);
    auto ns = std::uint64_t(elapsed(start) * 1e9);
    if (step == 4) {
      result.long_status[-direction - 1] = safe ? 1 : -1;
      stats.add(safe ? "axial_long_success" : "axial_long_failure");
      stats.add(safe ? "axial_long_success_ns" : "axial_long_failure_ns", ns);
    } else {
      stats.add(safe ? "axial_short_success" : "axial_short_failure");
      stats.add(safe ? "axial_short_success_ns" : "axial_short_failure_ns", ns);
    }
    if (safe) {
      result.direction = direction;
      result.step = step;
    }
    return safe;
  };
  if (leaves >= 5)
    for (int d : plan.long_directions)
      if (d && attempt(d, 4))
        return result;
  if (plan.short_direction)
    attempt(plan.short_direction, 1);
  return result;
}
} // namespace b2
