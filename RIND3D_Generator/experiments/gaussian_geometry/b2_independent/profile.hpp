// Timing-only build. Use a calibrated invariant cycle counter when available,
// because HPET can cost more than an individual MPFR exponential.
#pragma once
#include "common.hpp"
#include <thread>

#if defined(__x86_64__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace b2 {
namespace {
bool invariant_counter_available() {
#if defined(__x86_64__)
  unsigned a, b, c, d;
  if (!__get_cpuid(0x80000001, &a, &b, &c, &d) || !(d & (1u << 27)))
    return false;
  return __get_cpuid(0x80000007, &a, &b, &c, &d) && (d & (1u << 8));
#else
  return false;
#endif
}

const bool profile_cycle_counter = invariant_counter_available();

std::uint64_t ticks() {
#if defined(__x86_64__)
  if (profile_cycle_counter) {
    unsigned cpu;
    _mm_lfence();
    auto t = __rdtscp(&cpu);
    _mm_lfence();
    return t;
  }
#endif
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

const double ns_per_tick = [] {
  if (!profile_cycle_counter)
    return 1.;
  auto begin = Clock::now();
  auto t0 = ticks();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  auto t1 = ticks();
  return elapsed(begin) * 1e9 / double(t1 - t0);
}();

std::uint64_t tick_delta(std::uint64_t start) {
  auto stop = ticks();
  return stop >= start ? stop - start : 0;
}

inline thread_local bool profile_ray_active = false;
inline thread_local std::uint64_t profile_exp_ticks = 0, profile_exp_count = 0;

struct ProfileScope {
  Stats &stats;
  explicit ProfileScope(Stats &stats) : stats(stats) {
    profile_ray_active = true;
    profile_exp_ticks = 0;
    profile_exp_count = 0;
  }
  ~ProfileScope() {
    profile_ray_active = false;
    stats.add("profile_ray_exp_ns",
              std::uint64_t(profile_exp_ticks * ns_per_tick));
    stats.add("profile_ray_exp_endpoints", profile_exp_count);
    stats.maximum("profile_cycle_counter_used", profile_cycle_counter);
    stats.maximum("profile_picoseconds_per_tick",
                  std::uint64_t(ns_per_tick * 1e3));
  }
};

struct ProfileTimer {
  Stats &stats;
  const char *key;
  std::uint64_t start = ticks();
  bool failed = false, full_exp = false;

  ~ProfileTimer() {
    if (profile_ray_active) {
      auto ns = std::uint64_t(tick_delta(start) * ns_per_tick);
      stats.add(key, ns);
      if (failed) {
        stats.add("profile_inconclusive_bound_ns", ns);
        stats.add("profile_inconclusive_bounds");
      }
      if (full_exp)
        stats.add("profile_full_exp_retry_ns", ns);
    }
  }
};

void profile_exp(mpfr_ptr dst, mpfr_srcptr x, mpfr_rnd_t round) {
  if (!profile_ray_active) {
    mpfr_exp(dst, x, round);
    return;
  }
  auto start = ticks();
  mpfr_exp(dst, x, round);
  profile_exp_ticks += tick_delta(start);
  ++profile_exp_count;
}
} // namespace
} // namespace b2
