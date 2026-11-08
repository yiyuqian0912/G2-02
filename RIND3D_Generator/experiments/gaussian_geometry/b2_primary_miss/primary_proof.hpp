#pragma once
#include "interval.hpp"

namespace b2 {
struct PrimaryRayWork {
  std::uint64_t inconclusive = 0, exact_inconclusive = 0, splits = 0;
};

struct PrimaryCapture {
  struct Leaf {
    double lo, hi, margin;
  };
  static constexpr std::uint64_t whole = std::uint64_t(1) << 50;
  std::array<Leaf, 256> leaves;
  std::size_t size = 0;
  std::uint64_t covered = 0;
  bool usable = true, complete = false;

  void close(double lo, double hi, int depth, double margin) {
    covered += whole >> depth;
    if (size == leaves.size() || !(margin > 0) || !std::isfinite(margin)) {
      usable = false;
      return;
    }
    leaves[size++] = {lo, hi, margin};
  }

  void close(double lo, double hi, int depth, I value, I tau) {
    close(lo, hi, depth, sub(I(tau.lo), I(value.hi)).lo);
  }
};
} // namespace b2
