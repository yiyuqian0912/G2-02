#pragma once
#include "common.hpp"
#include <stdexcept>
namespace b2 {
// Shared traversal/storage only. Every mask/visibility decision is obtained
// separately from the selected implementation, never copied from its peer.
template <class Engine>
Field walk_field(const Engine &e, P source, int n, double seconds,
                 const Options &o, bool scalar) {
  if (n != 8 && n != 16 && n != 32 && n != 64 && n != 128)
    throw std::invalid_argument(
        "Small-field gate: resolution 8/16/32/64/128 only");
  if (!finite(source) || seconds <= 0)
    throw std::invalid_argument("Invalid field input");
  Field f;
  f.resolution = n;
  std::size_t count = std::size_t(n) * n * n;
  f.response.resize(count);
  f.solid.resize(count);
  f.valid.resize(count);
  auto start = Clock::now();
  struct Cell {
    std::array<int, 3> origin;
    int side, y, m;
  };
  std::vector<Cell> stack{{{0, 0, 0}, n, -1, -1}};
  auto index = [n](int x, int y, int z) {
    return (std::size_t(z) * n + y) * n + x;
  };
  while (!stack.empty()) {
    if (elapsed(start) > seconds) {
      f.reason = "field_time_budget";
      break;
    }
    Cell c = stack.back();
    stack.pop_back();
    int cy = c.y, cm = c.m;
    f.stats.add("field_cells");
    P lo, hi;
    for (int a = 0; a < 3; a++) {
      lo[a] = (c.origin[a] + .5) / n;
      hi[a] = (c.origin[a] + c.side - .5) / n;
    }
    if (!scalar && o.regions) {
      auto cert = e.certificate(source, lo, hi, o, f.stats,
                                o.inherited_regions ? cy : -1,
                                o.inherited_regions ? cm : -1);
      if (cy < 0)
        cy = cert.first;
      if (cm < 0)
        cm = cert.second;
    }
    if (cy >= 0 && cm >= 0) {
      for (int z = c.origin[2]; z < c.origin[2] + c.side; z++)
        for (int y = c.origin[1]; y < c.origin[1] + c.side; y++)
          for (int x = c.origin[0]; x < c.origin[0] + c.side; x++) {
            auto k = index(x, y, z);
            f.response[k] = cy;
            f.solid[k] = cm;
            f.valid[k] = 1;
          }
      f.stats.add("region_filled_voxels",
                  std::uint64_t(c.side) * c.side * c.side);
      continue;
    }
    if (c.side > 2 && !scalar) {
      int h = c.side / 2;
      for (int z : {0, h})
        for (int y : {0, h})
          for (int x : {0, h})
            stack.push_back(
                {{c.origin[0] + x, c.origin[1] + y, c.origin[2] + z},
                 h,
                 cy,
                 cm});
      continue;
    }
    for (int z = c.origin[2]; z < c.origin[2] + c.side; z++)
      for (int y = c.origin[1]; y < c.origin[1] + c.side; y++)
        for (int x = c.origin[0]; x < c.origin[0] + c.side; x++) {
          if (elapsed(start) > seconds) {
            f.reason = "field_time_budget";
            goto done;
          }
          P p{(x + .5) / n, (y + .5) / n, (z + .5) / n};
          int m = cm >= 0 ? cm : e.point(p, o, f.stats).status;
          auto k = index(x, y, z);
          if (m < 0) {
            f.stats.add("unresolved_membership_voxels");
            continue;
          }
          if (m == 1) {
            f.solid[k] = 1;
            f.valid[k] = 1;
            continue;
          }
          int v = cy >= 0 ? 1 - cy : e.ray(source, p, o, f.stats).status;
          if (v < 0) {
            f.stats.add("unresolved_visibility_voxels");
            continue;
          }
          f.response[k] = 1 - v;
          f.valid[k] = 1;
        }
  }
done:
  f.complete = std::all_of(f.valid.begin(), f.valid.end(),
                           [](auto x) { return x == 1; });
  if (f.reason.empty())
    f.reason = f.complete ? "complete" : "numerically_incomplete";
  f.seconds = elapsed(start);
  return f;
}
} // namespace b2
