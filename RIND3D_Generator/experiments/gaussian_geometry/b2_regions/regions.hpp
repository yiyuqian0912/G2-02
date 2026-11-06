#pragma once
#include "common.hpp"
#include "interval.hpp"

namespace b2r {
using namespace b2;
struct Settings {
  int mode =
      5; // 0 scalar,1 old hierarchy,2 FREE,3 SHADOW,4 both,5 proof reuse.
  int cover_depth = 6, cover_nodes = 32769, shadow_boxes = 256;
  int parameter_nodes = 31, parameter_depth = 6;
  double cover_seconds = 15.;
  bool fast_exp = false, volume_after_cache = false, trace = false;
  void validate() const;
};
struct Result {
  Field field;
  std::vector<std::uint8_t> evaluation_path;
  std::vector<std::uint64_t> ray_nanoseconds, point_nanoseconds;
  double setup_seconds = 0, cover_seconds = 0, region_seconds = 0,
         fallback_seconds = 0, assembly_seconds = 0;
};
class Engine {
  struct Impl;
  std::shared_ptr<Impl> p;

public:
  Engine(const Inputs &, bool independent);
  Result field(P source, int resolution, double seconds, const Options &,
               const Settings &) const;
  Result tile(P source, int world_resolution, std::array<int, 3> origin,
              int side, double seconds, const Options &,
              const Settings &) const;
  std::pair<int, int> region(P source, P lo, P hi, const Options &,
                             const Settings &, Stats &) const;
};
I certified_exp(double x, bool fast, Stats &);
std::map<std::string, bool> geometry_screen(P source, P receiver_lo,
                                            P receiver_hi, P obstacle_lo,
                                            P obstacle_hi);
} // namespace b2r
