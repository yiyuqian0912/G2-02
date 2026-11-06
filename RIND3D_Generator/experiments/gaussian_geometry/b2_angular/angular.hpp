#pragma once
#include "common.hpp"
#include "interval.hpp"
#include <functional>
#include <limits>

namespace b2a {
using namespace b2;
struct Box {
  P lo, hi;
};
bool good(Box);
bool contains(Box, Box);
bool overlaps(Box, Box);
// Necessary intersection interval; false is a sufficient exclusion.
bool beam_clip(P source, const std::array<I, 3> &direction, Box obstacle,
               double cap, double &lower, Stats &);
bool volume_possible(P, Box receiver, Box obstacle, Stats &);
struct Plane {
  std::array<I, 3> a;
  I rhs;
};
struct ExactPlane {
  std::array<Q, 3> a;
  Q rhs;
};
std::vector<ExactPlane> project_exact(P, Box);
bool plane_excludes(const std::vector<Plane> &, Box, Stats &);
bool plane_contains(const std::vector<Plane> &, Box, Stats &);
std::vector<Plane> enclose_planes(const std::vector<ExactPlane> &);
bool primary_same_box(P, Box, Box, Stats &);

struct ChartQuery {
  int face = -1;
  double u0 = 0, u1 = 0, v0 = 0, v1 = 0, near = 0, far = 0;
};
ChartQuery chart(P, Box);
struct AngularOptions {
  int depth = 10, nodes = 32766, references = 2000000;
  bool fusion = true, empty_prefix = false;
  int prefix_builds = 512;
  void validate() const;
};
struct AngularDecision {
  int visibility = -1;
  double first_hit_upper = INFINITY, empty_prefix = 0;
  int distinct_blockers = 0;
};
// Each map belongs to exactly one source and exactly one physical
// implementation.
class AngularMap {
  struct Impl;
  std::shared_ptr<Impl> p;

public:
  AngularMap(P, bool independent, const AngularOptions &, Stats &);
  void add_inner(Box);
  void outer_refined();
  AngularDecision
  query(Box, const std::function<double(int, double, double, double, double)>
                 &prefix = {});
  std::uint64_t bytes() const;
  std::vector<std::array<double, 8>> snapshot() const;
};

struct Settings {
  // 0 point outer-cover control, 1 adaptive spatial, 2 angular single,
  // 3 angular union, 4 hybrid.
  int method = 3, cover_depth = 6, cover_nodes = 32769, shadow_boxes = 256;
  int adaptive_depth = 9, adaptive_nodes = 8192, adaptive_batch = 8;
  int angular_depth = 10, angular_nodes = 32766, angular_references = 2000000;
  int angular_boxes = -1; // -1 all independently certified inner leaves.
  int prefix_builds = 512;
  double cover_seconds = 15., adaptive_seconds = 10.;
  bool empty_prefix = false, trace = false;
  void validate() const;
};
struct Result {
  Field field;
  std::vector<std::uint8_t> evaluation_path, visibility_evidence;
  std::vector<std::uint64_t> ray_nanoseconds, point_nanoseconds;
  std::vector<std::array<double, 8>> angular_tiles;
  double setup_seconds = 0, cover_seconds = 0, region_seconds = 0,
         angular_seconds = 0, angular_setup_seconds = 0,
         angular_leaf_seconds = 0, adaptive_seconds = 0, fallback_seconds = 0,
         assembly_seconds = 0, point_cover_seconds = 0;
};
class Engine {
  struct Impl;
  std::shared_ptr<Impl> p;

public:
  Engine(const Inputs &, bool independent);
  Result tile(P, int world_grid, std::array<int, 3>, int side, double seconds,
              const Options &, const Settings &) const;
};
} // namespace b2a
