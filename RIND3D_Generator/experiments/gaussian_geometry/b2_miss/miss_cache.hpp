// Included inside b2r. One tile, one independent engine and one exact source.
namespace {
using I3 = std::array<int, 3>;

class MissCache {
  static constexpr int empty = -1, visited = -2;
  const int side, mode;
  Stats &work;
  std::vector<int> seeds;

  std::size_t index(I3 xyz) const {
    return (std::size_t(xyz[2]) * side + xyz[1]) * side + xyz[0];
  }

  static bool future(int from, int to) {
    // The outer stack visits two-voxel cells in reverse Morton order.
    // Within a cell the innermost scalar loops ascend x, y, then z.
    return from / 2 == to / 2 ? to > from : to < from;
  }

  int new_targets(I3 xyz, int direction, int step) const {
    int k = std::abs(direction) - 1, delta = direction < 0 ? -1 : 1;
    const int from = xyz[k], end = from + delta * step;
    if (end < 0 || end >= side)
      return 0;
    int count = 0;
    for (int j = 1; j <= step; ++j) {
      xyz[k] = from + j * delta;
      count += future(from, xyz[k]) && seeds[index(xyz)] == empty;
    }
    return count;
  }

  std::vector<int> directions(I3 xyz) const {
    constexpr std::array<int, 6> order{-2, -3, -1, 2, 3, 1};
    if (mode == 2) {
      for (int d : order)
        if (new_targets(xyz, d, 1))
          return {d};
    } else if (mode == 3) {
      std::array<std::pair<int, int>, 3> ranked{};
      for (int i = 0; i < 3; ++i)
        ranked[i] = {new_targets(xyz, order[i], 4), order[i]};
      std::stable_sort(ranked.begin(), ranked.end(),
                       [](auto a, auto b) { return a.first > b.first; });
      std::vector<int> selected;
      for (int i = 0; i < 2 && ranked[i].first >= 2; ++i)
        selected.push_back(ranked[i].second);
      return selected;
    }
    return {};
  }

public:
  MissCache(int side, int mode, Stats &work, bool traced,
            std::vector<int> &trace)
      : side(side), mode(mode), work(work) {
    auto start = Clock::now();
    const auto size = std::size_t(side) * side * side;
    seeds.assign(size, empty);
    if (traced)
      trace.assign(size, empty);
    work.maximum("miss_cache_payload_bytes",
                 (seeds.capacity() + trace.capacity()) * sizeof(int));
    work.add("miss_cache_setup_ns", std::uint64_t(elapsed(start) * 1e9));
  }

  int ray(const Independent &backend, P source, P receiver, int grid, I3 xyz,
          const Options &options, const RayStudy &study,
          const Independent::PreparedOffsets *offsets,
          const Independent::MissGradient *gradient, std::vector<int> &trace) {
    auto start = Clock::now();
    const auto k = index(xyz);
    const int seed = seeds[k];
    seeds[k] = visited;
    work.add("miss_cache_lookup_ns", std::uint64_t(elapsed(start) * 1e9));
    work.add("miss_cache_lookups");
    if (seed >= 0) {
      if (!trace.empty())
        trace[k] = seed;
      work.add("miss_cache_hits");
      return 0;
    }

    start = Clock::now();
    const auto dirs = directions(xyz);
    work.add("miss_dispatch_ns", std::uint64_t(elapsed(start) * 1e9));
    if (mode != 1 && dirs.empty())
      return backend
          .precision_ray(source, receiver, options, work, study, offsets)
          .status;
    std::vector<I3> coverage;
    auto decision = backend.neighborhood_ray(
        source, receiver, grid, options, work, study, offsets, gradient, dirs,
        mode == 3 ? 4 : 1, mode == 3 ? 5 : 0, coverage);
    if (!coverage.empty()) {
      start = Clock::now();
      for (const auto &delta : coverage) {
        I3 dest{xyz[0] + delta[0], xyz[1] + delta[1], xyz[2] + delta[2]};
        if (std::any_of(dest.begin(), dest.end(),
                        [&](int a) { return a < 0 || a >= side; }))
          continue;
        auto idx = index(dest);
        if (seeds[idx] == empty) {
          seeds[idx] = int(k);
          work.add("miss_cache_new_marks");
        }
      }
      work.add("miss_index_ns", std::uint64_t(elapsed(start) * 1e9));
    }
    return decision.status;
  }
};
} // namespace
