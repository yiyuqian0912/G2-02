// Included inside b2r after the region engine definition. This state belongs
// to one backend instance, exact source, fixed options and global grid.
struct TileSession::State {
  Engine engine;
  P source;
  int grid;
  Options options;
  Settings settings;
  bool reuse_cover;
  Cover cover;
  std::shared_ptr<const Independent::PreparedOffsets> offsets;
  Stats preparation;
  double clearance_seconds = 0, cover_seconds = 0;
  std::mutex lock;

  State(const Engine &engine, P source, int grid, const Options &options,
        const Settings &settings, bool reuse_cover)
      : engine(engine), source(source), grid(grid), options(options),
        settings(settings), reuse_cover(reuse_cover) {}
};

TileSession::TileSession(const Engine &engine, P source, int grid,
                         const Options &options, const Settings &settings,
                         bool reuse_cover)
    : p(std::make_shared<State>(engine, source, grid, options, settings,
                                reuse_cover)) {
  require_environment();
  options.validate();
  settings.validate();
  if (!finite(source) || grid < 8 || grid > 1024 || (grid & (grid - 1)) ||
      std::any_of(source.begin(), source.end(),
                  [](double x) { return x <= 0 || x >= 1; }) ||
      (reuse_cover && settings.mode != 5))
    throw std::invalid_argument("Invalid source or grid for a tile session");

  auto start = Clock::now();
  Box clearance;
  constexpr double radius = 0x1p-26;
  for (int j = 0; j < 3; ++j) {
    clearance.lo[j] = std::nextafter(source[j] - radius, -INFINITY);
    clearance.hi[j] = std::nextafter(source[j] + radius, INFINITY);
  }
  if (engine.p->box(clearance, options, p->preparation).status != 0)
    throw std::invalid_argument(
        "Independent backend source clearance unresolved");
  p->clearance_seconds = elapsed(start);
  if (reuse_cover) {
    start = Clock::now();
    p->cover = engine.p->build(options, settings, p->preparation);
    p->cover_seconds = elapsed(start);
    if (engine.p->independent && settings.study.method == 5)
      p->offsets = engine.p->b->prepare_offsets(
          source, settings.study.filter_bits, p->preparation);
  }
}

Result TileSession::tile(std::array<int, 3> origin, int side, double seconds) {
  std::lock_guard guard(p->lock);
  return p->engine.tile_scoped(p->source, p->grid, origin, side, seconds,
                               p->options, p->settings, this);
}

std::map<std::string, double> TileSession::info() const {
  std::lock_guard guard(p->lock);
  std::size_t bytes = p->cover.nodes.capacity() * sizeof(Node) +
                      p->cover.inner.capacity() * sizeof(int);
  std::size_t projections = 0;
  for (const auto &node : p->cover.nodes) {
    bytes += node.planes.capacity() * sizeof(Plane);
    projections += node.projected;
  }
  return {{"independent", double(p->engine.p->independent)},
          {"grid", double(p->grid)},
          {"geometry_setup_seconds", p->engine.p->setup_seconds},
          {"source_clearance_seconds", p->clearance_seconds},
          {"cover_seconds", p->cover_seconds},
          {"cover_nodes", double(p->cover.nodes.size())},
          {"cover_payload_bytes", double(bytes)},
          {"projection_nodes", double(projections)}};
}

std::map<std::string, std::uint64_t> TileSession::preparation_work() const {
  std::lock_guard guard(p->lock);
  return p->preparation.n;
}
