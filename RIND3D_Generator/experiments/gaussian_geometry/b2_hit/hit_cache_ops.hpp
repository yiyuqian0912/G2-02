// Included inside MissCache. MISS planning and its signed seed array stay intact.
struct HitState {
  const int mode;
  const Field &field;
  std::vector<int> &trace;
  std::vector<int> marks;
  std::optional<AxialHistory> history;

  HitState(int mode, const Field &field, std::vector<int> &trace)
      : mode(mode), field(field), trace(trace) {}

  int at(std::size_t k) const { return marks.empty() ? -1 : marks[k]; }
  int physical(std::size_t k) const {
    return field.valid[k] ? 1 - field.response[k] : -1;
  }
};
std::optional<HitState> hits;

static int along(I3 &xyz, int direction, int distance) {
  const int k = std::abs(direction)-1;
  xyz[k] += direction < 0 ? -distance : distance;
  return k;
}

int hit_targets(I3 xyz, int direction, int distance) const {
  I3 end = xyz;
  const int k = along(end, direction, distance);
  if (end[k] < 0 || end[k] >= side)
    return 0;
  int count = 0;
  for (int j = 1; j <= distance; ++j) {
    I3 p = xyz;
    along(p, direction, j);
    const auto i = index(p);
    if (seeds[i] >= 0 || hits->physical(i) == 0)
      return 0; // A known lit center can only make a proposal unprofitable.
    count += future(xyz[k], p[k]) && hits->at(i) < 0 && hits->physical(i) < 0;
  }
  return count;
}

void mark_hit(I3 xyz, int direction, int distance) {
  const auto start = Clock::now();
  const int seed = int(index(xyz));
  for (int j = 1; j <= distance; ++j) {
    I3 p = xyz;
    const int k = along(p, direction, j);
    const auto i = index(p);
    require_compatible_hit_label(seeds[i] >= 0 ? 0 : -1, 1);
    require_compatible_hit_label(hits->physical(i), 1);
    if (future(xyz[k], p[k]) && hits->at(i) < 0 && !hits->field.valid[i]) {
      hits->marks[i] = seed;
      work.add("hit_cache_new_marks");
    }
  }
  work.add("hit_index_ns", std::uint64_t(elapsed(start)*1e9));
}

void try_hit(const Primary &backend, P source, P receiver, int grid, I3 xyz,
             HitWitness &w) {
  work.add("hit_strict_witnesses");
  if (hits->mode == 1)
    return; // Capture-only control.
  const int mode = hits->mode;
  constexpr std::array<int, 6> directions{-2, -3, -1, 2, 3, 1};
  const auto dispatch = Clock::now();
  int long_dir = 0, short_dir = 0, best = 1;
  if (mode >= 3 && (mode != 4 || w.margin >= .008 * w.t * w.tau_upper)) {
    for (int i = 0; i < 3; ++i) {
      const int d = directions[i], count = hit_targets(xyz, d, 4);
      if (count >= 2 && hits->history && !hits->history->permit(xyz, d)) {
        work.add("hit_long_cooldown");
        continue;
      }
      if (count > best) {
        best = count;
        long_dir = d;
      }
    }
  }
  if (mode == 2 || mode == 4)
    for (int d : directions)
      if (hit_targets(xyz, d, 1)) {
        short_dir = d;
        break;
      }
  work.add("hit_dispatch_ns", std::uint64_t(elapsed(dispatch)*1e9));
  auto attempt = [&](int d, int distance) {
    const auto start = Clock::now();
    std::uint64_t endpoints = 0;
    const bool ok = backend.extend_hit(source, receiver, grid, w, d, distance,
                                       mode != 5, endpoints);
    const auto ns = std::uint64_t(elapsed(start)*1e9);
    work.add("hit_extension_ns", ns);
    work.add("hit_extension_exp_endpoints", endpoints);
    if (distance == 4) {
      work.add(ok ? "hit_long_success" : "hit_long_failure");
      if (hits->history) {
        AxialResult r;
        r.long_status[-d-1] = ok ? 1 : -1;
        hits->history->observe(xyz, r);
      }
    } else {
      work.add(ok ? "hit_short_success" : "hit_short_failure");
    }
    if (ok)
      mark_hit(xyz, d, distance);
    else
      work.add("hit_failed_extension_ns", ns);
    return ok;
  };
  if (long_dir && attempt(long_dir, 4))
    return;
  if (short_dir)
    attempt(short_dir, 1);
}

public:
void attach_hit(int hit_mode, const Field &field, bool traced,
                std::vector<int> &trace) {
  const auto start = Clock::now();
  hits.emplace(hit_mode, field, trace);
  if (hit_mode > 1) {
    hits->marks.assign(seeds.size(), empty);
    if (traced)
      trace.assign(seeds.size(), empty);
    if (hit_mode == 4)
      hits->history.emplace(side, work);
  }
  work.maximum("hit_cache_payload_bytes",
               (hits->marks.capacity()+trace.capacity())*sizeof(int));
  work.add("hit_cache_setup_ns", std::uint64_t(elapsed(start)*1e9));
}

void check_hit_labels() const {
  if (!hits || hits->marks.empty())
    return;
  for (std::size_t i = 0; i < seeds.size(); ++i) {
    if (hits->at(i) >= 0) {
      require_compatible_hit_label(seeds[i] >= 0 ? 0 : -1, 1);
      require_compatible_hit_label(hits->physical(i), 1);
    } else if (seeds[i] >= 0) {
      require_compatible_hit_label(hits->physical(i), 0);
    }
  }
}

private:
