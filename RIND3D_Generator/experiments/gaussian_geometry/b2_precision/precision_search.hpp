// Included after the unchanged D search in the independent translation unit.
namespace b2 {
namespace {
struct PrecisionWork {
  Stats &stats;
  std::uint64_t low_nodes = 0, high_nodes = 0, low_exp = 0, high_exp = 0;
  std::uint64_t low_free = 0, high_free = 0, kept_before_high = 0;
  std::uint64_t low_fail = 0, high_line_kernels = 0, low_line_kernels = 0;
  std::uint64_t high_midpoint_rechecks = 0, high_terminal_tests = 0;
  std::uint64_t peak_pending = 1, cached_offsets = 0, splits = 0;
  std::uint64_t reused_gaussian_bounds = 0;
  ~PrecisionWork() {
    for (const auto &[key, value] :
         {std::pair{"precision_low_visits", low_nodes},
          {"precision_high_visits", high_nodes},
          {"precision_low_exp_endpoints", low_exp},
          {"precision_high_exp_endpoints", high_exp},
          {"precision_low_FREE", low_free},
          {"precision_high_FREE", high_free},
          {"precision_prefix_FREE_retained", kept_before_high},
          {"precision_low_inconclusive", low_fail},
          {"precision_low_line_kernels", low_line_kernels},
          {"precision_high_line_kernels", high_line_kernels},
          {"precision_high_midpoint_rechecks", high_midpoint_rechecks},
          {"precision_high_terminal_tests", high_terminal_tests},
          {"precision_source_offset_kernel_reuses", cached_offsets},
          {"precision_subdivisions", splits},
          {"precision_gaussian_term_reuses", reused_gaussian_bounds}}) {
      if (value)
        stats.add(key, value);
    }
    stats.maximum("precision_peak_pending_intervals", peak_pending);
  }
};
} // namespace

struct Independent::PreparedOffsets {
  struct Axis {
    MP lo, hi;
    explicit Axis(int precision) : lo(precision), hi(precision) {}
  };
  using Offset = std::array<Axis, 3>;
  const Impl *owner;
  P source;
  int precision;
  std::vector<Offset> values;
};

std::shared_ptr<const Independent::PreparedOffsets>
Independent::prepare_offsets(P source, int precision, Stats &stats) const {
  if (!finite(source) || impl->analytic)
    return {};
  auto start = Clock::now();
  auto result = std::make_shared<PreparedOffsets>();
  result->owner = impl.get();
  result->source = source;
  result->precision = precision;
  result->values.reserve(impl->kernels.size());
  Scratch z(precision);
  for (const auto &k : impl->kernels) {
    result->values.push_back({PreparedOffsets::Axis(precision),
                              PreparedOffsets::Axis(precision),
                              PreparedOffsets::Axis(precision)});
    for (int r = 0; r < 3; ++r) {
      affine(k, r, source, source, k.c, z);
      auto &v = result->values.back()[r];
      mpfr_set(v.lo.v, z.lo.v, MPFR_RNDD);
      mpfr_set(v.hi.v, z.hi.v, MPFR_RNDU);
    }
  }
  stats.add("precision_source_offset_preparations");
  stats.add("precision_source_offset_prepared_kernels", result->values.size());
  stats.add("precision_source_offset_ns", std::uint64_t(elapsed(start) * 1e9));
  stats.maximum("precision_source_offset_payload_bytes",
                result->values.capacity() * (sizeof(PreparedOffsets::Offset) +
                                             6 * std::size_t(precision) / 8));
  return result;
}

Decision Independent::precision_ray(P source, P receiver, const Options &o,
                                    Stats &s, const RayStudy &study,
                                    const PreparedOffsets *cached) const {
  if (!finite(source) || !finite(receiver) || impl->analytic)
    return ray(source, receiver, o, s);
  s.add("reference_ray_queries");
  auto start = Clock::now();
  PrecisionWork work{s};
  const int p = study.filter_bits;
  if (cached && (cached->owner != impl.get() || cached->source != source ||
                 cached->precision != p))
    throw std::invalid_argument("Source offset enclosure identity mismatch");

  std::vector<Line> low;
  if (cached) {
    Scratch z(p);
    low.reserve(impl->kernels.size());
    for (std::size_t i = 0; i < impl->kernels.size(); ++i) {
      auto &line = low.emplace_back(p);
      for (int r = 0; r < 3; ++r) {
        const auto &v = cached->values[i][r];
        mpfr_set(line.offset[r].l.v, v.lo.v, MPFR_RNDD);
        mpfr_set(line.offset[r].h.v, v.hi.v, MPFR_RNDU);
        affine(impl->kernels[i], r, receiver, receiver, source, z);
        mpfr_set(line.slope[r].l.v, z.lo.v, MPFR_RNDD);
        mpfr_set(line.slope[r].h.v, z.hi.v, MPFR_RNDU);
      }
    }
    work.cached_offsets = low.size();
    s.add("reference_line_builds", low.size());
  } else {
    low = impl->lines(source, receiver, s, p);
  }
  work.low_line_kernels = low.size();

  struct Node {
    double lo, hi;
    int depth;
  };
  std::deque<Node> pending{{0, 1, 0}};
  // Endpoints at depth <= 50 and the midpoint of every visited node are
  // exact binary64 dyadics. This integer also guards the coverage invariant.
  constexpr std::uint64_t whole = std::uint64_t(1) << 50;
  std::uint64_t covered = 0;
  auto close_free = [&](const Node &n, bool low_proof) {
    covered += whole >> n.depth;
    pending.pop_front();
    ++(low_proof ? work.low_free : work.high_free);
  };
  int nodes = 0, depth = 0;
  const char *reason = "reference_depth_or_precision_budget";
  std::vector<Line> high;
  std::unique_ptr<LocalTaylor> local;
  auto tighten_line = [&]() {
    high = impl->lines(source, receiver, s);
    work.high_line_kernels = high.size();
    if (local) {
      local->prepared = false;
      mpfr_set_zero(local->global.v, 0);
    }
  };
  while (!pending.empty()) {
    if (nodes >= o.max_nodes || elapsed(start) > o.query_seconds) {
      reason = nodes >= o.max_nodes ? "reference_node_budget"
                                    : "reference_time_budget";
      break;
    }
    const auto n = pending.front();
    ++nodes;
    depth = std::max(depth, n.depth);
    s.add("reference_ray_nodes");
    s.maximum("reference_max_depth", depth);
    const bool local_node = o.taylor && n.depth >= study.curvature_depth;
    const bool high_node = local_node || n.depth > study.filter_depth;
    if (local_node && !local)
      local = std::make_unique<LocalTaylor>(impl->kernels.size(), p);
    ++work.low_nodes;
    int state = impl->natural(low, n.lo, n.hi, s, o.tail_bounds, p,
                              local_node ? &local->interval_terms : nullptr,
                              nullptr, &work.low_exp);
    if (state == 0) {
      close_free(n, true);
      continue;
    }
    if (state == 1)
      return {1, "independent_parameter_interval_solid", nodes, depth};
    ++work.low_fail;
    const double mid = (n.lo + n.hi) / 2;
    auto eval_high = [&](double lo, double hi, bool tails,
                         std::vector<Bound> *terms = nullptr,
                         Bound *sum = nullptr) {
      return impl->natural(high.empty() ? low : high, lo, hi, s, tails, bits,
                           terms, sum, &work.high_exp);
    };
    if (high_node) {
      if (work.high_nodes == 0)
        work.kept_before_high = work.low_free;
      ++work.high_nodes;
      auto terms = local_node ? &local->midpoint_terms : nullptr;
      auto sum = local_node ? &local->fmid : nullptr;
      state = eval_high(mid, mid, o.tail_bounds, terms, sum);
      if (state < 0 && o.tail_bounds)
        state = eval_high(mid, mid, false, terms, sum);
      if (state < 0 && high.empty()) {
        // An enclosure that still straddles the threshold warrants tightening
        // the line itself. Widening its storage alone would prove no more.
        tighten_line();
        ++work.high_midpoint_rechecks;
        state = eval_high(mid, mid, false, terms, sum);
      }
      if (state == 0 && local_node) {
        work.reused_gaussian_bounds += low.size();
        int proof = local->certify(
            high.empty() ? low : high, impl->kernels, n.lo, mid, n.hi,
            impl->tau.l.v,
            [&](double lo, double hi) {
              return eval_high(lo, hi, o.tail_bounds);
            },
            study.stationary, s);
        if (proof == 0) {
          close_free(n, false);
          continue;
        }
        if (proof == 1)
          return {1, "independent_local_witness", nodes, depth};
      }
    } else {
      state = impl->natural(low, mid, mid, s, o.tail_bounds, p, nullptr,
                            nullptr, &work.low_exp);
      if (state < 0 && o.tail_bounds)
        state = impl->natural(low, mid, mid, s, false, p, nullptr, nullptr,
                              &work.low_exp);
    }
    if (state == 1)
      return {1, "independent_midpoint_witness", nodes, depth};
    if (n.depth >= o.max_depth) {
      if (high.empty())
        tighten_line();
      ++work.high_terminal_tests;
      state = eval_high(n.lo, n.hi, false);
      if (state == 0) {
        close_free(n, false);
        continue;
      }
      if (state == 1)
        return {1, "independent_full_exp_interval", nodes, depth};
      break;
    }
    pending.pop_front();
    pending.push_back({n.lo, mid, n.depth + 1});
    pending.push_back({mid, n.hi, n.depth + 1});
    ++work.splits;
    work.peak_pending =
        std::max(work.peak_pending, std::uint64_t(pending.size()));
  }
  if (pending.empty()) {
    if (covered == whole)
      return {0, "independent_partition_free", nodes, depth};
    s.add("reference_unknown_rays");
    return {-1, "precision_partition_invariant", nodes, depth};
  }

  double remaining = o.query_seconds - elapsed(start);
  if (o.taylor && o.fallback && nodes < o.max_nodes && remaining > 0) {
    Options fallback = o;
    fallback.max_nodes -= nodes;
    fallback.query_seconds = remaining;
    s.add("study_old_independent_fallbacks");
    auto result = ray(source, receiver, fallback, s);
    result.nodes += nodes;
    result.depth = std::max(result.depth, depth);
    return result;
  }
  s.add("reference_unknown_rays");
  return {-1, reason, nodes, depth};
}
} // namespace b2
