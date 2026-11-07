// This header is included only in the independent translation unit.
namespace b2 {
struct MissLeaf {
  double lo, hi, margin;
};
struct MissLocalResult {
  int direction;
  bool certified;
  double seconds;
  std::uint64_t exps;
};
struct MissProof {
  Bound enclosure;
  MP best_lower{bits}, tmp{bits};
  double margin = INFINITY, normalized = INFINITY, limiting_lo = 0,
         limiting_hi = 1;
  int leaves = 0, grid = 256, step = 1;
  int minimum_leaves = 0;
  std::vector<int> directions;
  std::vector<MissLeaf> partition;
  std::vector<MissLocalResult> local;
  std::size_t limiting_leaf = 0;
  bool partition_truncated = false;
  bool complete = false, usable = true;
  bool audit = false, sample_capture = false;
  std::uint64_t capture_ns = 0;

  void observe(mpfr_srcptr lower) {
    if (audit && mpfr_number_p(lower) && mpfr_cmp(lower, best_lower.v) > 0)
      mpfr_set(best_lower.v, lower, MPFR_RNDD);
  }

  void close(double lo, double hi, mpfr_srcptr tau) {
    Clock::time_point start;
    if (sample_capture)
      start = Clock::now();
    ++leaves;
    if (!mpfr_number_p(tau) || !mpfr_number_p(enclosure.h.v) || !(hi > 0) ||
        mpfr_cmp(enclosure.h.v, tau) >= 0) {
      usable = false;
      return;
    }
    mpfr_sub(tmp.v, tau, enclosure.h.v, MPFR_RNDD);
    const double lower = mpfr_get_d(tmp.v, MPFR_RNDD);
    margin = std::min(margin, lower);
    if (!directions.empty()) {
      if (partition.size() < 256)
        partition.push_back({lo, hi, lower});
      else
        partition_truncated = true;
    }
    mpfr_div_d(tmp.v, tmp.v, hi, MPFR_RNDD);
    const double slack = mpfr_get_d(tmp.v, MPFR_RNDD);
    if (slack < normalized) {
      normalized = slack;
      if (!partition.empty())
        limiting_leaf = partition.size() - 1;
      limiting_lo = lo;
      limiting_hi = hi;
    }
    if (sample_capture)
      capture_ns += std::uint64_t(elapsed(start) * 1e9);
  }
};

#include "miss_local.hpp"

struct Independent::MissGradient {
  std::array<MP, 3> component{MP(bits), MP(bits), MP(bits)};
  MP sum{bits};

  std::array<double, 3> stencil(P receiver, int grid, double slack,
                                std::vector<std::array<int, 3>> &out,
                                double *cube = nullptr) const {
    MP capacity(bits), v(bits), cost(bits), tmp(bits);
    mpfr_set_d(capacity.v, slack, MPFR_RNDD);
    mpfr_mul_ui(capacity.v, capacity.v, grid, MPFR_RNDD);
    auto radius = [&](mpfr_srcptr bound) {
      mpfr_div(v.v, capacity.v, bound, MPFR_RNDD);
      return mpfr_get_d(v.v, MPFR_RNDD);
    };
    if (cube)
      *cube = radius(sum.v);
    std::array<double, 3> radii{};
    std::array<int, 3> extents{};
    for (int k = 0; k < 3; ++k) {
      radii[k] = radius(component[k].v);
      extents[k] = int(std::clamp(radii[k], 0., 4.));
    }
    for (int z = -extents[2]; z <= extents[2]; ++z)
      for (int y = -extents[1]; y <= extents[1]; ++y)
        for (int x = -extents[0]; x <= extents[0]; ++x) {
          std::array<int, 3> offset{x, y, z};
          if (x == 0 && y == 0 && z == 0)
            continue;
          bool inside = true;
          for (int k = 0; k < 3; ++k) {
            const double y = receiver[k] + double(offset[k]) / grid;
            inside = inside && y > 0 && y < 1;
          }
          if (!inside)
            continue;
          mpfr_set_zero(cost.v, 0);
          for (int k = 0; k < 3; ++k) {
            mpfr_mul_ui(tmp.v, component[k].v, std::abs(offset[k]), MPFR_RNDU);
            mpfr_add(cost.v, cost.v, tmp.v, MPFR_RNDU);
          }
          if (mpfr_cmp(cost.v, capacity.v) < 0)
            out.push_back(offset);
        }
    return radii;
  }
};

std::shared_ptr<const Independent::MissGradient>
Independent::prepare_miss_gradient(Stats &stats) const {
  auto start = Clock::now();
  auto out = std::make_shared<MissGradient>();
  MP radius_peak(bits), v(bits);
  mpfr_set_d(radius_peak.v, -.5, MPFR_RNDN);
  mpfr_exp(radius_peak.v, radius_peak.v, MPFR_RNDU);
  for (const auto &kernel : impl->kernels) {
    for (int k = 0; k < 3; ++k) {
      Q norm2;
      for (int j = 0; j < 3; ++j) {
        const auto &entry = kernel.inv[j * 3 + k];
        norm2 += entry * entry;
      }
      mpfr_set_q(v.v, norm2.get_mpq_t(), MPFR_RNDU);
      mpfr_sqrt(v.v, v.v, MPFR_RNDU);
      mpfr_mul_d(v.v, v.v, kernel.w, MPFR_RNDU);
      mpfr_mul(v.v, v.v, radius_peak.v, MPFR_RNDU);
      mpfr_add(out->component[k].v, out->component[k].v, v.v, MPFR_RNDU);
    }
  }
  for (const auto &g : out->component)
    mpfr_add(out->sum.v, out->sum.v, g.v, MPFR_RNDU);
  stats.add("miss_global_gradient_preparation_ns",
            std::uint64_t(elapsed(start) * 1e9));
  stats.add("miss_global_gradient_columns", impl->kernels.size() * 3);
  return out;
}

Decision Independent::neighborhood_ray(
    P source, P receiver, int grid, const Options &o, Stats &stats,
    const RayStudy &study, const PreparedOffsets *offsets,
    const MissGradient *gradient, const std::vector<int> &directions, int step,
    int minimum_leaves, std::vector<std::array<int, 3>> &out) const {
  auto begin = Clock::now();
  MissProof proof;
  proof.grid = grid;
  proof.directions = directions;
  proof.step = step;
  proof.minimum_leaves = minimum_leaves;
  proof.sample_capture = (stats.n["miss_capture_rays"] & 255) == 0;
  stats.add("miss_capture_rays");
  if (!directions.empty())
    proof.partition.reserve(16);
  auto d = precision_ray(source, receiver, o, stats, study, offsets, &proof);
  stats.add("miss_captured_leaves", proof.leaves);
  stats.maximum("miss_partition_payload_bytes",
                proof.partition.capacity() * sizeof(MissLeaf));
  if (proof.sample_capture) {
    stats.add("miss_capture_sample_rays");
    stats.add("miss_capture_sample_leaves", proof.leaves);
    stats.add("miss_capture_sample_ns", proof.capture_ns);
  }
  stats.add("miss_solver_and_certificate_ns",
            std::uint64_t(elapsed(begin) * 1e9));
  if (d.status != 0 || !proof.complete || !proof.usable)
    return d;
  if (gradient) {
    auto start = Clock::now();
    gradient->stencil(receiver, grid, proof.normalized, out);
    stats.add("miss_global_receiver_test_ns",
              std::uint64_t(elapsed(start) * 1e9));
    stats.add(out.empty() ? "miss_global_empty" : "miss_global_certificates");
  }
  for (const auto &local : proof.local) {
    if (!local.certified)
      continue;
    for (int j = 1; j <= step; ++j) {
      std::array<int, 3> offset{};
      offset[std::abs(local.direction) - 1] = local.direction > 0 ? j : -j;
      out.push_back(offset);
    }
  }
  return d;
}

std::pair<Decision, std::map<std::string, double>>
Independent::miss_probe(P source, P receiver, int grid, const Options &o,
                        Stats &stats, const RayStudy &study,
                        const PreparedOffsets *offsets,
                        const MissGradient &gradient,
                        const std::vector<int> &directions, int step) const {
  if (study.stationary)
    throw std::invalid_argument("Neighborhood audit uses the frozen F policy");
  if (directions.size() > 6 || step < 1 || step > 4 ||
      std::any_of(directions.begin(), directions.end(),
                  [](int d) { return d == 0 || d < -3 || d > 3; }))
    throw std::invalid_argument("Invalid bounded axial neighborhood");
  if (!directions.empty() &&
      std::any_of(receiver.begin(), receiver.end(), [&](double v) {
        double index = v * grid - .5;
        return !std::isfinite(index) || index < 0 || index >= grid ||
               index != std::floor(index);
      }))
    throw std::invalid_argument("Local neighbor checks require a voxel center");
  MissProof proof;
  proof.directions = directions;
  proof.grid = grid;
  proof.step = step;
  proof.audit = true;
  auto d = precision_ray(source, receiver, o, stats, study, offsets, &proof);
  std::map<std::string, double> info{{"complete_proof", 0}};
  if (d.status != 0 || !proof.complete || !proof.usable)
    return {d, info};

  // Normalize by the common exact global voxel spacing. All radii round
  // down; the certified support uses a strict inequality.
  auto start = Clock::now();
  MP v(bits);
  std::vector<std::array<int, 3>> covered;
  double cube;
  const auto radii =
      gradient.stencil(receiver, grid, proof.normalized, covered, &cube);
  info = {{"complete_proof", 1},
          {"leaves", double(proof.leaves)},
          {"margin_lower", proof.margin},
          {"normalized_margin_lower", proof.normalized},
          {"cube_voxels", cube},
          {"limiting_lo", proof.limiting_lo},
          {"limiting_hi", proof.limiting_hi}};
  for (int k = 0; k < 3; ++k)
    info["axis_" + std::to_string(k)] = radii[k];
  mpfr_sub(v.v, impl->tau.h.v, proof.best_lower.v, MPFR_RNDU);
  info["true_margin_upper_from_F_witnesses"] = mpfr_get_d(v.v, MPFR_RNDU);
  info["certified_receivers_in_radius4_stencil"] = 1 + covered.size();
  for (const auto &local : proof.local) {
    std::string key = "local_" + std::to_string(local.direction);
    info[key + "_FREE"] = local.certified;
    info[key + "_seconds"] = local.seconds;
    info[key + "_exp_endpoints"] = local.exps;
  }
  stats.add("miss_global_receiver_test_ns",
            std::uint64_t(elapsed(start) * 1e9));
  return {d, info};
}
} // namespace b2
