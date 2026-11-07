// Included in an isolated copy of the independent reference translation unit.
// All Gaussian terms and transformed coordinates come from that reference.
namespace b2 {
namespace {

void squared(Bound &out, const Bound &x) {
  if (mpfr_sgn(x.l.v) >= 0)
    mpfr_sqr(out.l.v, x.l.v, MPFR_RNDD);
  else if (mpfr_sgn(x.h.v) <= 0)
    mpfr_sqr(out.l.v, x.h.v, MPFR_RNDD);
  else
    mpfr_set_zero(out.l.v, 0);
  mpfr_sqr(out.h.v, mpfr_cmpabs(x.l.v, x.h.v) > 0 ? x.l.v : x.h.v, MPFR_RNDU);
}

// The second operand is nonnegative. These are exact interval extrema with
// directed arithmetic, including when a positive tail has lower bound zero.
void nonnegative_product(Bound &out, const Bound &x, const Bound &y) {
  mpfr_mul(out.l.v, x.l.v, mpfr_sgn(x.l.v) >= 0 ? y.l.v : y.h.v, MPFR_RNDD);
  mpfr_mul(out.h.v, x.h.v, mpfr_sgn(x.h.v) >= 0 ? y.h.v : y.l.v, MPFR_RNDU);
}

void product(Bound &out, const Bound &x, const Bound &y, MP &tmp) {
  if (mpfr_sgn(y.l.v) >= 0) {
    nonnegative_product(out, x, y);
    return;
  }
  if (mpfr_sgn(x.l.v) >= 0) {
    nonnegative_product(out, y, x);
    return;
  }
  // Generic product is needed only for the once-per-ray dot coefficients.
  mpfr_set_inf(out.l.v, 1);
  mpfr_set_inf(out.h.v, -1);
  for (auto a : {x.l.v, x.h.v}) {
    for (auto b : {y.l.v, y.h.v}) {
      mpfr_mul(tmp.v, a, b, MPFR_RNDD);
      mpfr_min(out.l.v, out.l.v, tmp.v, MPFR_RNDD);
      mpfr_mul(tmp.v, a, b, MPFR_RNDU);
      mpfr_max(out.h.v, out.h.v, tmp.v, MPFR_RNDU);
    }
  }
}

void add(Bound &out, const Bound &value) {
  mpfr_add(out.l.v, out.l.v, value.l.v, MPFR_RNDD);
  mpfr_add(out.h.v, out.h.v, value.h.v, MPFR_RNDU);
}

void zero(Bound &x) {
  mpfr_set_zero(x.l.v, 0);
  mpfr_set_zero(x.h.v, 0);
}

bool ordered_finite(const Bound &x) {
  return mpfr_number_p(x.l.v) && mpfr_number_p(x.h.v) &&
         mpfr_cmp(x.l.v, x.h.v) <= 0;
}

struct LocalTaylor {
  std::vector<Bound> interval_terms, midpoint_terms;
  Bound fmid, dmid, derivative, curvature, u, factor, term;
  MP global{bits}, abs_derivative{bits}, h{bits}, tmp{bits}, bound{bits};
  bool prepared = false;
  int stationary_left = 4;

  explicit LocalTaylor(std::size_t kernels)
      : interval_terms(kernels), midpoint_terms(kernels) {}

  void prepare(std::vector<Line> &line, const std::vector<RefKernel> &kernels,
               Stats &stats) {
    // Interval dot products of the independently transformed ray coordinates
    // enclose the exact rational coefficients. No per-ray GMP transform.
    for (std::size_t i = 0; i < line.size(); ++i) {
      auto &l = line[i];
      for (int j = 0; j < 3; ++j) {
        squared(term, l.slope[j]);
        add(l.curvature, term);
        product(term, l.offset[j], l.slope[j], tmp);
        add(l.derivative_offset, term);
      }
      mpfr_mul_d(tmp.v, l.curvature.h.v, kernels[i].w, MPFR_RNDU);
      mpfr_mul_2ui(tmp.v, tmp.v, 1, MPFR_RNDU);
      mpfr_add(global.v, global.v, tmp.v, MPFR_RNDU);
    }
    prepared = true;
    stats.add("study_interval_coefficient_preparations");
    stats.add("study_interval_coefficient_kernels", line.size());
  }

  void slope_argument(const Line &line, double lo, double hi) {
    // a = ||v||² >= 0, t >= 0.
    mpfr_mul_d(u.l.v, line.curvature.l.v, lo, MPFR_RNDD);
    mpfr_add(u.l.v, u.l.v, line.derivative_offset.l.v, MPFR_RNDD);
    mpfr_mul_d(u.h.v, line.curvature.h.v, hi, MPFR_RNDU);
    mpfr_add(u.h.v, u.h.v, line.derivative_offset.h.v, MPFR_RNDU);
  }

  void accumulate_derivative(Bound &sum, const Bound &density) {
    mpfr_neg(factor.l.v, u.h.v, MPFR_RNDD);
    mpfr_neg(factor.h.v, u.l.v, MPFR_RNDU);
    nonnegative_product(term, factor, density);
    add(sum, term);
  }

  void midpoint_derivative(const std::vector<Line> &line, double midpoint,
                           Stats &stats) {
    zero(dmid);
    for (std::size_t i = 0; i < line.size(); ++i) {
      slope_argument(line[i], midpoint, midpoint);
      accumulate_derivative(dmid, midpoint_terms[i]);
    }
    mpfr_abs(abs_derivative.v,
             mpfr_cmpabs(dmid.l.v, dmid.h.v) >= 0 ? dmid.l.v : dmid.h.v,
             MPFR_RNDU);
    stats.add("study_midpoint_derivative_terms", line.size());
  }

  void local_derivatives(const std::vector<Line> &line, double lo, double hi,
                         Stats &stats) {
    zero(derivative);
    zero(curvature);
    for (std::size_t i = 0; i < line.size(); ++i) {
      slope_argument(line[i], lo, hi);
      accumulate_derivative(derivative, interval_terms[i]);
      squared(factor, u);
      mpfr_sub(factor.l.v, factor.l.v, line[i].curvature.h.v, MPFR_RNDD);
      mpfr_sub(factor.h.v, factor.h.v, line[i].curvature.l.v, MPFR_RNDU);
      nonnegative_product(term, factor, interval_terms[i]);
      add(curvature, term);
    }
    stats.add("study_signed_derivative_terms", line.size());
  }

  bool taylor_free(mpfr_srcptr second_upper, mpfr_srcptr tau) {
    if (!mpfr_number_p(second_upper) || !ordered_finite(fmid) ||
        !ordered_finite(dmid))
      return false;
    // Replace a negative quadratic remainder by zero for the radius bound.
    mpfr_mul(bound.v, abs_derivative.v, h.v, MPFR_RNDU);
    if (mpfr_sgn(second_upper) > 0) {
      mpfr_sqr(tmp.v, h.v, MPFR_RNDU);
      mpfr_mul(tmp.v, tmp.v, second_upper, MPFR_RNDU);
      mpfr_div_2ui(tmp.v, tmp.v, 1, MPFR_RNDU);
      mpfr_add(bound.v, bound.v, tmp.v, MPFR_RNDU);
    }
    mpfr_add(bound.v, fmid.h.v, bound.v, MPFR_RNDU);
    return mpfr_number_p(bound.v) && mpfr_cmp(bound.v, tau) < 0;
  }

  bool concave_free(mpfr_srcptr tau) {
    if (!ordered_finite(curvature) || mpfr_sgn(curvature.h.v) >= 0 ||
        !ordered_finite(fmid) || !ordered_finite(dmid))
      return false;
    // The unrestricted maximum of F + D*h + K*h²/2 (K<0) is F+D²/(-2K).
    mpfr_neg(tmp.v, curvature.h.v, MPFR_RNDD);
    mpfr_mul_2ui(tmp.v, tmp.v, 1, MPFR_RNDD);
    mpfr_sqr(bound.v, abs_derivative.v, MPFR_RNDU);
    mpfr_div(bound.v, bound.v, tmp.v, MPFR_RNDU);
    mpfr_add(bound.v, fmid.h.v, bound.v, MPFR_RNDU);
    return mpfr_number_p(bound.v) && mpfr_cmp(bound.v, tau) < 0;
  }

  template <class Eval>
  int certify(std::vector<Line> &line, const std::vector<RefKernel> &kernels,
              double lo, double mid, double hi, mpfr_srcptr tau, Eval eval,
              bool stationary, Stats &stats) {
    if (!prepared)
      prepare(line, kernels, stats);
    // Initial intervals are dyadic bisections to depth at most 50.
    mpfr_set_d(h.v, (hi - lo) / 2, MPFR_RNDN);
    midpoint_derivative(line, mid, stats);
    stats.add("study_global_taylor_tests");
    if (taylor_free(global.v, tau)) {
      stats.add("study_global_taylor_free");
      return 0;
    }
    local_derivatives(line, lo, hi, stats);
    stats.add("study_local_curvature_tests");
    if (ordered_finite(curvature)) {
      if (concave_free(tau)) {
        stats.add("study_concave_free");
        return 0;
      }
      if (taylor_free(curvature.h.v, tau)) {
        stats.add("study_local_taylor_free");
        return 0;
      }
    }
    if (ordered_finite(derivative) &&
        (mpfr_sgn(derivative.l.v) > 0 || mpfr_sgn(derivative.h.v) < 0)) {
      double end = mpfr_sgn(derivative.l.v) > 0 ? hi : lo;
      stats.add("study_monotone_endpoint_tests");
      int state = eval(end, end);
      if (state >= 0) {
        stats.add("study_monotone_decisions");
        return state;
      }
    }
    if (stationary && stationary_left > 0 && ordered_finite(dmid) &&
        ordered_finite(curvature) &&
        (mpfr_sgn(curvature.l.v) > 0 || mpfr_sgn(curvature.h.v) < 0)) {
      --stationary_left;
      stats.add("study_stationary_tests");
      // Enclose all stationary points by m - f'(m) / f''(I).
      mpfr_ui_div(factor.l.v, 1, curvature.h.v, MPFR_RNDD);
      mpfr_ui_div(factor.h.v, 1, curvature.l.v, MPFR_RNDU);
      product(term, dmid, factor, tmp);
      mpfr_d_sub(u.l.v, mid, term.h.v, MPFR_RNDD);
      mpfr_d_sub(u.h.v, mid, term.l.v, MPFR_RNDU);
      if (ordered_finite(u)) {
        double a = std::max(lo, mpfr_get_d(u.l.v, MPFR_RNDD));
        double b = std::min(hi, mpfr_get_d(u.h.v, MPFR_RNDU));
        // Width is a dispatch heuristic only; extrema are bounded by endpoints
        // and the entire outward-rounded stationary enclosure.
        if (b - a < (hi - lo) / 4) {
          stats.add("study_stationary_contractions");
          int left = eval(lo, lo), right = eval(hi, hi);
          if (left == 1 || right == 1)
            return 1;
          if (left == 0 && right == 0) {
            int state = a > b ? 0 : eval(a, b);
            if (state >= 0) {
              stats.add("study_stationary_decisions");
              return state;
            }
          }
        }
      }
    }
    stats.add("study_failed_local_bounds");
    return -1;
  }
};
} // namespace

void RayStudy::validate() const {
  if (method < 0 || method > 3 || (filter_bits != 64 && filter_bits != 128) ||
      filter_depth < 0 || filter_depth > 6 || curvature_depth < 0 ||
      curvature_depth > 50)
    throw std::invalid_argument("Invalid independent experiment settings");
}

Decision Independent::study_ray(P source, P receiver, const Options &o,
                                Stats &s, const RayStudy &study) const {
  if (!finite(source) || !finite(receiver) || impl->analytic)
    return ray(source, receiver, o, s);
  s.add("reference_ray_queries");
  const bool curvature = study.method >= 2 && o.taylor;
  bool low = study.method == 1 || study.method == 3;
  const int taylor_depth = curvature ? study.curvature_depth : o.taylor_depth;
  int precision = low ? study.filter_bits : bits;
  auto start = Clock::now();
  auto line = impl->lines(source, receiver, s, precision);
  struct Node {
    double lo, hi;
    int depth;
  };
  std::deque<Node> queue{{0, 1, 0}};
  std::unique_ptr<LocalTaylor> local;
  MP second(bits);
  bool old_derivatives = false;
  int nodes = 0, depth = 0;
  const char *reason = "reference_depth_or_precision_budget";
  auto escalate = [&]() {
    low = false;
    precision = bits;
    line = impl->lines(source, receiver, s);
    s.add("study_precision_escalations");
  };
  while (!queue.empty()) {
    if (nodes >= o.max_nodes || elapsed(start) > o.query_seconds) {
      reason = nodes >= o.max_nodes ? "reference_node_budget"
                                    : "reference_time_budget";
      break;
    }
    auto n = queue.front();
    depth = std::max(depth, n.depth);
    if (low && (n.depth > study.filter_depth || n.depth >= taylor_depth))
      escalate();
    ++nodes;
    s.add("reference_ray_nodes");
    s.maximum("reference_max_depth", depth);
    s.add(low ? "study_low_nodes" : "study_high_nodes");
    const bool local_node = curvature && !low && n.depth >= taylor_depth;
    if (local_node && !local)
      local = std::make_unique<LocalTaylor>(impl->kernels.size());
    auto eval = [&](double lo, double hi, bool tails) {
      return impl->natural(line, lo, hi, s, tails, precision);
    };
    int state = impl->natural(line, n.lo, n.hi, s, o.tail_bounds, precision,
                              local_node ? &local->interval_terms : nullptr);
    if (state >= 0) {
      queue.pop_front();
      if (state == 1)
        return {1, "independent_parameter_interval_solid", nodes, depth};
      continue;
    }
    double mid = (n.lo + n.hi) / 2;
    if (local_node) {
      state = impl->natural(line, mid, mid, s, o.tail_bounds, bits,
                            &local->midpoint_terms, &local->fmid);
      if (state < 0 && o.tail_bounds) {
        s.add("study_full_exp_midpoint_retries");
        state = impl->natural(line, mid, mid, s, false, bits,
                              &local->midpoint_terms, &local->fmid);
      }
      if (state == 0) {
        int certified = local->certify(
            line, impl->kernels, n.lo, mid, n.hi, impl->tau.l.v,
            [&](double lo, double hi) { return eval(lo, hi, o.tail_bounds); },
            study.stationary, s);
        if (certified == 0) {
          queue.pop_front();
          continue;
        }
        if (certified == 1)
          return {1, "independent_local_witness", nodes, depth};
      }
    } else if (!low && !curvature && o.taylor && n.depth >= taylor_depth) {
      // Precision-only ablation retains the old 256-bit Taylor route.
      if (!old_derivatives) {
        impl->derivatives(line, source, receiver, second, s);
        old_derivatives = true;
      }
      state = impl->taylor(line, n.lo, n.hi, second, s, o.tail_bounds);
      if (state == 0) {
        queue.pop_front();
        continue;
      }
    } else {
      state = eval(mid, mid, o.tail_bounds);
    }
    if (state == 1)
      return {1, "independent_midpoint_witness", nodes, depth};
    if (!local_node && state < 0 && o.tail_bounds) {
      s.add("study_full_exp_midpoint_retries");
      if (eval(mid, mid, false) == 1)
        return {1, "independent_full_exp_midpoint_witness", nodes, depth};
    }
    if (n.depth >= o.max_depth) {
      if (low) {
        escalate();
        continue;
      }
      if (o.tail_bounds) {
        s.add("study_final_full_exp_checks");
        state = eval(n.lo, n.hi, false);
        if (state >= 0) {
          queue.pop_front();
          if (state == 1)
            return {1, "independent_full_exp_interval", nodes, depth};
          continue;
        }
      }
      break;
    }
    queue.pop_front();
    queue.push_back({n.lo, mid, n.depth + 1});
    queue.push_back({mid, n.hi, n.depth + 1});
    s.add("study_subdivisions");
  }
  if (queue.empty())
    return {0, "independent_partition_free", nodes, depth};
  double remaining = o.query_seconds - elapsed(start);
  if (curvature && o.fallback && nodes < o.max_nodes && remaining > 0) {
    Options fallback = o;
    fallback.max_nodes -= nodes;
    fallback.query_seconds = remaining;
    s.add("study_old_independent_fallbacks");
    auto result = ray(source, receiver, fallback, s);
    s.add(result.status < 0 ? "study_old_fallback_unresolved"
                            : "study_old_fallback_decisions");
    result.nodes += nodes;
    result.depth = std::max(result.depth, depth);
    return result;
  }
  s.add("reference_unknown_rays");
  return {-1, reason, nodes, depth};
}
} // namespace b2
