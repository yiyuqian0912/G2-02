#include "angular.hpp"
#include <numeric>
#include <stdexcept>

namespace b2a {
void Settings::validate() const {
  AngularOptions{angular_depth, angular_nodes, angular_references,
                 method != 2,   empty_prefix,  prefix_builds}
      .validate();
  if (method < 0 || method > 4 || angular_boxes < -1 ||
      angular_boxes > 100000 || cover_depth < 0 || cover_depth > 10 ||
      cover_nodes < 1 || cover_nodes > 1000000 || shadow_boxes < 0 ||
      shadow_boxes > 100000 || adaptive_depth < cover_depth ||
      adaptive_depth > 12 || adaptive_nodes < 0 || adaptive_nodes > 1000000 ||
      adaptive_batch < 0 || adaptive_batch > 256 || !(cover_seconds > 0) ||
      !std::isfinite(cover_seconds) || !(adaptive_seconds > 0) ||
      !std::isfinite(adaptive_seconds))
    throw std::invalid_argument("Invalid bounded research settings");
}
struct Engine::Impl {
  Inputs input;
  bool reference;
  std::unique_ptr<Primary> a;
  std::unique_ptr<Independent> b;
  double setup_seconds;
  Impl(const Inputs &i, bool independent) : input(i), reference(independent) {
    auto start = Clock::now();
    require_environment();
    if (reference)
      b = std::make_unique<Independent>(i);
    else
      a = std::make_unique<Primary>(i);
    setup_seconds = elapsed(start);
  }
  Decision box(Box q, const Options &o, Stats &s) const {
    return reference ? b->box(q.lo, q.hi, o, s) : a->box(q.lo, q.hi, o, s);
  }
  Decision point(P q, const Options &o, Stats &s) const {
    return reference ? b->point(q, o, s) : a->point(q, o, s);
  }
  Decision ray(P source, P r, const Options &o, Stats &s) const {
    return reference ? b->ray(source, r, o, s) : a->ray(source, r, o, s);
  }
  std::pair<int, int> old(P source, Box r, const Options &o, Stats &s, int y,
                          int m) const {
    return reference ? b->certificate(source, r.lo, r.hi, o, s, y, m)
                     : a->certificate(source, r.lo, r.hi, o, s, y, m);
  }
  struct Node {
    Box box;
    int label = -1, depth = 0;
    std::array<int, 8> child{};
    bool split = false, projected = false;
    std::uint64_t heat = 0;
    std::vector<Plane> planes;
  };
  struct Cover {
    std::vector<Node> nodes;
    std::vector<int> inner;
    int initial_nodes = 0;
    std::size_t initial_inner = 0;
  };
  Cover build(const Options &o, const Settings &cfg, Stats &s) const {
    Cover tree;
    Node root;
    root.box = {{0, 0, 0}, {1, 1, 1}};
    tree.nodes.push_back(root);
    std::vector<int> todo{0};
    auto start = Clock::now();
    while (!todo.empty()) {
      if (elapsed(start) > cfg.cover_seconds) {
        s.add("cover_budget_exhausted");
        break;
      }
      int id = todo.back();
      todo.pop_back();
      Node node = tree.nodes[id];
      int label = box(node.box, o, s).status;
      s.add("cover_box_queries");
      tree.nodes[id].label = label;
      if (label == 0) {
        s.add("cover_FREE_leaves");
        continue;
      }
      if (label == 1) {
        tree.inner.push_back(id);
        s.add("cover_SOLID_leaves");
        continue;
      }
      if (node.depth >= cfg.cover_depth ||
          int(tree.nodes.size() + 8) > cfg.cover_nodes) {
        s.add("cover_UNKNOWN_leaves");
        continue;
      }
      subdivide(tree, id);
      for (int child : tree.nodes[id].child)
        todo.push_back(child);
    }
    std::stable_sort(tree.inner.begin(), tree.inner.end(), [&](int x, int y) {
      return tree.nodes[x].depth < tree.nodes[y].depth;
    });
    tree.initial_nodes = tree.nodes.size();
    tree.initial_inner = tree.inner.size();
    s.add("cover_nodes", tree.nodes.size());
    s.add("cover_pending_UNKNOWN", todo.size());
    s.add("cover_all_inner", tree.inner.size());
    return tree;
  }
  static void subdivide(Cover &tree, int id) {
    Node parent = tree.nodes[id];
    tree.nodes[id].split = true;
    for (int c = 0; c < 8; ++c) {
      Node n;
      n.depth = parent.depth + 1;
      for (int j = 0; j < 3; ++j) {
        double m = (parent.box.lo[j] + parent.box.hi[j]) / 2;
        n.box.lo[j] = (c & (1 << j)) ? m : parent.box.lo[j];
        n.box.hi[j] = (c & (1 << j)) ? parent.box.hi[j] : m;
      }
      tree.nodes[id].child[c] = tree.nodes.size();
      tree.nodes.push_back(std::move(n));
    }
  }
  int mask(Cover &tree, int id, Box r, Stats &s) const {
    auto &n = tree.nodes[id];
    if (!overlaps(n.box, r))
      return -3;
    s.add("cover_membership_visits");
    if (n.label >= 0)
      return n.label;
    if (!n.split)
      return -1;
    int result = -3;
    for (int child : n.child) {
      int label = mask(tree, child, r, s);
      if (label == -3)
        continue;
      if (label < 0)
        return -1;
      if (result == -3)
        result = label;
      else if (result != label)
        return -1;
    }
    return result == -3 ? -1 : result;
  }
  void project(Cover &tree, int id, P source, Stats &s) const {
    auto &n = tree.nodes[id];
    if (n.projected)
      return;
    n.planes = enclose_planes(project_exact(source, n.box));
    n.projected = true;
    s.add("reference_projection_builds");
  }
  bool excluded(Cover &tree, int id, P source, Box r, Stats &s) const {
    auto &n = tree.nodes[id];
    if (n.label == 0)
      return true;
    s.add("cover_volume_visits");
    if (!reference)
      return !volume_possible(source, r, n.box, s);
    if (contains({source, source}, n.box))
      return false;
    project(tree, id, source, s);
    return plane_excludes(tree.nodes[id].planes, r, s);
  }
  bool free_volume(Cover &tree, int id, P source, Box r, Stats &s) const {
    if (excluded(tree, id, source, r, s))
      return true;
    if (!tree.nodes[id].split)
      return false;
    for (int child : tree.nodes[id].child)
      if (!free_volume(tree, child, source, r, s))
        return false;
    return true;
  }
  bool shadow(Cover &tree, P source, Box r, const Settings &cfg,
              Stats &s) const {
    std::vector<int> ids;
    // Static methods preserve the old coarse-first subset. Adaptive methods
    // reserve half this bounded same-body work for newly proved local solids.
    int fresh = (cfg.method == 1 || cfg.method == 4)
                    ? std::min<int>(cfg.shadow_boxes / 2,
                                    tree.inner.size() - tree.initial_inner)
                    : 0;
    int coarse = cfg.shadow_boxes - fresh;
    for (int i = 0; i < std::min<int>(coarse, tree.inner.size()); ++i)
      ids.push_back(tree.inner[i]);
    for (int i = int(tree.inner.size()) - 1;
         i >= int(tree.initial_inner) && fresh > 0; --i, --fresh)
      ids.push_back(tree.inner[i]);
    for (int id : ids) {
      s.add("inner_shadow_candidates");
      if (reference) {
        project(tree, id, source, s);
        if (plane_contains(tree.nodes[id].planes, r, s))
          return true;
      } else if (primary_same_box(source, r, tree.nodes[id].box, s))
        return true;
    }
    return false;
  }
  void candidates(Cover &tree, int id, P source, Box r, std::uint64_t receivers,
                  const Settings &cfg, Stats &s, std::vector<int> &out) const {
    if (excluded(tree, id, source, r, s))
      return;
    if (tree.nodes[id].label >= 0)
      return;
    if (tree.nodes[id].split) {
      auto kids = tree.nodes[id].child;
      for (int child : kids)
        candidates(tree, child, source, r, receivers, cfg, s, out);
      return;
    }
    auto &n = tree.nodes[id];
    n.heat += receivers;
    if (n.depth < cfg.adaptive_depth)
      out.push_back(id);
  }
  void refine(Cover &tree, P source, Box r, int side, const Options &o,
              const Settings &cfg, Stats &s) const {
    if (int(tree.nodes.size()) + 8 > tree.initial_nodes + cfg.adaptive_nodes) {
      s.add("adaptive_node_budget_exhausted");
      return;
    }
    std::vector<int> ids;
    candidates(tree, 0, source, r, std::uint64_t(side) * side * side, cfg, s,
               ids);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
      auto &x = tree.nodes[a];
      auto &y = tree.nodes[b];
      // Scheduling only: prioritize repeatedly touched, larger unresolved
      // boxes.
      auto sx = double(x.heat) * std::ldexp(1., -x.depth),
           sy = double(y.heat) * std::ldexp(1., -y.depth);
      return sx == sy ? a < b : sx > sy;
    });
    int spent = 0;
    for (int id : ids) {
      if (spent + 8 > cfg.adaptive_batch ||
          int(tree.nodes.size()) + 8 > tree.initial_nodes + cfg.adaptive_nodes)
        break;
      subdivide(tree, id);
      auto children = tree.nodes[id].child;
      spent += 8;
      s.add("adaptive_splits");
      for (int child : children) {
        int label = box(tree.nodes[child].box, o, s).status;
        tree.nodes[child].label = label;
        s.add("adaptive_box_queries");
        if (label == 1) {
          tree.inner.push_back(child);
          s.add("adaptive_new_SOLID");
        }
        if (label == 0)
          s.add("adaptive_new_FREE");
        if (label < 0)
          s.add("adaptive_remaining_UNKNOWN");
      }
    }
  }
  double prefix(Cover &tree, P source, int face, double u0, double u1,
                double v0, double v1, Stats &s) const {
    int axis = face / 2;
    double cap = face % 2 ? source[axis] : step(1 - source[axis], true);
    std::array<I, 3> d;
    d[axis] = I(face % 2 ? -1 : 1);
    d[(axis + 1) % 3] = I(u0, u1);
    d[(axis + 2) % 3] = I(v0, v1);
    double first = cap;
    std::vector<int> todo{0};
    while (!todo.empty()) {
      int id = todo.back();
      todo.pop_back();
      auto &n = tree.nodes[id];
      if (n.label == 0)
        continue;
      double low;
      s.add("empty_prefix_outer_visits");
      if (reference) {
        // Independent necessary beam intersection: exact rational linear
        // inequalities in rho, not the primary binary64 clipping routine.
        Q left = 0, right = Q(cap);
        bool possible = true;
        auto cut = [&](Q k, Q rhs) {
          if (k == 0) {
            if (rhs < 0)
              possible = false;
            return;
          }
          Q ratio = rhs / k;
          if (k > 0) {
            if (ratio < right)
              right = ratio;
          } else if (ratio > left)
            left = ratio;
        };
        for (int j = 0; j < 3; ++j) {
          cut(Q(d[j].lo), Q(n.box.hi[j]) - Q(source[j]));
          cut(-Q(d[j].hi), Q(source[j]) - Q(n.box.lo[j]));
        }
        s.add("reference_empty_prefix_exact_clips");
        if (!possible || left > right)
          continue;
        low = enclose(left).lo;
      } else if (!beam_clip(source, d, n.box, cap, low, s))
        continue;
      if (low >= first)
        continue;
      if (n.split) {
        for (int child : n.child)
          todo.push_back(child);
      } else
        first = std::min(first, low);
    }
    return first;
  }
};
Engine::Engine(const Inputs &i, bool ref) : p(std::make_shared<Impl>(i, ref)) {}
Result Engine::tile(P source, int grid, std::array<int, 3> origin, int n,
                    double seconds, const Options &o,
                    const Settings &cfg) const {
  o.validate();
  cfg.validate();
  require_environment();
  if (!finite(source) || !contains({source, source}, {{0, 0, 0}, {1, 1, 1}}) ||
      !(seconds > 0) || !std::isfinite(seconds) ||
      (n != 8 && n != 16 && n != 32 && n != 64 && n != 128) || grid < n ||
      grid > 1024 || (grid & (grid - 1)) ||
      std::any_of(origin.begin(), origin.end(),
                  [&](int x) { return x < 0 || x > grid - n; }))
    throw std::invalid_argument("Bounded field/tile required");
  Result result;
  result.setup_seconds = p->setup_seconds;
  auto &f = result.field;
  f.resolution = n;
  std::size_t total = std::size_t(n) * n * n;
  f.response.resize(total);
  f.solid.resize(total);
  f.valid.resize(total);
  if (cfg.trace) {
    result.evaluation_path.resize(total);
    result.visibility_evidence.resize(total);
    result.ray_nanoseconds.resize(total);
    result.point_nanoseconds.resize(total);
  }
  auto start = Clock::now();
  auto t0 = Clock::now();
  Settings limited = cfg;
  limited.cover_seconds = std::min(seconds, cfg.cover_seconds);
  auto tree = p->build(o, limited, f.stats);
  result.cover_seconds = elapsed(t0);
  std::unique_ptr<AngularMap> map;
  std::size_t inserted = 0;
  if (cfg.method >= 2) {
    t0 = Clock::now();
    map = std::make_unique<AngularMap>(
        source, p->reference,
        AngularOptions{cfg.angular_depth, cfg.angular_nodes,
                       cfg.angular_references, cfg.method != 2,
                       cfg.empty_prefix, cfg.prefix_builds},
        f.stats);
    for (std::size_t i = 0; i < tree.inner.size(); ++i)
      if (cfg.angular_boxes < 0 || int(i) < cfg.angular_boxes)
        map->add_inner(tree.nodes[tree.inner[i]].box);
    inserted = tree.inner.size();
    result.angular_setup_seconds = elapsed(t0);
    result.angular_seconds += result.angular_setup_seconds;
  }
  std::size_t outer_nodes = tree.nodes.size();
  auto query_angular = [&](Box r, bool leaf = false) {
    auto t = Clock::now();
    if (cfg.empty_prefix && outer_nodes != tree.nodes.size()) {
      map->outer_refined();
      outer_nodes = tree.nodes.size();
    }
    for (; inserted < tree.inner.size(); ++inserted)
      if (cfg.angular_boxes < 0 || int(inserted) < cfg.angular_boxes)
        map->add_inner(tree.nodes[tree.inner[inserted]].box);
    auto decision = map->query(
        r, [&](int face, double u0, double u1, double v0, double v1) {
          return p->prefix(tree, source, face, u0, u1, v0, v1, f.stats);
        });
    double dt = elapsed(t);
    result.angular_seconds += dt;
    if (leaf)
      result.angular_leaf_seconds += dt;
    return decision;
  };
  struct Cell {
    std::array<int, 3> origin;
    int side, y, m;
    std::uint8_t evidence;
  };
  std::vector<Cell> todo{{{0, 0, 0}, n, -1, -1, 0}};
  auto index = [n](int x, int y, int z) {
    return (std::size_t(z) * n + y) * n + x;
  };
  while (!todo.empty()) {
    if (elapsed(start) > seconds) {
      f.reason = "field_time_budget";
      break;
    }
    Cell c = todo.back();
    todo.pop_back();
    f.stats.add("field_cells");
    Box r;
    for (int j = 0; j < 3; ++j) {
      r.lo[j] = (origin[j] + c.origin[j] + .5) / grid;
      r.hi[j] = (origin[j] + c.origin[j] + c.side - .5) / grid;
    }
    int m = c.m, y = c.y;
    std::uint8_t evidence = c.evidence;
    auto ct = Clock::now();
    if (m < 0)
      m = p->mask(tree, 0, r, f.stats);
    if (m == 1)
      y = 0;
    if (y < 0) {
      if (p->free_volume(tree, 0, source, r, f.stats)) {
        y = 1;
        m = 0;
        f.stats.add("cached_whole_volume_FREE");
      } else if (p->shadow(tree, source, r, cfg, f.stats)) {
        y = 0;
        f.stats.add("cached_inner_SHADOW");
      }
    }
    if ((y < 0 || m < 0)) {
      auto old = p->old(source, r, o, f.stats, y, m);
      y = old.first;
      m = old.second;
    }
    // New proofs run only where the unchanged local certificates failed.
    if (y < 0 && map) {
      auto d = query_angular(r);
      if (d.visibility >= 0) {
        y = d.visibility;
        evidence = y == 0 ? 1 : 2;
        if (y == 1)
          m = 0;
      }
    }
    if (y < 0 && (cfg.method == 1 || cfg.method == 4) && c.side >= 4 &&
        result.adaptive_seconds < cfg.adaptive_seconds) {
      auto at = Clock::now();
      p->refine(tree, source, r, c.side, o, cfg, f.stats);
      result.adaptive_seconds += elapsed(at);
      if (p->free_volume(tree, 0, source, r, f.stats)) {
        y = 1;
        m = 0;
        evidence = 3;
      } else if (p->shadow(tree, source, r, cfg, f.stats)) {
        y = 0;
        evidence = 3;
      }
      if (y < 0 && map) {
        auto d = query_angular(r);
        if (d.visibility >= 0) {
          y = d.visibility;
          evidence = y == 0 ? 1 : 2;
          if (y == 1)
            m = 0;
        }
      }
    }
    result.region_seconds += elapsed(ct);
    if (y >= 0 && m >= 0) {
      auto at = Clock::now();
      for (int z = c.origin[2]; z < c.origin[2] + c.side; ++z)
        for (int yy = c.origin[1]; yy < c.origin[1] + c.side; ++yy)
          for (int x = c.origin[0]; x < c.origin[0] + c.side; ++x) {
            auto k = index(x, yy, z);
            f.response[k] = y;
            f.solid[k] = m;
            f.valid[k] = 1;
            if (cfg.trace) {
              result.evaluation_path[k] = m == 1 ? 3 : y == 1 ? 1 : 2;
              result.visibility_evidence[k] = evidence;
            }
          }
      f.stats.add("region_filled_voxels",
                  std::uint64_t(c.side) * c.side * c.side);
      f.stats.add("certificate_side_" + std::to_string(c.side));
      if (evidence)
        f.stats.add("new_evidence_region_voxels",
                    std::uint64_t(c.side) * c.side * c.side);
      result.assembly_seconds += elapsed(at);
      continue;
    }
    if (c.side > 2) {
      int h = c.side / 2;
      for (int z : {0, h})
        for (int yy : {0, h})
          for (int x : {0, h})
            todo.push_back(
                {{c.origin[0] + x, c.origin[1] + yy, c.origin[2] + z},
                 h,
                 y,
                 m,
                 evidence});
      continue;
    }
    auto at = Clock::now();
    for (int z = c.origin[2]; z < c.origin[2] + c.side; ++z)
      for (int yy = c.origin[1]; yy < c.origin[1] + c.side; ++yy)
        for (int x = c.origin[0]; x < c.origin[0] + c.side; ++x) {
          if (elapsed(start) > seconds) {
            f.reason = "field_time_budget";
            result.fallback_seconds += elapsed(at);
            goto done;
          }
          P v{(origin[0] + x + .5) / grid, (origin[1] + yy + .5) / grid,
              (origin[2] + z + .5) / grid};
          auto k = index(x, yy, z);
          int mask = m, visible = y;
          if (mask < 0) {
            auto pt = Clock::now();
            mask = p->point(v, o, f.stats).status;
            if (cfg.trace)
              result.point_nanoseconds[k] = std::uint64_t(elapsed(pt) * 1e9);
          }
          if (mask < 0) {
            f.stats.add("unresolved_membership_voxels");
            if (cfg.trace)
              result.evaluation_path[k] = 4;
            continue;
          }
          if (mask == 1) {
            f.solid[k] = 1;
            f.valid[k] = 1;
            if (cfg.trace)
              result.evaluation_path[k] = 5;
            continue;
          }
          std::uint8_t ev = evidence;
          if (visible < 0 && cfg.method == 0) {
            auto pt = Clock::now();
            f.stats.add("point_cover_leaf_queries");
            if (p->free_volume(tree, 0, source, {v, v}, f.stats)) {
              visible = 1;
              ev = 4;
              f.stats.add("point_cover_rays_avoided");
            }
            result.point_cover_seconds += elapsed(pt);
          }
          if (visible < 0 && map) {
            auto d = query_angular({v, v}, true);
            if (d.visibility >= 0) {
              visible = d.visibility;
              ev = visible == 0 ? 1 : 2;
              f.stats.add("angular_leaf_rays_avoided");
            }
          }
          int blocked = 1 - visible;
          if (visible < 0) {
            auto rt = Clock::now();
            blocked = p->ray(source, v, o, f.stats).status;
            if (cfg.trace)
              result.ray_nanoseconds[k] = std::uint64_t(elapsed(rt) * 1e9);
          }
          if (blocked < 0) {
            f.stats.add("unresolved_visibility_voxels");
            if (cfg.trace)
              result.evaluation_path[k] = 8;
            continue;
          }
          f.response[k] = 1 - blocked;
          f.valid[k] = 1;
          if (cfg.trace) {
            result.evaluation_path[k] =
                visible < 0 ? (blocked ? 7 : 6) : (blocked ? 10 : 9);
            result.visibility_evidence[k] = ev;
          }
        }
    result.fallback_seconds += elapsed(at);
  }
done:
  f.complete = std::all_of(f.valid.begin(), f.valid.end(),
                           [](auto v) { return v == 1; });
  if (f.reason.empty())
    f.reason = f.complete ? "complete" : "numerically_incomplete";
  if (map) {
    f.stats.add("angular_allocated_bytes_lower_bound", map->bytes());
    if (cfg.trace)
      result.angular_tiles = map->snapshot();
  }
  f.stats.add("final_cover_nodes", tree.nodes.size());
  f.stats.add("final_inner_boxes", tree.inner.size());
  f.seconds = elapsed(start);
  return result;
}
} // namespace b2a
