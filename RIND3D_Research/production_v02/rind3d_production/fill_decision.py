"""Fill the canary (§10) and scale (§11) sections of RIND3D_PRODUCTION_DECISION.md from the result JSON files."""
import json

from .env import ROOT


def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}g}" if isinstance(x, float) else str(x))


def main():
    res = json.loads((ROOT/"RIND3D_PRODUCTION_CANARY_RESULTS.json").read_text())
    cost = json.loads((ROOT/"RIND3D_LIFECYCLE_COST_MODEL.json").read_text())
    g = res["gates"]
    m = cost["measured"]
    per = m["per_profile"]
    lines = []
    lines += [f"**Verdict: canary {'ACCEPTED' if res['accepted'] else 'NOT ACCEPTED'}.** The frozen gates C1–C8 were evaluated "
              "mechanically by `canary_analysis.py` against the frozen contract (`outputs/canary/contract.json`).", ""]
    lines += ["| Gate | Result | Evidence |", "|---|---|---|"]
    lines.append(f"| C1 yield | {'pass' if g['C1_yield']['pass'] else 'FAIL'} | {g['C1_yield']['published']}/{g['C1_yield']['attempted']} slots published; unpublished: {g['C1_yield']['unpublished_with_reason'] or 'none'} |")
    lines.append(f"| C2 validation | {'pass' if g['C2_validation']['pass'] else 'FAIL'} | independent validator: {g['C2_validation']['scenes']} scenes, all checks {g['C2_validation']['passed_counts']} |")
    lines.append(f"| C3 physics | {'pass' if g['C3_physics']['pass'] else 'FAIL'} | {g['C3_physics']['replayed_bricks']} bricks replayed by both independent predicates at generation, all bits equal; validation-time independent re-render of 4 random bricks per source equal |")
    c4 = g["C4_windows"]
    lines.append(f"| C4 windows | {'pass' if c4['pass'] else 'FAIL'} | natural 100% filled: {c4['natural_100pct_filled']}; informative shortfall {f(c4['informative_shortfall_fraction'])}; natural all-shadow SB windows: {c4['natural_all_shadow_SB_any_side']}; per side {json.dumps(c4['coverage_per_side'])} |")
    lines.append(f"| C5 leakage | {'pass' if g['C5_leakage']['pass'] else 'FAIL'} | tests: {g['C5_leakage']['tests']}; public/private separation {g['C5_leakage']['V9_public_private']}; MA local palettes {g['C5_leakage']['V8_multi_local_palette']} scenes |")
    c6 = g["C6_recovery"]
    lines.append(f"| C6 recovery | {'pass' if c6['pass'] else 'FAIL'} | SIGKILL of runner + worker at {c6['interruption']['time'] if c6['interruption'] else '–'} during slot {c6['interruption']['inflight']['slot'] if c6['interruption'] else '–'}; partial attempt moved aside: {c6['partial_attempts_moved_aside']}; duplicate sealed slots: {c6['duplicate_sealed_slots'] or 'none'}; determinism re-runs identical: {[d['identical'] for d in c6['determinism']]} |")
    c7 = g["C7_resources"]
    lines.append(f"| C7 resources | {'pass' if c7['pass'] else 'FAIL'} | max scene wall {f(c7['max_scene_wall_seconds'])} s; max worker RSS {f(c7['max_worker_RSS_MiB'])} MiB; max GPU memory in use {f(c7['max_gpu_memory_used_MiB'])} MiB; canary wall {f(c7['canary_wall_seconds'])} s |")
    c8 = g["C8_usability"]
    lines.append(f"| C8 usability | {'pass' if c8['pass'] else 'FAIL'} | SB 64^3 student random access {f(c8['student_SB_L64_windows_per_second'])} windows/s; DataLoader {c8['dataloader']}; Teacher queries equal stored: {c8['teacher_equal_stored']} |")
    sc = res["scenes"]
    lines += ["", f"**Scenes.** {sc['published']} published: families {sc['families']}; profiles {sc['profiles']}; splits {sc['splits']}; "
              f"{sc['windows_total']} window records; {sc['bytes_total']/1e9:.2f} GB.", ""]
    lines += ["| Profile | Scenes | Wall s median (p95) | Bytes median | Stage medians (s) |", "|---|---|---|---|---|"]
    for k, v in per.items():
        lines.append(f"| {k} | {v['scenes']} | {f(v['wall_seconds']['median'])} ({f(v['wall_seconds']['p95'])}) | {v['bytes']['median']/1e6:.2f} MB | "
                     + ", ".join(f"{s} {f(x, 2)}" for s, x in v["stages_median_seconds"].items()) + " |")
    od = res["observation_distribution"]
    lines += ["", f"**Natural observation distribution** [M]: {od['natural_categories']}. SB natural lit-ratio quantiles {od['natural_SB_lit_ratio_quantiles']}. "
              f"Subset shortfalls {od['subset_shortfalls']} of quotas {od['subset_quotas']}. MA non-natural windows with ≥1/≥2/≥3/all informative sources: "
              f"{od['multi_source_non_natural_windows']}.", ""]
    dv = res["diversity"]
    lines += [f"**Diversity** [M]: SOLID fraction quantiles {dv['all']['solid_quantiles']} (max {f(dv['all']['solid_max'])}); per family medians: "
              + "; ".join(f"{fam} solid {f(v['solid']['median'])}, components {f(v['components']['median'])}" for fam, v in dv.items() if fam != "all") + ".", ""]
    th = res["throughput"]
    lines += [f"**Throughput** [M]: {f(th['scenes_per_hour'])} published scenes/hour over all sessions (wall {f(th['wall_seconds_all_sessions'])} s, "
              f"killed session counted to its last journal event — a lower bound); median scene {f(th['per_scene_wall_median'])} s."]
    canary = "\n".join(lines)
    proj = cost["projections"]
    s = ["All figures below are **[X]** extrapolations from the measured canary rate and mix on one RTX 3060 laptop; they assume linear scaling.", "",
         "| Scale | Producer hours (1 laptop GPU) | ~4 GPU workers | Storage | Files | Window records | Validation CPU-hours (1 process) |", "|---|---|---|---|---|---|---|"]
    for k, v in proj.items():
        s.append(f"| {int(k):,} | {f(v['producer_hours'])} | {f(v['with_4_gpu_workers_hours_estimate'])} | {f(v['storage_GB'])} GB | {int(v['files']):,} | {int(v['windows']):,} | {f(v['validation_cpu_hours_single_process'])} |")
    s += ["", "**Failure risks.**",
          "- Yield and recovery are measured only at 216 scenes.",
          "- Multi-source cost grows roughly linearly in K.",
          "- Storage is dominated by the multi-source public observation payloads and the private per-source fields.",
          "- One producer GPU is a single point of throughput, not of integrity: a crash costs at most the in-flight scene.",
          "- **Before any public release:**",
          "  - the master seed must be private (`seed_file`, as in v0.1's HMAC scheme), otherwise hidden labels are regenerable from code plus seed;",
          "  - a release must not distribute the private `src*/` channels.",
          "", "**Next approved scale: 1K. Commands are provided, not executed:**", "",
          "```bash", "cd /home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Production",
          "# 1. implement/verify private seed-file support (runner.load_cfg) and write a private 256-bit seed file",
          "./run.sh rind3d_production.runner create /data/rind3d-v0.2-1k configs/production-1k-v0.2.json",
          "./run.sh rind3d_production.runner run /data/rind3d-v0.2-1k --max-runtime 43200",
          "./run.sh rind3d_production.runner status /data/rind3d-v0.2-1k",
          "./run.sh rind3d_production.validate /data/rind3d-v0.2-1k --determinism 7 501",
          "RIND3D_TEST_ROOTS=/data/rind3d-v0.2-1k ./run.sh pytest -q -p no:cacheprovider tests", "```", "",
          "10K and 100K would use the same commands with a new versioned root and `slots`. They must not be launched without a separate authorization and an approved storage and compute budget."]
    doc = (ROOT/"RIND3D_PRODUCTION_DECISION.md").read_text()
    doc = doc.replace("PLACEHOLDER_CANARY", canary).replace("PLACEHOLDER_SCALE", "\n".join(s))
    (ROOT/"RIND3D_PRODUCTION_DECISION.md").write_text(doc)
    print("filled")


if __name__ == "__main__":
    main()
