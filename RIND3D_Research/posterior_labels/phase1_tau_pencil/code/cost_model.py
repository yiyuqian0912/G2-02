"""Production cost projection from MEASURED per-window CPU seconds (declared rule, no fitted model).

Rule: per side, the mean measured CPU-seconds per window (engine only; verification excluded) over the declared
random validation sample; windows that did not reach the certificate are charged their measured cost (they would
fall back to an uncertified status); production mix = v0.1 quotas 85/85/86 windows of sides 64/128/256 per scene.
Reported for 100K scenes x 256 windows (25.6M) and for ~102 windows/scene (10.16M; v0.2-style)."""
import json, glob, os, sys
import numpy as np

QUOTA = {64: 85, 128: 85, 256: 86}


def load(dirs):
    R = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, '*.json')):
            if os.path.basename(f).startswith('SAMPLE') or os.path.basename(f) == 'SUMMARY.json':
                continue
            R.append(json.load(open(f)))
    return R


def project(R, label):
    out = dict(sample=label, per_side={})
    tot_per_scene = 0.0
    for side, q in QUOTA.items():
        rr = [r for r in R if r['side'] == side]
        if not rr:
            continue
        c = np.array([r['cpu_seconds'] for r in rr]); tv = np.array([r['tv_bound'] for r in rr])
        mem = [r.get('maxrss_mb') for r in rr if r.get('maxrss_mb')]
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(c, len(c)).mean() for _ in range(5000)])
        out['per_side'][side] = dict(n=len(rr), certified_tv0=int((tv == 0).sum()), cert_le_0_05=int((tv <= 0.05).sum()),
                                     cpu_mean_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                     cpu_mean=float(c.mean()), cpu_median=float(np.median(c)), cpu_p90=float(np.percentile(c, 90)),
                                     cpu_max=float(c.max()), maxrss_mb_median=float(np.median(mem)) if mem else None)
        tot_per_scene += q * c.mean()
    out['cpu_seconds_per_scene_256'] = tot_per_scene
    lo = sum(QUOTA[s] * out['per_side'][s]['cpu_mean_ci95'][0] for s in out['per_side'])
    hi = sum(QUOTA[s] * out['per_side'][s]['cpu_mean_ci95'][1] for s in out['per_side'])
    out['cpu_seconds_per_window_mix_ci95'] = [lo / 256, hi / 256]
    out['cpu_seconds_per_window_mix'] = tot_per_scene / 256
    for nscenes, wins, name in ((100_000, 256, '25.6M windows'), (100_000, 101.6, '10.16M windows')):
        core_h = nscenes * tot_per_scene * (wins / 256) / 3600
        out[name] = dict(core_hours=core_h, core_years=core_h / 8766,
                         core_hours_ci95=[nscenes * x * 256 * (wins / 256) / 3600 for x in out['cpu_seconds_per_window_mix_ci95']],
                         gap_vs_1s_target=out['cpu_seconds_per_window_mix'] / 1.0,
                         gap_vs_5s_ceiling=out['cpu_seconds_per_window_mix'] / 5.0)
    return out


def combine_b4(R, reruns_dir='outputs/real_b4_reruns'):
    """Replace b3 records that ended uncertified by their b4 rerun (same algorithm + deferred expansion)."""
    rr = {(r['scene'], r['window']): r for r in load([reruns_dir])} if os.path.isdir(reruns_dir) else {}
    return [rr.get((r['scene'], r['window']), r) if r['tv_bound'] > 0 else r for r in R]


if __name__ == '__main__':
    res = {}
    R = load(['outputs/real_val_random90_b3'])
    res['random90_b3'] = project(R, 'declared random sample, 30 val scenes x 3 sides, engine b3 as run')
    res['random90_b3_with_b4_reruns'] = project(combine_b4(R), 'same sample; the b3 windows that overflowed replaced by their engine-b4 reruns')
    json.dump(res, open('outputs/COST_MODEL.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
