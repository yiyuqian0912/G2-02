"""Metadata-only training sampling; labels and evaluation rows stay unchanged."""
from collections import Counter, defaultdict
import math

import numpy as np


def allocate(total, weights):
    if type(total) is not int or total < 1 or not weights or sum(weights) <= 0:
        raise ValueError("invalid sampling allocation")
    raw = np.asarray(weights, dtype=np.float64) * total / sum(weights)
    result = np.floor(raw).astype(np.int64)
    for i in sorted(range(len(weights)), key=lambda i: (-(raw[i] - result[i]), i))[:total - int(result.sum())]:
        result[i] += 1
    return result.tolist()


def plan(records, options):
    groups = options.get("groups", [])
    if not groups or any(set(g) != {"name", "min_layers", "max_layers", "probability"} for g in groups):
        raise ValueError("sampling.groups need name/min_layers/max_layers/probability")
    probabilities = [g["probability"] for g in groups]
    if (len({g['name'] for g in groups}) != len(groups) or
            any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probabilities) or
            not math.isclose(sum(probabilities), 1, abs_tol=1e-9)):
        raise ValueError("sampling probabilities must be finite, nonnegative and sum to one")
    for g in groups:
        if type(g['min_layers']) is not int or type(g['max_layers']) is not int or not 1 <= g['min_layers'] <= g['max_layers'] <= 32:
            raise ValueError("sampling layer range must be in [1,32]")
    total = options.get("epoch_windows", len(records))
    quotas = allocate(total, probabilities)
    pools = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(records.entries):
        r = len(row['intensities'])
        matches = [j for j, g in enumerate(groups) if g['min_layers'] <= r <= g['max_layers']]
        if len(matches) != 1:
            raise ValueError(f"sampling ranges must cover every training row once; R={r}")
        pools[matches[0]][(int(records.window_sizes[i]), int(records.candidate_counts[i]))].append(i)
    pieces, summary = [], []
    for j, (g, quota) in enumerate(zip(groups, quotas)):
        buckets = pools[j]
        if quota and not buckets:
            raise ValueError(f"no training windows in requested sampling group {g['name']}")
        sizes = sorted(buckets)
        if sizes:
            local = allocate(quota, [len(buckets[k]) for k in sizes]) if quota else [0] * len(sizes)
            for key, count in zip(sizes, local):
                if count:
                    pieces.append((key, buckets[key], count))
        summary.append({'name': g['name'], 'source_windows': sum(map(len, buckets.values())),
                        'draws_before_ddp_padding': quota, 'probability': g['probability']})
    return pieces, {'source_windows': len(records), 'epoch_draws_before_ddp_padding': total,
                    'replacement': True, 'groups': summary,
                    'size_policy': 'source native-size proportions conditional on layer group',
                    'note': 'DDP native-size tail padding can add a small number of repeated draws'}


def draw_by_bucket(records, options, rng):
    pieces, _ = plan(records, options)
    buckets = defaultdict(list)
    for key, pool, count in pieces:
        # batch_count() needs only fixed bucket sizes; do not consume epoch RNG.
        chosen = ([pool[0]] * count if rng is None else rng.choice(pool, size=count, replace=True).tolist())
        buckets[key].extend(chosen)
    return buckets
