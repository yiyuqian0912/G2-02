"""Replay the frozen 256-window demo, retaining both physical implementations."""
from collections import Counter
import json
from pathlib import Path
import time
import resource
import numpy as np
from rind3d.geometry import Oracle, load_geometry, Budget
from rind3d.windows import Window
from rind3d.pack import RINDPack, atomic_json, sha256_file
from rind3d.information import InformationIndex
from rind3d.cli import load_config
from rind3d.stored_windows import select_stored_windows
from rind3d.search import verify_grid_exact
from experiments.shadow_volumes.renderer import SharedGeometry
from experiments.shadow_volumes.cases import SOURCES

ROOT = Path('outputs/cpu-cuda-efficiency')
OLD = Path('outputs/shadow-volume-experiment')


def main():
    start = time.perf_counter(); cpu = time.process_time()
    t = time.perf_counter(); base = Oracle(load_geometry(OLD/'cases/box/geometry.npz'))
    init = time.perf_counter()-t
    t = time.perf_counter(); a = SharedGeometry(base).bind(SOURCES[0]); prep = time.perf_counter()-t
    path = OLD/'lite-compatibility/scenes/scene_box_source0'
    records = np.fromfile(path/'window_records.bin', dtype='<u2').reshape(-1, 4)
    windows = [Window(tuple(map(int, r[:3])), int(r[3])) for r in records]
    result = {'geometry_initialization_seconds': init, 'shadow_preparation_seconds': prep,
              'candidate_search_seconds': 0., 'candidate_search_note': 'Not run in the historical demonstration',
              'frozen_records_sha256': sha256_file(path/'window_records.bin'),
              'historical_full_field': json.loads((OLD/'full/box/C-source0/result.json').read_text()),
              'windows': [], 'complete': False}
    with RINDPack(path/'response.rindb', verify_file=True) as response, RINDPack(path/'solid.rindb', verify_file=True) as solid:
        t = time.perf_counter(); index = InformationIndex.build(response, solid)
        result['index_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        cfg = load_config('configs/pilot.yaml')
        selection = select_stored_windows(index, SOURCES[0], 256, seed=20261031, scene=0, settings=cfg['window_selection'])
        result['selection_seconds'] = time.perf_counter()-t
        assert [w.record() for w in selection.windows] == [w.record() for w in windows]
        for i, w in enumerate(windows):
            row = {'window': i, 'origin': list(w.origin), 'side': w.size}
            t = time.perf_counter(); observed = response.read_window(*w.origin, w.size)
            row['read_seconds'] = time.perf_counter()-t
            for key, active in [('true_source_verification', a), ('independent_recheck', base)]:
                budget = Budget(120, 100_000_000); before = dict(active.metrics)
                t = time.perf_counter(); v = verify_grid_exact(active, SOURCES[0], SOURCES[0], w, 1024, budget, observed=observed)
                row[key+'_seconds'] = time.perf_counter()-t
                row[key+'_points'] = budget.points
                row[key+'_metrics'] = {k: x-before.get(k, 0) for k, x in active.metrics.items()}
                assert v['status'] == 'grid_exact' and v['covered_voxels'] == w.size**3
            result['windows'].append(row)
            if i % 16 == 0: atomic_json(ROOT/'lite-profile-progress.json', result)
    result['complete'] = True
    result['total_wall_seconds_with_setup'] = time.perf_counter()-start
    result['total_cpu_seconds_with_setup'] = time.process_time()-cpu
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result['totals'] = {key: sum(r[key] for r in result['windows']) for key in
                        ('read_seconds', 'true_source_verification_seconds', 'independent_recheck_seconds',
                         'true_source_verification_points', 'independent_recheck_points')}
    result['logical_voxels_per_pass'] = sum(w.size**3 for w in windows)
    bricks = Counter()
    for w in windows:
        for z in range(w.origin[2]//64, (w.origin[2]+w.size-1)//64+1):
            for y in range(w.origin[1]//64, (w.origin[1]+w.size-1)//64+1):
                for x in range(w.origin[0]//64, (w.origin[0]+w.size-1)//64+1): bricks[(x,y,z)] += 1
    result['brick_reuse'] = {'distinct_bricks': len(bricks), 'window_brick_references': sum(bricks.values()),
                            'maximum_reuse': max(bricks.values()), 'all_counts': list(bricks.values())}
    atomic_json(ROOT/'lite-profile.json', result)


if __name__ == '__main__': main()
