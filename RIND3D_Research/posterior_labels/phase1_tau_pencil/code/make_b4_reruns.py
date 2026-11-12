"""List the windows of the b3 batches that ended uncertified (TV > 0) for a b4 rerun (deferred expansion)."""
import json, glob, os
tasks = []
for d in ('outputs/real_val_random90_b3', 'outputs/real_paired30_b3'):
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        if os.path.basename(f).startswith('SAMPLE'):
            continue
        r = json.load(open(f))
        if r['tv_bound'] > 0:
            tasks.append([r['scene'], r['window'], d, r['stats']['overflow'], r['stats']['budget_undet']])
json.dump(dict(rule='windows with b3 TV bound > 0, rerun with engine b4', tasks=[t[:2] for t in tasks], detail=tasks),
          open('outputs/tasks_b4_reruns.json', 'w'), indent=1)
print(len(tasks), tasks)
