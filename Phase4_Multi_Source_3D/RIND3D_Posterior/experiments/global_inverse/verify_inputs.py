"""Freeze or verify the public geometry and observation actually used by each task."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from research import read_obs


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('receipts',type=Path)
    ap.add_argument('--write',action='store_true')
    args=ap.parse_args()
    first=json.loads(Path('tasks.json').read_bytes())
    fresh=json.loads(Path('fresh.json').read_bytes())['holdout']
    groups=[first['archive'],first['holdout'],fresh]
    used=set()
    seen_geometry=set()
    receipts={}
    for group in groups:
        slots={t['slot'] for t in group}
        if slots & used: raise ValueError('overlapping geometry slots across splits')
        used.update(slots)
        ids=set()
        for t in group:
            obs,geom=read_obs(args.root,t)
            ids.add(geom)
            receipts[t['key']]=dict(geometry=geom,origin=obs.origin,resolution=obs.denom//2,
                                    side=obs.side,classes_xyz=sha256(obs.cls.tobytes()).hexdigest())
        if ids & seen_geometry or len(ids)!=len(slots):
            raise ValueError('canonical geometry identity repeated across slots or splits')
        seen_geometry.update(ids)
    data=(json.dumps(receipts,indent=1,sort_keys=True)+'\n').encode()
    if args.write:
        with args.receipts.open('xb') as f: f.write(data)
    elif args.receipts.read_bytes()!=data:
        raise ValueError('public geometry/observations differ')
    print(len(receipts),'inputs;',len(used),'distinct geometry slots;',sha256(data).hexdigest())
