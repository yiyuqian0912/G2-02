"""Post-pilot diagnostic. Intersect all proved bands in each preselected Q."""
import argparse
from fractions import Fraction
from itertools import groupby
import io
import json
from pathlib import Path
import hashlib
import numpy as np
from rind3d.geometry import load_geometry
from rind3d.rendering.polyhedra import decode_convex
from certificates import integer_row
from intersection import joint_fraction, band_cuts
from pilot import source_box


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison',type=Path)
    p.add_argument('pilot',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    manifest = (a.comparison/'tasks.json').read_bytes()
    first = json.loads(a.pilot.read_bytes())
    assert hashlib.sha256(manifest).hexdigest() == first['task_manifest_sha256']
    tasks = {t['key']:t for t in json.loads(manifest)}
    out = []
    for t in first['tasks']:
        key = t['key']
        with np.load(tasks[key]['container'],allow_pickle=False) as z:
            g = z['geometry'].tobytes()
            assert hashlib.sha256(g).hexdigest() == t['geometry_sha256']
            denom = 2*int(json.loads(z['meta'].tobytes())['resolution'])
            bodies = [tuple(integer_row(r) for r in b.rows) for b in decode_convex(load_geometry(io.BytesIO(g)))]
        with np.load(a.comparison/'labels'/(key+'_tp.npz')) as lab:
            raw_cells, raw_lower = lab['cell'],lab['lo']
        for mode,group in groupby(sorted(t['records'],key=lambda r:r['box']),key=lambda r:r['box']):
            rows = list(group)
            excluded = any(r['state'].startswith('constant') for r in rows)
            div = 16 if mode == 'parent' else 128
            cell = rows[0]['source_cell_128']
            box = source_box(cell,denom//div,denom)
            cuts = []
            hinges = set()
            for r in rows:
                if r['state'] != 'pencil':
                    continue
                c,i,j = r['chart']
                hinges.add((c,i,j))
                pair = band_cuts(bodies[c][i],bodies[c][j],Fraction(r['lower']),Fraction(r['upper']))
                if joint_fraction(box,denom,pair) != Fraction(r['fraction_exact']):
                    raise AssertionError('independent single band volume mismatch')
                cuts.extend(pair)
            v = Fraction(0) if excluded else joint_fraction(box,denom,cuts)
            selected = [source_box(x,denom//div,denom) == box for x in raw_cells]
            lower = sum(Fraction(float(l)) for l in raw_lower[selected])*div**3
            out.append(dict(key=key,box=mode,bands=len(cuts)//2,hinges=len(hinges),retained_exact=str(v),
                            retained=float(v),archived_lower=float(lower), lower_exceeds_outer=(lower>v)))
    a.output.write_text(json.dumps(out,separators=(',',':'))+'\n')


if __name__ == '__main__':
    main()
