"""Read-only necessary check independent of physics: lower volume cannot exceed cell volume."""
import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    tasks = json.loads((a.comparison/'tasks.json').read_bytes())
    report = []
    for task in tasks:
        key = task['key']
        label = a.comparison/'labels'/f'{key}_tp.npz'
        sums = defaultdict(Fraction)
        with np.load(label,allow_pickle=False) as z:
            for cell,vol in zip(z['cell'],z['lo'],strict=True):
                sums[int(cell)] += Fraction(float(vol))*128**3
        bad = [(cid,str(v-1)) for cid,v in sums.items() if v>1]
        report.append(dict(key=key,sha256=sha256(label.read_bytes()).hexdigest(),cells=len(sums),
                           above_capacity=bad))
    a.output.write_text(json.dumps(report,separators=(',',':'))+'\n')


if __name__ == '__main__':
    main()
