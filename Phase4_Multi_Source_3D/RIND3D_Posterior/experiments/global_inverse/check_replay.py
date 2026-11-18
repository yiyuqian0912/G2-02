"""Compare two runs' closed, positive-volume answers independently of timing."""
import argparse
from hashlib import sha256
import json
from query import load_records


def solved(rows):
    return {r['task']['key']:r for r in rows if r['positive'] and r['unknown_pieces']==0}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('original','replay'): ap.add_argument(name)
    a=ap.parse_args()
    original=solved(load_records(a.original))
    repeated=solved(load_records(a.replay))
    if not original or original.keys()!=repeated.keys():
        raise ValueError('different solved task sets')
    for k in original:
        def digest(row):
            subset={key:row.get(key) for key in ('task','geometry_sha256','inner_pieces','probabilities',
                                                   'known_lower','outer_upper','scale_bits','grid_status')}
            return sha256(json.dumps(subset,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        expected=digest(original[k])
        if expected!=digest(repeated[k]): raise ArithmeticError(k)
        print(k,expected)
