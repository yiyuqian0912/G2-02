"""Exact allowance for binary32 rounding followed by *exact* normalization of those rounded weights."""
from fractions import Fraction
import struct

from grid_accounting import wire_ticks
from ledger import TICKS


def audit(wire, issued_tv):
    """Consumer must first authenticate the issued wire. No guarantee for subsequent tensor arithmetic."""
    if not isinstance(issued_tv,Fraction) or not 0<=issued_tv<=Fraction(1,20):
        raise ValueError("expected issued rational TV")
    def rounded(k):
        (value,)=struct.unpack("<f",struct.pack("<f",k/TICKS))
        ticks=value*TICKS
        if int(ticks)!=ticks or ticks<=0:
            raise ArithmeticError("unexpected float32 conversion")
        return int(ticks)
    total=sum(rounded(k) for _,k in wire_ticks(wire))
    drift=Fraction(sum(abs(TICKS*rounded(k)-total*k) for _,k in wire_ticks(wire)),2*TICKS*total)
    return dict(raw_sum=Fraction(total,TICKS),renormalized_tv_change=drift,
                renormalized_tv_upper=min(1,issued_tv+drift))


def main():
    import argparse,base64,json
    from pathlib import Path
    import evidence
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/"results/manifest.json").read_bytes())
    issued=json.loads((here/"results/issued-sha256.json").read_bytes())
    results=[]
    for key,ident in manifest['tasks'].items():
        path=sorted((args.campaign/'tasks'/key).glob('*/label.json'))[-1]
        rec=evidence.read_issued(path,ident,manifest['build'],issued[key])
        v=audit(base64.b64decode(rec['wire'],validate=True),Fraction(rec['tv']))
        results.append(dict(key=key,**{k:str(value) for k,value in v.items()}))
    with args.output.open('x') as dst:
        for row in results:
            dst.write(json.dumps(row,sort_keys=True)+'\n')


if __name__=='__main__':
    main()
