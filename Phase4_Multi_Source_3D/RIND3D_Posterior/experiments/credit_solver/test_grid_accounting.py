"""Exact independent polyhedral clipping and reference wire comparisons."""
from collections import Counter
from fractions import Fraction
from itertools import product
from random import Random

import pytest
import inputs
from cbc_kernel import cell_box_rows, encode_int
from grid_accounting import CELL, dyadic_bound, half_cell, intersecting
from kernel import SCALE,wire_bound
from ledger import RECORD
from poly import Poly,within


@pytest.mark.parametrize("seed",range(40))
def test_cell_formula(seed):
    rng=Random(seed)
    for xyz in [(0,0,0),(127,127,127),tuple(rng.randrange(128) for _ in range(3))]:
        cell=within(Poly.cube(),cell_box_rows(*xyz))
        n=tuple(rng.choice((-7,-2,0,0,3,5)) for _ in range(3))
        if n==(0,0,0):
            n=(1,0,0)
        # Integer homogeneous row with its offset near this cell, including contacts.
        for shift in (0,rng.randrange(-8,9),min(0,n[0])+min(0,n[1])+min(0,n[2])):
            row=tuple(128*v for v in n)+(-sum(a*x for a,x in zip(n,xyz))-shift,)
            part=within(cell,[row])
            mass=(part.volume if part is not None else Fraction(0))*SCALE
            assert half_cell(row,*xyz)==(int(mass),-(-mass.numerator//mass.denominator))
            box=tuple(t for v in xyz for t in (v,v))
            if intersecting([row],box) is None:
                assert mass==0
            elif not intersecting([row],box):
                assert mass==CELL


@pytest.mark.parametrize("seed",range(20))
def test_dyadic_equivalence(seed):
    rng=Random(seed)
    lo=Counter({i:rng.randrange(1,1<<163) for i in range(50)})
    hi=Counter({i:m+1 for i,m in lo.items()})
    live=Counter({0:17,2:13})
    p=encode_int(lo)
    assert dyadic_bound(p,lo,hi,100,live)==wire_bound(p,lo,hi,100,live)


@pytest.mark.parametrize("p",[b"",b"x",RECORD.pack(0,float("nan")),RECORD.pack(0,0.5),
                            RECORD.pack(0,1-2**-53)+RECORD.pack(1,2**-53),
                            RECORD.pack(0,0.5)*2,RECORD.pack(128**3,1)])
def test_bad_wire(p):
    with pytest.raises(ValueError):
        dyadic_bound(p,{0:1},{0:1},0,{})



def test_deferred_snapshot_is_above_threshold_for_disjoint_single_cell_live():
    from types import SimpleNamespace
    from cbc_kernel import Kernel, Piece
    # 4096 occupied cells activate the large-support scheduling path. A live piece in
    # one of those cells exercises the subtraction of single-cell unknown mass.
    lo=Counter({i:1<<115 for i in range(4096)})
    live={i:Piece(within(Poly.cube(),cell_box_rows(0,0,i)+[(0,0,256,-(2*i+1))]))
          for i in range(3)}
    state=SimpleNamespace(live=live,lower=lo,lower_units=sum(lo.values()),live_units=None)
    obj=SimpleNamespace(_state=state,stats=Counter())
    actual=dyadic_bound(encode_int(lo),lo,lo,3*CELL//2,{i:CELL//2 for i in live})
    for tolerance in (Fraction(1,20),Fraction(1,2),Fraction(1)):
        if not Kernel.checkpoint_worthwhile(obj,tolerance):
            assert actual>tolerance
    assert not Kernel.checkpoint_worthwhile(obj,Fraction(1,20))
    assert Kernel.checkpoint_worthwhile(obj,Fraction(1))
