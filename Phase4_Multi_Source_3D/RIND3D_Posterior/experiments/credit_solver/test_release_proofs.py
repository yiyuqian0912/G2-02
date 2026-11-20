import copy
from fractions import Fraction as Q
from random import Random
import pytest

import inputs
import cbc_kernel as ck
from grid_accounting import CELL,half_cell,intersecting,coefficients
from kernel import SCALE
from poly import Poly,within
from solver import Budget
from collections import Counter
from assurance.exact_reference import volume
import evidence


@pytest.mark.parametrize("seed",range(32))
def test_nonshared_reference(seed):
    rng=Random(seed)
    xyz=[rng.randrange(6,121) for _ in range(3)]
    i,j,k=xyz
    cell=ck.cell_box_rows(*xyz)
    n=tuple(rng.choice((-7,0,1,3)) for _ in xyz)
    if not any(n): n=(1,0,0)
    # Negative/zero coefficients, large cancellation, grazing a corner, nontrivial tick residues.
    big=(1<<257)+1
    row=(*[x*128*big for x in n],-sum(x*y for x,y in zip(xyz,n))*big+rng.choice((-1,0,1)))
    expected=volume(cell+[row])*SCALE
    assert half_cell(row,*xyz)==(int(expected),-(-expected.numerator//expected.denominator))
    if seed%4==0:
        r=(128*big,128*big,0,-(i+j+1)*big+1)
        s=(0,-128*big,128*big,-(k-j)*big+1)
        assert len(intersecting([r,s],(i,i,j,j,k,k)))==2
        upper=[(128,0,0,-(i+2)),(-128,0,0,i),(0,128,0,-(j+2)),(0,-128,0,j),(0,0,128,-(k+2)),(0,0,-128,k)]
        poly=within(Poly.cube(),upper+[r,s])
        assert poly is not None
        kernel=ck.Kernel.__new__(ck.Kernel)
        kernel.budget=Budget(30,4096)
        kernel.stats=Counter()
        lo,hi=kernel._grid_credit([poly])
        for a in range(i,i+2):
            for b in range(j,j+2):
                for c in range(k,k+2):
                    val=volume(ck.cell_box_rows(a,b,c)+[r,s])*SCALE
                    cid=ck.cell_id((a,b,c))
                    assert (lo[cid],hi[cid])==(int(val),-(-val.numerator//val.denominator))


def test_problem_binding_and_release_ceiling():
    with pytest.raises(ValueError):
        ck.certify(None,Q(1,19),1,1,None)
    st=dict(problem={},build=dict(kernel=evidence.VERSION,backend="python-integer-fraction"),wire_format=evidence.WIRE,
            tv="0",tolerance="1/20",wire_sha256=evidence.digest(b""))
    binding=dict(statement=st,statement_sha256=evidence.digest(evidence.canonical(st)))
    with pytest.raises(ArithmeticError):
        evidence.check_seal(binding,b"","0",Q(1,20),{})
    assert type(coefficients((2,0,1))[-1]) is tuple
