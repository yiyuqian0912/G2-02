"""Scientific acceptance regressions for the 2b05802 independent audit findings."""
import copy
from fractions import Fraction as Q
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

import cbc_kernel
import evidence
from cbc_search import solve
import inputs
from ledger import decode
from poly import Poly
from screen2 import check_ranges
from solver import Budget, Observation
from validated_input import canonical_row, verify_solid

SCENE = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/sampling_impact/v3/scenes/0000/000004.npz")
ARCHIVE = Path(__file__).resolve().parents[1]/"global_inverse/records/heldout_raster.jsonl.gz"


def archive():
    if not SCENE.exists():
        pytest.skip("v3 fixture not installed")
    return inputs.load(SCENE,0,0)


def test_positive_rescaling_and_wire():
    obs, identity = archive()
    with gzip.open(ARCHIVE,"rt") as f:
        ref = next(r for r in map(json.loads,f) if r["task"]["key"]=="000004_w00_s0")
    assert ref["geometry_sha256"]==identity["geometry_sha256"] and ref["unknown_pieces"]==0
    assert ref["grid_status"]=="exact_normalized"
    q={int(c):Q(v) for c,v in ref["probabilities"].items()}
    expected=None
    for multiplier in (None,10**8):
        rows=obs.bodies if multiplier is None else tuple(tuple(
            tuple(Q(v*multiplier,max(map(abs,r))) for v in r) for r in b) for b in obs.bodies)
        test=Observation(rows,obs.cls,obs.origin,1024)
        wire,d=solve(test,"000004_w00_s0",Q(1,20),20)
        assert d["status"]=="CERTIFIED"
        p=decode(wire)
        assert sum(abs(p.get(i,0)-q.get(i,0)) for i in p.keys()|q.keys())/2<=Q(d["tv"])
        evidence.check_seal(d["certificate"],wire,d["tv"],Q(1,20),identity)
        for w,b,i in [(wire+b'x',d["tv"],identity),(wire,"1",identity),(wire,d["tv"],dict(identity,side=128))]:
            with pytest.raises(ArithmeticError):
                evidence.check_seal(d["certificate"],w,b,Q(1,20),i)
        if expected is not None:
            assert wire==expected
        expected=wire


def test_reject_bad_input_before_search():
    obs,_=archive()
    arr=obs.cls.copy()
    arr[0,0,30]=2
    for changed, message in [(Observation(obs.bodies,arr,obs.origin,1024),"SOLID"),
         (Observation(obs.bodies,obs.cls.astype(int)+256,obs.origin,1024),"classes"),
         (Observation(obs.bodies,obs.cls,tuple(v+.1 for v in obs.origin),1024),"integer")]:
        with pytest.raises(ValueError,match=message):
            solve(changed,"invalid",Q(1,20),1)
    with pytest.raises(ValueError):
        canonical_row((1.,0,0,0))
    with pytest.raises(ArithmeticError):
        check_ranges(np.array([[-2**63,0,0,0]],np.int64),(0,0,0),64,2048)


def test_no_alias_to_trusted_state():
    obs,_=archive()
    k=cbc_kernel.Kernel(obs,Budget(20,1000))
    for a in (k.receiver_classes(), k.proposal_problem().cls):
        assert not np.shares_memory(a,k._obs.cls) and not np.shares_memory(a,k._obs.scancls)
        a.setflags(write=True)
        a.fill(2)
    assert (k._obs.cls<2).any() and np.array_equal(k._obs.cls,k._obs.scancls.T)
    ids=np.arange(4,dtype=np.int64)
    assert not np.shares_memory(cbc_kernel._view(ids),ids)
    with pytest.raises(AttributeError):
        cbc_kernel.ProposalAccess(k).release


def test_oblique_exact_solid_rows_and_endpoint_contact():
    # Almost coincident faces on either side of a voxel centre defeat rounded rows.
    den=1<<75
    for delta in (0,1,-1):
        rows=((den*2048,den*4096,den*2048,-(20*den+delta)),)
        x,y,z=np.indices((12,12,12))
        # 2x+1 + 2(2y+1) + 2z+1 <= 20+delta/den.
        a=(2*x+4*y+2*z+4 <= (19 if delta<0 else 20)).astype(np.uint8)*2
        verify_solid(a,(0,0,0),(rows,))
        a[1,3,1]^=2
        with pytest.raises(ValueError,match="SOLID"):
            verify_solid(a,(0,0,0),(rows,))


def test_gmp_boundary_leaf():
    import sys
    sys.path.insert(0,str(Path(__file__).parent/"native"))
    native=pytest.importorskip("cbc_cells")
    p=Poly.cube()
    for i in range(3):
        p,_=p.split(tuple(256*int(j==i) for j in range(3))+(-1,))
    assert Q(*native.volume(p.faces))==Q(1,256**3)
    assert native.grid_credit(p.faces,128,160)==([(0,1<<136,1<<136)],1,0)
