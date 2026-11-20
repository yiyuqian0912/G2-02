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


def test_training_conversion_requires_new_bound():
    from assurance.training_conversion import audit
    from ledger import TICKS,RECORD
    raw=RECORD.pack(0,(TICKS//2+1)/TICKS)+RECORD.pack(1,(TICKS//2-1)/TICKS)
    r=audit(raw,Q(1,20))
    assert r['raw_sum']==1 and r['renormalized_tv_change']==Q(1,TICKS)
    assert r['renormalized_tv_upper']>Q(1,20)


def test_resealed_and_truncated_artifacts(tmp_path):
    import json
    from test_validated import archive
    from cbc_search import solve
    from run import write_result
    obs, ident=archive()
    wire,d=solve(obs,"000004_w00_s0",Q(1,20),20)
    path=tmp_path/"a"
    write_result(path,ident,Q(1,20),wire,d)
    raw=path.read_bytes()
    digest=evidence.digest(raw)
    build=json.loads(evidence._build_bytes())
    rec=evidence.read_issued(path,ident,build,digest)
    assert Q(rec['tv'])>0
    rec['tv']="0"
    cert=rec['certificate']
    cert['statement']['tv']="0"
    cert['statement_sha256']=evidence.digest(evidence.canonical(cert['statement']))
    rec['artifact_sha256']=evidence.artifact_digest(ident,cert)
    # Internal consistency is insufficient for issuer authentication.
    evidence.check_artifact(rec,ident,expected_build=build,materialize=False)
    for bad in (evidence.canonical(rec),raw[:-8]):
        path.write_bytes(bad)
        with pytest.raises(ValueError,match="issuance digest"):
            evidence.read_issued(path,ident,build,digest)


def test_credit_update_cancellation_and_cached_snapshot(monkeypatch):
    from test_validated import archive
    from dataclasses import replace
    obs,_=archive()
    kr=ck.Kernel(obs,Budget(30,5000))
    old=kr._state
    with monkeypatch.context() as m:
        def interrupt(*a):
            raise KeyboardInterrupt()
        m.setattr(ck,"State",interrupt)
        with pytest.raises(KeyboardInterrupt):
            kr.split(0,(2,0,0,-1))
    assert kr._state is old and kr.handles()==(0,)
    # Fault injection into the private cache simulates an implementation bug.
    kr._state=replace(old,lower=Counter({0:1}),upper_in=Counter({0:1}),lower_units=1,live_units=0)
    assert kr.release(Q(1,20)).status=="ERROR"
    assert kr._best is None


def test_noncanonical_geometry_dtype_rejected():
    import io,numpy as np
    from test_validated import SCENE
    if not SCENE.exists():
        pytest.skip('requires installed fixture')
    with np.load(SCENE,allow_pickle=False) as scene:
        raw=scene["geometry"].tobytes()
    with np.load(io.BytesIO(raw),allow_pickle=False) as geom:
        arrays={k:geom[k] for k in geom.files}
    key=next(k for k,v in arrays.items() if v.dtype.kind=='u' and v.dtype.itemsize<=2 and v.size)
    arrays[key]=arrays[key].astype(np.int64)
    stream=io.BytesIO()
    np.savez(stream,**arrays)
    with pytest.raises((ValueError,TypeError)):
        inputs._geometry_rows(stream.getvalue())


def test_diffuse_allocation_end_to_end_small_exact_measure():
    from ledger import decode
    from grid_accounting import dyadic_bound
    # Analogue of the ledger after all receiver and clearance obligations passed.
    # Source body has a skew cut, a mixture of empty/full/partial grid cells, and
    # an unresolved slab with unknown feasible subset (specified only to the reference).
    rows=[(-128,0,0,4),(128,0,0,-7),(0,-128,0,4),(0,128,0,-7),
          (0,0,-128,4),(0,0,128,-7),(128,256,128,-23)]
    cut=(777,0,0,-36)
    p=within(Poly.cube(),rows)
    settled,live=p.split(cut)
    assert settled is not None and live is not None
    k=ck.Kernel.__new__(ck.Kernel)
    k.stats=Counter()
    k.budget=Budget(30,4096)
    lo,hi=k._grid_credit([settled])
    charge=ck.ceil_mass(live)
    wire=ck.encode_int(lo)
    bound=dyadic_bound(wire,lo,hi,charge,{})
    exact={}
    for x in range(4,7):
        for y in range(4,7):
            for z in range(4,7):
                box=ck.cell_box_rows(x,y,z)
                cid=ck.cell_id((x,y,z))
                mass=volume(rows+[cut]+box)*SCALE
                assert lo[cid]<=mass<=hi[cid] and hi[cid]-lo[cid]<=1
                # Only part of the live slab is truly feasible.
                exact[cid]=volume(rows+[(1554,0,0,-73)]+box)
    norm=sum(exact.values())
    dist=decode(wire)
    actual=sum(abs(dist.get(i,0)-m/norm) for i,m in exact.items())/2
    assert sum(dist.values())==1 and 0<actual<=bound<1


def test_write_interruption_preserves_committed_bytes(tmp_path,monkeypatch):
    import run
    from test_validated import archive
    from cbc_search import solve
    obs,ident=archive()
    wire,detail=solve(obs,"000004_w00_s0",Q(1,20),10)
    out=tmp_path/"posterior"
    run.write_result(out,ident,Q(1,20),wire,detail)
    before=out.read_bytes()
    def stop(*args):
        raise OSError("injected before atomic replacement")
    monkeypatch.setattr(run.os,"replace",stop)
    with pytest.raises(OSError):
        run.write_result(out,ident,Q(1,20),wire,detail)
    assert out.read_bytes()==before and list(tmp_path.iterdir())==[out]


def test_population_verification_fails_on_missing_or_duplicate_issuance(tmp_path):
    import json
    from assurance.verify_release import verify
    from test_validated import archive
    from cbc_search import solve
    from run import write_result
    obs,identity=archive()
    key="000004_w00_s0"
    plan=dict(tasks={key:identity},build=json.loads(evidence._build_bytes()))
    pin=evidence.digest(evidence.canonical(plan))
    task=tmp_path/"tasks"/key
    folder=task/"001"
    folder.mkdir(parents=True)
    wire,details=solve(obs,key,Q(1,20),12)
    write_result(folder/"label.json",identity,Q(1,20),wire,details)
    sha=evidence.digest((folder/"label.json").read_bytes())
    (folder/"start.json").write_text(json.dumps(dict(key=key,manifest_sha=pin)))
    receipt=dict(status='CERTIFIED',sha256=sha)
    (folder/"receipt.json").write_text(json.dumps(receipt))
    assert verify(tmp_path,plan,{key:sha},pin)==1
    with pytest.raises(ValueError,match='population'):
        verify(tmp_path,plan,{},pin)
    later=task/"002"
    later.mkdir()
    (later/"start.json").write_bytes((folder/"start.json").read_bytes())
    (later/"receipt.json").write_bytes((folder/"receipt.json").read_bytes())
    with pytest.raises(ValueError,match='duplicate'):
        verify(tmp_path,plan,{key:sha},pin)
