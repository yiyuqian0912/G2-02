"""Exact sparse-training regressions. Real-corpus tests use explicit environment paths."""
from collections import defaultdict
from fractions import Fraction as Q
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys

import numpy as np
import pytest
import torch

import decoder as d
from release_gate import collect

HERE = Path(__file__).resolve().parent
BOUND = Q(1,20)
PLAN = "0f1fa5c3df22f32085cd25f44a73db66dc475d9af8490830c0bea657be833f3d"
INDEX = "13a0dacfe3141dc551c355ac51b9b1c743f292fac4fd98510d44d193ea1baaef"
ISSUED = HERE.parents[1]/"RIND3D_Posterior/experiments/credit_solver/assurance/results/issued-sha256.json"


def wire(entries):
    return b"".join(struct.pack("<Id", i, v) for i, v in entries)


def reference(t):
    masses = defaultdict(Q)
    for i, val in zip(t.cells, t.weights, strict=True):
        masses[int(i)] += Q.from_float(float(val))
    return dict(masses)


@pytest.mark.parametrize("size", [1,2,3,17,513,1025])
@pytest.mark.parametrize("mode", ["binary64","float32_terms"])
def test_exact_target_and_ownership(size, mode):
    rng = random.Random(size)
    cuts = [0, *sorted(rng.sample(range(1,d.TICKS),size-1)), d.TICKS]
    ids = sorted(rng.sample(range(d.CELLS),size))
    p = {i:Q(b-a,d.TICKS) for i,a,b in zip(ids,cuts[:-1],cuts[1:],strict=True)}
    t = d._decode(wire(p.items()), BOUND, mode)
    assert t.tv_bound == BOUND and reference(t) == p and sum(p.values()) == 1
    for a in (t.cells,t.weights):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    assert len(t.cells) <= 3*size


@pytest.mark.parametrize("bad", [
    [(0,0)], [(0,float("nan"))], [(0,float("inf"))], [(0,-1.)],
    [(d.CELLS,1)], [(1,.5),(0,.5)], [(0,.5),(0,.5)],
    [(0,2**-53),(1,1)], [(0,.25)], [(0,1.01)],
    [(i,1.) for i in range(4097)], # naive signed int64 sums wrap to TICKS
], ids=["zero","nan","infinite","negative","address","order","duplicate","half-tick","short-sum","large","wrap"])
def test_invalid_wire(bad):
    with pytest.raises(ValueError):
        d._decode(wire(bad), BOUND, "float32_terms")


def test_minimal_rounding_counterexample_and_boundary_ids():
    p = {0:Q(1,d.TICKS),d.CELLS-1:Q(d.TICKS-1,d.TICKS)}
    assert sum(Q(float(np.float32(v))) for v in p.values()) == 1 + Q(1,d.TICKS)
    t = d._decode(wire(p.items()), BOUND, "float32_terms")
    assert reference(t) == p and len(t.cells) == 4
    for bad in (b"", wire(p.items())[:-1]):
        with pytest.raises(ValueError):
            d._decode(bad, BOUND, "binary64")


def test_loss_coefficients_and_omitted_infinite_logloss():
    p = {1:Q(1,d.TICKS),123:Q(d.TICKS-1,d.TICKS)}
    targets = [d._decode(wire(p.items()), BOUND, "float32_terms")]
    logp = torch.full((1,d.CELLS),-torch.inf,dtype=torch.float64)
    logp[0,1],logp[0,123] = -1.,-2.
    logp.requires_grad_()
    loss = d.sparse_cross_entropy(logp, targets)
    assert loss.item() == float(2*p[123]+p[1])
    loss.backward()
    assert torch.count_nonzero(logp.grad)==2
    for i,mass in p.items():
        assert Q(logp.grad[0,i].item()) == -mass


def reader(mode):
    if not (os.environ.get("RIND_CBC_CAMPAIGN") and os.environ.get("RIND_V3_SCENES")):
        pytest.skip("set RIND_CBC_CAMPAIGN and RIND_V3_SCENES for the pinned real-data checks")
    return d.IssuedExamples(os.environ["RIND_CBC_CAMPAIGN"], os.environ["RIND_V3_SCENES"], ISSUED,
                            manifest_sha256=PLAN, index_sha256=INDEX, representation=mode)


def collate(batch):
    obs, targets = zip(*batch)
    assert all(set(x)=={"solid","response","origin","L"} for x in obs)
    return d.torch_batch(targets)


def test_real_workers_and_pilot_loss():
    from torch.utils.data import DataLoader, Subset
    sys.path.insert(0,str(HERE.parent))
    from pilot.student import full_logmass, target_ce
    class Head:
        def fine_logits(self, emb):
            return emb * torch.linspace(-1.,1.,512)
    rd = reader("float32_terms")
    ix = [rd.keys.index(k) for k in ("000004_w00_s0","000000_w01_s0")]
    # Two independent worker reads; batch extent stays bounded for large windows.
    batches = list(DataLoader(Subset(rd,ix), batch_size=1, num_workers=2, timeout=15, collate_fn=collate))
    for batch,i in zip(batches,ix,strict=True):
        obs,t = rd[i]
        assert t.weights.dtype == np.float32 and set(obs)=={"solid","response","origin","L"}
        assert all(torch.equal(a,b) for a,b in zip(batch,d.torch_batch([t]),strict=True))
        # Test existing hierarchical CE consumer against independent dense-logit reference.
        head = Head()
        cl = torch.linspace(-1.,2.,4096)[None,:]
        out = cl,cl[:,:,None]
        origin,side = torch.tensor([obs["origin"]]),torch.tensor([obs["L"]])
        actual = target_ce(head,out,origin,side,*batch)
        logs = full_logmass(head,out,origin,side)
        _, target64 = reader("binary64")[i]
        expected = d.sparse_cross_entropy(logs, [target64])
        torch.testing.assert_close(actual.double(),expected,atol=3e-5,rtol=2e-6)


def test_authentication_rejects_self_consistent_replacement(tmp_path):
    original = reader("binary64")
    index = json.loads(ISSUED.read_bytes())
    key = original.keys[0]
    index[key] = "0"*64
    changed = tmp_path/"index.json"
    changed.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        d.IssuedExamples(original._root, original._scenes, changed, manifest_sha256=PLAN, index_sha256=INDEX)
    new = d.IssuedExamples(original._root, original._scenes, changed, manifest_sha256=PLAN,
                           index_sha256=hashlib.sha256(changed.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match="authorized label"):
        new[key]
    identity = original._plan["tasks"][key].copy()
    identity["observation_xyz_sha256"] = "0"*64
    with pytest.raises(ValueError,match="local observation"):
        original._observation(identity)


def test_incomplete_shard_never_earns_index(tmp_path):
    key = "000009_w00_s0"
    ident = dict(slot=9,window=0,channel=0)
    build = dict(backend="test")
    plan = dict(tasks={key:ident},build=build,backend="test",schema="cbc-campaign/1",tolerance="1/20")
    raw = json.dumps(plan).encode()
    (tmp_path/"manifest.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    attempt = tmp_path/"tasks"/key/"0001"
    attempt.mkdir(parents=True)
    (attempt/"start.json").write_text(json.dumps(dict(key=key,manifest_sha=sha)))
    (attempt/"receipt.json").write_text('{"status":"INTERRUPTED"}')
    with pytest.raises(ValueError,match="unissued channel"):
        collect(tmp_path, sha, build, None)
    from release_gate import export
    with (tmp_path/"owner.lock").open("wb") as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            export(tmp_path,tmp_path/"issued.json",sha,build,None)
    assert not (tmp_path/"issued.json").exists()
