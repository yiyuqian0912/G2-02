"""Validation, leakage, access-boundary and recovery tests for production roots.

RIND3D_TEST_ROOTS: colon-separated published roots to test (default: the canary root). Every root must contain
both profiles across the list; tests skip a profile that no root contains.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from rind3d_production import loader as LD
from rind3d_production import validate as V
from rind3d_production.env import ROOT, sha256_file
from rind3d_production.teacher import TeacherPhysics, window_centres

ROOTS = [Path(p) for p in os.environ.get("RIND3D_TEST_ROOTS", str(ROOT/"outputs"/"canary")).split(":") if p]


def datasets(**kw):
    out = []
    for r in ROOTS:
        if (r/"shards").exists():
            out.append(LD.ProductionDataset(r, **kw))
    return out


def by_profile(ds_list, short):
    for ds in ds_list:
        for i in range(len(ds)):
            if ds.scenes[int(ds.index[i][0])]["profile"].split("-")[2] == ("single" if short == "SB" else "multi"):
                return ds, i
    pytest.skip(f"no {short} scene in test roots")


def test_student_local_keys_exact():
    for ds in datasets():
        for i in np.linspace(0, len(ds)-1, 12).astype(int):
            s = ds[int(i)]
            assert set(s) == set(LD.STUDENT_KEYS)


def test_sparse_and_geometry_views_declared_keys():
    for ds in datasets(sparse=("uniform", 64), view="student_geometry"):
        s = ds[0]
        assert set(s) == LD.allowed_keys("student_geometry", sparse=True)
        free = int((~s["observed_solid"]).sum())
        assert np.isclose(s["sparse_weight"].sum(), free)
        assert np.array_equal(s["sparse_values"], s["observed_response"].ravel()[s["sparse_local_indices"]])


def test_privileged_access_refused():
    for r in ROOTS:
        with pytest.raises(LD.AccessError):
            LD.ProductionDataset(r, view="teacher")
        ds = LD.ProductionDataset(r)
        with pytest.raises(LD.AccessError):
            ds.physics(0)
        with pytest.raises(LD.AccessError):
            LD.ProductionDataset(r, sparse=("stratified", 64))[0]


def test_multi_source_observation_is_combined_and_local():
    ds, i = by_profile(datasets(view="teacher", research_access=True, subsets=("natural", "informative")), "MA")
    s = ds[i]
    total = np.zeros(s["channels"].shape[1:])
    for k, a in enumerate(s["strengths"]):
        total = total + np.where(s["channels"][k], a, 0.0)
    assert np.array_equal(total, s["observed_response"])
    stu = LD.ProductionDataset(Path(s["scene_dir"]).parents[3], subsets=("natural", "informative"))
    j = next(j for j in range(len(stu)) if stu.scenes[int(stu.index[j][0])]["scene_id"] == s["scene_id"]
             and int(stu.index[j][1]) == s["window_id"])
    obs = stu[j]["observed_response"]
    assert set(np.unique(obs)) == set(np.unique(total))          # local palette: no value absent from the window


def test_teacher_physics_matches_stored_fields():
    ds, i = by_profile(datasets(view="teacher", research_access=True), "SB")
    s = ds[i]
    phys = ds.physics(i)
    L = s["window_size"]
    rng = np.random.default_rng(0)
    free = np.flatnonzero(~s["observed_solid"].ravel())
    take = rng.choice(free, min(512, len(free)), replace=False)
    xyz = window_centres(s["window_origin"], L)[take]
    vis = phys.visibility(s["sources"], xyz)
    assert np.array_equal(vis[0] == 1, s["channels"][0].ravel()[take])
    bank = phys.receiver_bank(s["window_origin"], L, "uniform", 64, 1)
    assert np.isclose(bank["weight"].sum(), len(free))


def test_stratified_bank_weights_cover_free_receivers():
    ds, i = by_profile(datasets(view="teacher", research_access=True, subsets=("informative",)), "SB")
    s = ds[i]
    phys = ds.physics(i)
    bank = phys.receiver_bank(s["window_origin"], s["window_size"], "stratified", 64, 2, response=s["observed_response"])
    assert np.isclose(bank["weight"].sum(), int((~s["observed_solid"]).sum()))


def test_public_files_and_records_contain_no_private_fields():
    for r in ROOTS:
        for d in (r/"shards").glob("shard-*/scenes/*"):
            names = {p.name for p in (d/"public").iterdir()}
            assert names in (V.PUBLIC_SB, V.PUBLIC_MA)
            pub = (d/"public"/"scene.json").read_text().lower()
            assert not any(w in pub for w in V.PRIVATE_WORDS)
            w = np.load(d/"public"/"windows.npy")
            assert set(w.dtype.names) == {"window", "x", "y", "z", "L"}


def test_frozen_config_and_contract():
    for r in ROOTS:
        f = json.loads((r/"FROZEN.json").read_text())
        assert sha256_file(r/"config.json") == f["config_SHA256"]
        assert sha256_file(r/"contract.json") == f["contract_SHA256"]
        assert f["certificate"]["matches_current_build"]


def test_tampered_shard_is_detected(tmp_path):
    r = ROOTS[0]
    shard = sorted((r/"shards").glob("shard-*"))[0]
    copy = tmp_path/"root"/"shards"/shard.name
    shutil.copytree(shard, copy)
    target = next(copy.glob("scenes/*/public/windows.npy"))
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    man = json.loads((copy/"MANIFEST.json").read_text())
    rel = str(target.relative_to(copy))
    assert sha256_file(target) != man["files"][rel]["SHA256"]
