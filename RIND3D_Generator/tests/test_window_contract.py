import copy,json
from pathlib import Path
import numpy as np
import pytest
import yaml
from rind3d.window_contract import (
    WINDOW_SIDES,WINDOW_PROTOCOL,PILOT_SIDE_COUNTS,PRODUCTION_SIDE_COUNTS,
    require_window_side,require_side_counts,validate_window_config,
)
from rind3d.windows import Window,sample_windows
from rind3d.cli import load_config
from rind3d.pilot import config_hash,source_fingerprint,run_pilot
from rind3d.pipeline import validate_records,write_candidate_files,gate_pilot
from rind3d.search import CandidateBank


BAD_SIDES=[110,96,127,192,32,31,33,63,65,512]

@pytest.mark.parametrize('side',BAD_SIDES)
def test_non_contract_side_is_never_a_window(side):
    with pytest.raises(ValueError,match='window side'):Window((0,0,0),side)


@pytest.mark.parametrize('side',WINDOW_SIDES)
def test_only_three_exact_sides_are_window_categories(side):
    w=Window((0,0,0),side)
    assert w.stratum==WINDOW_SIDES.index(side)


@pytest.mark.parametrize('side',BAD_SIDES[:4])
@pytest.mark.parametrize('layout',['pool-offsets-v2','fixed-128-v1'])
def test_bad_side_in_canonical_records_is_rejected_even_in_diagnostics(tmp_path,side,layout):
    bank=CandidateBank();bank.add([.1,.2,.3])
    dtype,_=write_candidate_files(tmp_path,[Window((512,0,0),64)],bank,[[0]],[{}])
    records=np.fromfile(tmp_path/'window_records.bin',dtype='<u2').reshape(-1,4)
    records[0,3]=side;records.tofile(tmp_path/'window_records.bin')
    if layout=='fixed-128-v1':
        row=np.full(128,65535,dtype='<u2');row[0]=0;row.tofile(tmp_path/'candidate_ids.bin')
        (tmp_path/'candidate_offsets.bin').unlink()
    meta={'resolution':1024,'candidate_id_dtype':dtype,'candidate_storage':layout,'min_source_separation':32/1024}
    with pytest.raises(ValueError,match='window side'):validate_records(tmp_path,meta,strict=False)


@pytest.mark.parametrize('change',[
    {'window_sides':[32,64,110,256]},
    {'size_strata':[[64,95],[96,127],[128,191],[192,256]]},
    {'pilot_side_counts':{32:9,64:7,128:8,256:8}},
    {'production_side_counts':{32:1025,64:1023,128:1024,256:1024}},
    {'placement_version':'scrambled-sobol-size-strata-v1'},
])
def test_bad_configs_fail_both_cli_and_direct_sampler(tmp_path,change):
    cfg=load_config('configs/pilot.yaml');cfg.update(change)
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(p)
    with pytest.raises(ValueError):sample_windows([.1,.1,.1],cfg,0,32)


def test_quotas_balance_32_and_4096_across_three_exact_sides():
    cfg=load_config('configs/pilot.yaml')
    assert cfg['pilot_side_counts']==PILOT_SIDE_COUNTS=={64:10,128:11,256:11}
    assert cfg['production_side_counts']==PRODUCTION_SIDE_COUNTS=={64:1365,128:1365,256:1366}
    assert require_side_counts([s for s,n in PILOT_SIDE_COUNTS.items() for _ in range(n)],PILOT_SIDE_COUNTS)==PILOT_SIDE_COUNTS
    with pytest.raises(ValueError,match='quotas'):require_side_counts([64]*32,PILOT_SIDE_COUNTS)
    with pytest.raises(ValueError,match='quotas'):require_side_counts([64]*4096,PRODUCTION_SIDE_COUNTS)


def test_pilot_quota_cannot_be_bypassed_by_claiming_publication_feasible(tmp_path):
    cfg=load_config('configs/pilot.yaml');cfg['families']=[['slabs']]
    r={'format':'rind3d-pilot-v2','config_sha256':config_hash(cfg),'publication_feasible':True,
       'completed':True,'scene_count':8,'windows_per_scene':32,'window_protocol':WINDOW_PROTOCOL,
       'window_sides':list(WINDOW_SIDES),'source_sha256':source_fingerprint(),
       'scenes':[{'families':['slabs'],'capability':{'ready_capable':True},'windows':[{'size':64}]*32} for _ in range(8)]}
    p=tmp_path/'pilot.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='quotas'):gate_pilot(cfg,p)


def test_nonstandard_pilot_window_count_is_rejected_before_geometry(tmp_path):
    with pytest.raises(ValueError,match='exactly 32'):run_pilot(load_config('configs/pilot.yaml'),tmp_path/'pilot',1,31)


def test_config_sides_are_integer_categories_not_continuous_values():
    cfg=load_config('configs/pilot.yaml');cfg['window_sides']=[64.,128,256]
    with pytest.raises(ValueError,match='window side'):validate_window_config(cfg)
