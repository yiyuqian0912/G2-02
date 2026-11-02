import json,os
from pathlib import Path
import numpy as np
import pytest,yaml
from rind3d.cli import load_config,main
from rind3d.enrichment_cli import run
from rind3d.enrichment_research import read_artifact
from rind3d.pack import sha256_file

def test_core_research_configs_reject_silent_unknown_semantics(tmp_path):
    c=yaml.safe_load(Path('configs/lite-core-research-v1.yaml').read_text());c['geometry_mixture']='unknown';p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='geometry mixture'):load_config(p)
    with pytest.raises(SystemExit):main(['lite-pilot','--config','configs/lite-core-research-v1.yaml','--output',str(tmp_path/'not-created')])
    assert not (tmp_path/'not-created').exists()

def test_external_zero_budget_enrichment_is_private_immutable_and_read_only(tmp_path):
    root=Path(os.environ.get('RIND_LITE_TEST_ROOT','outputs/lite-v02-research/core-fast-mixed-100'))
    if not (root/'scenes').exists():pytest.skip('Requires the actual bounded READY_LITE pilot')
    scene=next((root/'scenes').iterdir());before={p.name:sha256_file(p) for p in scene.iterdir() if p.is_file()}
    c=yaml.safe_load(Path('configs/ambiguity-enrichment-research-v1.yaml').read_text());c['window_budget_seconds']=0;config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(c));output=tmp_path/'v1.private.json'
    r=run(scene,[0],config,output);assert r['additional_references']==0 and not r['core_scene_modified']
    sources=read_artifact(output,scene);truth=np.fromfile(scene/'driver.f64',dtype='<f8');assert np.array_equal(sources[0],truth[None])
    with pytest.raises(FileExistsError):run(scene,[0],config,output)
    assert before=={p.name:sha256_file(p) for p in scene.iterdir() if p.is_file()}
