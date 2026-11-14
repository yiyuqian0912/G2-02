"""Actual READY fixture opt-in; malformed transport unit tests need no generator."""
import json,os
from pathlib import Path
import pytest
from rind3d_reader.releases import safe_path,verify_release,pack_release,extract_release,apply_delta

@pytest.mark.parametrize('name',['/absolute','../up','a/../b','a\\b','a//b','./x',''])
def test_release_paths_reject_traversal(name):
 with pytest.raises(ValueError):safe_path(name)

def test_real_whitelist_full_delta_archive_and_default_privacy(tmp_path):
 root=os.environ.get('RIND_CAMPAIGN_TEST_ROOT')
 if not root:pytest.skip('Set RIND_CAMPAIGN_TEST_ROOT to a real READY diagnostic campaign')
 from rind3d.release_export import export_release
 from rind3d_reader import RINDDataset
 a=tmp_path/'subset-2';b=tmp_path/'delta-2-to-3';full=tmp_path/'full-3'
 export_release(root,a,'diagnostic-subset-2',count=2,diagnostic=True)
 export_release(root,b,'diagnostic-delta-3',count=3,parent=a,delta=True,diagnostic=True)
 export_release(root,full,'diagnostic-full-3',count=3,parent=a,diagnostic=True)
 r=apply_delta(a,b,tmp_path/'composed');assert r['scenes']==3
 with RINDDataset(a,split=None) as data:
  assert len(data)==512
  item=data[0];assert 'candidate_sources' not in item
  assert data.scene(item['scene_id']).counts(tuple(item['window_origin']),item['window_size']).informative
  with pytest.raises(PermissionError):data.private_oracle()
 assert not list(a.rglob('*.private.*')) and not list(a.rglob('driver.f64')) and not list(a.rglob('candidate*'))
 pack_release(a,tmp_path/'a.tar.zst');pack_release(a,tmp_path/'a-again.tar.zst')
 assert (tmp_path/'a.tar.zst').read_bytes()==(tmp_path/'a-again.tar.zst').read_bytes()
 extracted=extract_release(tmp_path/'a.tar.zst',tmp_path/'fresh');assert extracted['scenes']==2
 verify_release(Path(extracted['root']),full=True,crc=True)
 with (a/'scenes'/item['scene_id']/'window_records.bin').open('r+b') as f:f.write(b'broken')
 with pytest.raises(ValueError,match='integrity'):apply_delta(a,b,tmp_path/'must-fail')
