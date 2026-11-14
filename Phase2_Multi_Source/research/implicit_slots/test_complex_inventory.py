import copy
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from expand_complex_masks import prepare_complex
from train_decomposition import main, train
from test_count_first import prepare_fixture


def inventory_fixture(root):
    data,base,config,index=prepare_fixture(root)
    config['output_root']=str(root/'inventory')
    config['training_expansion'].update(mode='inventory',additional_complex_windows='all',
        max_added_per_configuration=1,max_added_per_geometry=1)
    config['label_preparation']['record_counts']['train']='all'
    config['training'].pop('initial_decomposition_checkpoint',None)
    return data,base,config,index


class ComplexInventoryTests(unittest.TestCase):
    def test_full_census_not_limited_by_training_caps_or_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,config,index=inventory_fixture(root)
            source=Path(base['output_root'])
            original={p.name:p.read_bytes() for p in (source/'mask_cache').iterdir()}
            with patch('expand_complex_masks._assemble',side_effect=AssertionError('no cache assembly')):
                report=prepare_complex(config,data)
            self.assertTrue(report['scan_complete'])
            self.assertGreater(report['new_unique_complex_windows'],report['available_with_current_diversity_caps'])
            self.assertEqual(report['new_unique_complex_windows'],sum(report['new_unique_complex_by_R'].values()))
            self.assertFalse(report['training_started'])
            self.assertFalse(report['training_cache_created'])
            out=Path(config['output_root'])
            self.assertFalse((out/'mask_records.json').exists())
            self.assertFalse((out/'mask_cache').exists())
            self.assertFalse((out/'models').exists())
            catalog=[]
            for p in sorted((out/'inventory_chunks').glob('*.json.gz')):
                value=json.loads(gzip.decompress(p.read_bytes()))
                catalog.extend(value['records'])
                self.assertNotIn('response',value)
                self.assertNotIn('masks',value)
            self.assertEqual(len(catalog),report['new_unique_complex_windows'])
            self.assertEqual(len({r['configuration_window_fingerprint'] for r in catalog}),len(catalog))
            self.assertTrue(all(r['layer_count']>=3 and len(r['teacher_label_sha256'])==64 for r in catalog))
            self.assertEqual(sum(r['within_diversity_caps'] for r in catalog),report['available_with_current_diversity_caps'])
            for name,content in original.items():self.assertEqual((source/'mask_cache'/name).read_bytes(),content)
            data.calls.clear()
            self.assertEqual(prepare_complex(config),report)  # completed inventory needs no raw reread
            self.assertEqual(data.calls,[])

    def test_resume_matches_fresh_census_and_does_not_repeat_consumed_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,config,index=inventory_fixture(root)
            with self.assertRaises(InterruptedError):prepare_complex(config,data,stop_after_chunks=1)
            out=Path(config['output_root'])
            partial=json.loads((out/'inventory_report.json').read_text())
            self.assertFalse(partial['scan_complete'])
            before=len(data.calls)
            resumed=prepare_complex(config,data)
            # Fixture deliberately repeats an ROI under another view ID, so
            # compare serial counts rather than assuming different calls imply different ROIs.
            self.assertEqual(len(data.calls),resumed['scanned_new_view_ids'])
            self.assertGreater(len(data.calls),before)
            fresh=copy.deepcopy(config);fresh['output_root']=str(root/'fresh')
            data.calls.clear();reference=prepare_complex(fresh,data)
            for key in ('new_unique_complex_windows','available_with_current_diversity_caps',
                        'new_unique_complex_by_R','new_unique_complex_by_size','statistics','scanned_new_view_ids'):
                self.assertEqual(resumed[key],reference[key])

    def test_corrupt_catalogue_and_changed_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,config,index=inventory_fixture(root)
            prepare_complex(config,data)
            changed=copy.deepcopy(config);changed['training_expansion']['min_layers']=2
            with self.assertRaisesRegex(ValueError,'differs'):prepare_complex(changed,data)
            path=next((Path(config['output_root'])/'inventory_chunks').glob('*.json.gz'))
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'catalogue changed'):prepare_complex(config)

    def test_all_command_is_inventory_only_even_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,config,index=inventory_fixture(root)
            file=root/'config.json';file.write_text(json.dumps(config))
            actual=prepare_complex
            with patch('sys.argv',['train_decomposition.py','all','--config',str(file)]), \
                 patch('expand_complex_masks.prepare_complex',side_effect=lambda c:actual(c,data)), \
                 patch('train_decomposition.make_model',side_effect=AssertionError('no model')), \
                 patch('train_decomposition.train',side_effect=AssertionError('no training')):
                main()
            self.assertTrue((Path(config['output_root'])/'inventory_report.json').exists())
            with self.assertRaisesRegex(ValueError,'inventory-only'):train(config,root, {})
            with patch('sys.argv',['train_decomposition.py','train','--config',str(file)]):
                with self.assertRaisesRegex(ValueError,'inventory-only'):main()

    def test_tmux_all_becomes_prepare_and_explicit_train_is_rejected(self):
        root=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);out=directory/'output';file=directory/'inventory.json'
            file.write_text(json.dumps({'output_root':str(out),'training_expansion':{'mode':'inventory'}}))
            tmux=directory/'tmux';tmux.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n');tmux.chmod(0o755)
            env={**os.environ,'PATH':str(directory)+os.pathsep+os.environ['PATH'],
                 'RIND_PYTHON':sys.executable,'RIND_MASKS_CONFIG':str(file)}
            env.pop('RIND_TMUX_SESSION',None)
            result=subprocess.run(['bash',str(root/'run_count_masks_tmux.sh'),'start','all'],
                                  env=env,capture_output=True,text=True,check=True)
            self.assertIn('Started background inventory.',result.stdout)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_TMUX_TASK=prepare',command)
            self.assertIn('RIND_TMUX_JOB_LABEL=inventory',command)
            self.assertNotIn('--gpus',command)
            rejected=subprocess.run(['bash',str(root/'run_count_masks_tmux.sh'),'start','train'],
                                    env=env,capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('disabled',rejected.stderr)


if __name__=='__main__':unittest.main()
