import copy,gzip,json,os,shlex,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from test_count_first import prepare_fixture
from expand_complex_masks import prepare_complex
from inventory_selection import select_catalogue

ROOT=Path(__file__).resolve().parents[3]

class InventoryTrainingTests(unittest.TestCase):
    def test_explicit_catalogue_quota_freezes_eval_and_checks_relabelled_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,base,new,old=prepare_fixture(root)
            inv=copy.deepcopy(new);inv['output_root']=str(root/'inventory')
            inv['training_expansion'].update(mode='inventory',additional_complex_windows='all')
            inv['label_preparation']['record_counts']['train']='all'
            report=prepare_complex(inv,data)
            self.assertGreaterEqual(report['available_with_current_diversity_caps'],2)
            new['training_expansion']['inventory_root']=inv['output_root']
            rows,sig=select_catalogue(Path(inv['output_root']),old,new['training_expansion'],2)
            with self.assertRaisesRegex(ValueError,'inventory has'):
                select_catalogue(Path(inv['output_root']),old,new['training_expansion'],100)
            data.calls.clear()
            index=prepare_complex(new,data)
            self.assertEqual(len(data.calls),2)  # only chosen windows, no full rescan
            self.assertEqual(index['records']['validation'],old['records']['validation'])
            self.assertEqual(index['records']['test'],old['records']['test'])
            self.assertEqual([(r['scene_id'],r['view_id']) for r in index['records']['train'][2:]],
                             [(r['scene_id'],r['view_id']) for r in rows])
            protocol=json.loads((Path(new['output_root'])/'complex_expansion_protocol.json').read_text())
            self.assertEqual(protocol['inventory_signature'],sig)
            # Changing a local response after a completed census must not
            # silently substitute another decomposition under the same identity.
            new['output_root']=str(root/'bad')
            get=data.get_region
            def changed(*args):
                value=get(*args);value['response']=value['response']*.8;return value
            data.get_region=changed
            with self.assertRaisesRegex(ValueError,'differs from inventory'):prepare_complex(new,data)

    def test_tmux_runs_offline_records_command_and_does_not_change_pointer_on_failed_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'results';config=root/'config.json'
            config.write_text(json.dumps({'output_root':str(out)}))
            fake=root/'tmux';fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit "${MOCK_SESSION_RUNNING:-1}";fi\nexit 0\n');fake.chmod(0o755)
            env={**os.environ,'PATH':str(root)+os.pathsep+os.environ['PATH'],'RIND_PYTHON':sys.executable,
                 'RIND_MASKS_CONFIG':str(config),'RIND_TMUX_SESSION':'rind_v5_test'}
            # Mock the outer pointer into this temporary directory so testing
            # never creates user experiment state under the source tree.
            import shutil
            frontend=root/'run_disentangled_masks_tmux.sh'
            text=(ROOT/'run_disentangled_masks_tmux.sh').read_text().replace('ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',f'ROOT={shlex.quote(str(ROOT))}')
            text=text.replace('pointer="$ROOT/Phase2_Multi_Source/outputs/disentangled_v5_launcher/$session.txt"',f'pointer={shlex.quote(str(root/"pointer"))}')
            frontend.write_text(text)
            result=subprocess.run(['bash',str(frontend),'start','all','--gpus','6'],env=env,capture_output=True,text=True,check=True)
            self.assertIn('Started background training.',result.stdout)
            run=Path((out/'tmux_runs/latest_run.txt').read_text().strip())
            command=shlex.split((run/'command.txt').read_text())
            self.assertIn('RIND_MASKS_CONFIG='+str(config.resolve()),command);self.assertIn('RIND_TMUX_TASK=all',command)
            self.assertEqual(command[-2:],['--gpus','6'])
            (root/'pointer').write_text('keep-this')
            env['MOCK_SESSION_RUNNING']='0'
            result=subprocess.run(['bash',str(frontend),'start','all'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((root/'pointer').read_text(),'keep-this')

if __name__=='__main__':unittest.main()
