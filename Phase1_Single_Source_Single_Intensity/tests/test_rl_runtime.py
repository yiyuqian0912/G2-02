import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rind_phase1.active_rl.experiment import load_config, main
from rind_phase1.active_rl.prepare import write_json
from rind_phase1.active_rl.runtime import (
    RunLock, merge_training_limits, rank_zero_action, resume_checkpoint, training_limits,
)


class RuntimeTests(unittest.TestCase):
    def test_single_writer_lock_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'injected'):
                with RunLock(directory, 'train'):
                    with self.assertRaisesRegex(RuntimeError, 'already active'):
                        with RunLock(directory, 'evaluate'):
                            self.fail('concurrent writer acquired the same output')
                    raise ValueError('injected')
            with RunLock(directory, 'resume'):
                owner=json.loads((Path(directory)/'.active.lock').read_text())
                self.assertEqual(owner['role'],'resume')

    def test_checkpoint_priority_and_durable_extended_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            self.assertIsNone(resume_checkpoint(output))
            (output/'best.pt').touch()
            self.assertEqual(resume_checkpoint(output).name,'best.pt')
            (output/'last.pt').touch()
            self.assertEqual(resume_checkpoint(output).name,'last.pt')
            config=dict(updates=15,minimum_scene_visits=3)
            write_json(output/'training_schedule.json',training_limits(config,'protocol-1'))
            request=dict(updates=5,minimum_scene_visits=1)
            merge_training_limits(request,output,'protocol-1',dict(updates=9,minimum_scene_visits=2))
            self.assertEqual(request,config)
            with self.assertRaisesRegex(ValueError,'protocol mismatch'):
                merge_training_limits(request,output,'different')
            with self.assertRaisesRegex(ValueError,'invalid stored'):
                merge_training_limits(request,output,'protocol-1',dict(updates=-1))

    def test_root_failure_is_reported_and_streamed_json_is_identical(self):
        with self.assertRaisesRegex(RuntimeError,'rank-zero action failed: ValueError: injected'):
            rank_zero_action(lambda:(_ for _ in ()).throw(ValueError('injected')),0,1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'protocol.json';value=dict(label='观测',array=[1,.25,None])
            write_json(path,value)
            self.assertEqual(path.read_text(),json.dumps(value,indent=2,allow_nan=False)+'\n')
            with self.assertRaises(ValueError):write_json(path,dict(bad=float('nan')))
            self.assertEqual(json.loads(path.read_text()),value)

    def test_invalid_numeric_settings_and_parent_run_names(self):
        base=Path(__file__).resolve().parents[1]/'configs/active_rl_frontier16_minimum_full.json'
        args=SimpleNamespace(config=str(base),set=[],phase1_run=None,data_root=None,
                             scene_counts=None,updates=None,num_envs=None,rollout_steps=None)
        for setting in ('learning_rate=NaN','target_kl=Infinity','value_weight=-1','grad_clip=0'):
            args.set=[setting]
            with self.assertRaises(ValueError):load_config(args)
        for name in ('.','..','a/b'):
            with patch('sys.argv',['experiment','prepare','--run-name',name]):
                with self.assertRaises(SystemExit) as caught:main()
                self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
