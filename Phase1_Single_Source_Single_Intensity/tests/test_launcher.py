import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from scripts.launch_training import configure,parser,PROJECT


class LauncherTests(unittest.TestCase):
    def test_package_root_wrapper_routes_to_phase1_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            wrapper=root/'run_phase1_100k.sh'
            shutil.copy2(PROJECT/'run_phase1_100k.sh',wrapper)
            phase1=root/'Phase1_Single_Source_Single_Intensity'
            (phase1/'src/rind_phase1').mkdir(parents=True)
            capture=root/'captured.txt'
            (phase1/'run_training.sh').write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{}"\n'.format(capture))
            (phase1/'run_training.sh').chmod(0o755)
            # A misleading package-root runner reproduces the original layout.
            (root/'run_training.sh').write_text('#!/usr/bin/env bash\nexit 91\n')
            (root/'run_training.sh').chmod(0o755)
            result=subprocess.run([str(wrapper),'prepare','--gpus','8'],cwd=root)
            self.assertEqual(result.returncode,0)
            arguments=capture.read_text().splitlines()
            self.assertIn('phase1_100k_multiscale_h20_memmap',arguments)
            self.assertEqual(arguments[arguments.index('--stage')+1],'prepare')
            self.assertEqual(arguments[arguments.index('--gpus')+1],'8')

    def test_custom_parameters_and_frozen_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'configs').mkdir()
            (root/'configs/experiment_round1.json').write_bytes((PROJECT/'configs/experiment_round1.json').read_bytes())
            args=parser().parse_args(['--run-name','custom','--epochs','50','--seeds','0',
                '--set','training.weight_decay=0.002','--set','teacher.boundary_lambda=0'])
            path,c,out=configure(args,root)
            self.assertEqual(c['training']['epochs'],50)
            self.assertEqual(c['training']['weight_decay'],.002)
            self.assertEqual(c['teacher']['boundary_lambda'],0)
            self.assertFalse(path.exists())
            path.write_text(json.dumps(c))
            out.mkdir(parents=True)
            (out/'protocol.json').write_text('{}')
            self.assertEqual(configure(parser().parse_args(['--run-name','custom']),root)[1],c)
            with self.assertRaisesRegex(ValueError,'new --run-name'):
                configure(parser().parse_args(['--run-name','custom','--epochs','60']),root)

    def test_invalid_overrides(self):
        for value in ('training.typo=1','training.batch_size=0','training.epochs=1.5',
                      'training.learning_rate=NaN','training.seeds=[]','name="replacement"'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                configure(parser().parse_args(['--run-name','invalid_check','--set',value]))


if __name__=='__main__': unittest.main()
