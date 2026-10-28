import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("package_update", Path(__file__).with_name("package_update.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackageTests(unittest.TestCase):
    def test_canonical_source_only_package_has_current_and_full_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = module.build(Path(temporary) / "RIND_Phase2_Update.tar.gz")
            with tarfile.open(output) as tar:
                names = set(tar.getnames())
                for name in ('run_masks_expanded_tmux.sh', 'run_decomposition_union_tmux.sh',
                             'Phase2_Multi_Source/research/implicit_slots/train_decomposition.py',
                             'Phase1_Single_Source_Single_Intensity/part_e/evaluate.py',
                             'Phase1_Single_Source_Single_Intensity/vendor/rind-dataset/pyproject.toml'):
                    self.assertIn(module.PREFIX + '/' + name, names)
                for name in names:
                    self.assertFalse(set(Path(name).parts) & {'data', 'outputs', '.venv', '__pycache__', '.git', 'phase2_slot_update', 'server_bundle', 'pilot'})


if __name__ == '__main__':
    unittest.main()
