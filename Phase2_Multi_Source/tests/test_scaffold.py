import contextlib
import io
import unittest
from rind_phase2.experiments import main
from rind_phase2.model import SourceSetEnergyField

class ScaffoldTests(unittest.TestCase):
    def test_check_states_not_trainable(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):self.assertEqual(main(['--check']),0)
        self.assertIn('training is NOT ready',out.getvalue())
    def test_training_is_not_silently_single_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:main([])
        self.assertEqual(e.exception.code,2)
    def test_model_is_explicitly_unimplemented(self):
        with self.assertRaises(NotImplementedError):SourceSetEnergyField()
