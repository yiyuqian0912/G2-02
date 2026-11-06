import importlib.util
import json
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
                for name in ("run_single_source_rl.sh","run_single_source_rl_tmux.sh",
                             "Phase1_Single_Source_Single_Intensity/configs/active_rl_frontier16_minimum.json",
                             "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/frontier.py",
                             "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/trajectory_view.py",
                             "Phase1_Single_Source_Single_Intensity/docs/FRONTIER16_MINIMUM_RL.zh-CN.md",
                             "Phase1_Single_Source_Single_Intensity/configs/active_rl.json",
                             "Phase1_Single_Source_Single_Intensity/src/rind_phase1/active_rl/deploy.py",
                             "Phase1_Single_Source_Single_Intensity/docs/SINGLE_SOURCE_ACTIVE_RL.zh-CN.md",'Phase2_Multi_Source/research/implicit_slots/gpu_mask_inputs.py',
                             'Phase2_Multi_Source/research/implicit_slots/pairwise_ops.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_pairwise_memory.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_effective_assignment.py',
                             'Phase2_Multi_Source/docs/V6_RUNTIME_AUDIT.zh-CN.md',
                             'Phase2_Multi_Source/research/implicit_slots/test_gpu_mask_inputs.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_sequential_fast.py',
                             'Phase2_Multi_Source/docs/V6_BEST16_RUN.zh-CN.md',
                             'Phase2_Multi_Source/research/implicit_slots/model_comparison.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_model_comparison.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_optimizer_normalization.py',
                             'run_implicit_masks_tmux.sh',
                             'Phase2_Multi_Source/research/implicit_slots/optimization_batches.py',
                             'run_sequential_masks_tmux.sh',
                             'Phase2_Multi_Source/configs/masks_sequential.json',
                             'Phase2_Multi_Source/research/implicit_slots/sequential_count_shape.py',
                             'Phase2_Multi_Source/research/implicit_slots/sequential_objective.py',
                             'Phase2_Multi_Source/research/implicit_slots/evaluate_sequential.py',
                             'Phase2_Multi_Source/research/implicit_slots/configure_sequential.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_sequential.py',
                             'Phase2_Multi_Source/docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md',
                             'run_failure_diagnostics_tmux.sh',
                             'Phase2_Multi_Source/docs/FAILURE_DIAGNOSTICS.zh-CN.md',
                             'Phase2_Multi_Source/research/implicit_slots/failure_diagnostics.py',
                             'Phase2_Multi_Source/research/implicit_slots/evaluation_snapshot.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_failure_diagnostics.py',
                             'Phase2_Multi_Source/research/implicit_slots/gpu_runtime.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_gpu_runtime.py',
                             'run_disentangled_masks_tmux.sh',
                             'Phase2_Multi_Source/configs/masks_disentangled.json',
                             'Phase2_Multi_Source/docs/DISENTANGLED_TRAINING.zh-CN.md',
                             'Phase2_Multi_Source/research/implicit_slots/disentangled_model.py',
                             'Phase2_Multi_Source/research/implicit_slots/disentangled_objective.py',
                             'Phase2_Multi_Source/research/implicit_slots/inventory_selection.py',
                             'Phase2_Multi_Source/research/implicit_slots/configure_disentangled.py',
                             'Phase2_Multi_Source/research/implicit_slots/inventory_complex.py',
                             'Phase2_Multi_Source/research/implicit_slots/test_complex_inventory.py',
                             'Phase2_Multi_Source/docs/COMPLEX_WINDOW_INVENTORY.zh-CN.md',
                             'Phase2_Multi_Source/configs/masks_count_first_train_template.json',
                             'run_count_masks.sh', 'run_count_masks_tmux.sh',
                             'Phase2_Multi_Source/configs/masks_count_first.json',
                             'Phase2_Multi_Source/research/implicit_slots/count_objective.py',
                             'Phase2_Multi_Source/research/implicit_slots/expand_complex_masks.py',
                             'run_active_observation.sh',
                             'Phase2_Multi_Source/research/active_observation/demo.py',
                             'Phase2_Multi_Source/docs/ACTIVE_OBSERVATION.zh-CN.md',
                             'run_masks_expanded_tmux.sh', 'run_decomposition_union_tmux.sh', 'run_slot_search_tmux.sh',
                             'run_refined_masks_tmux.sh', 'run_refined_masks.sh',
                             'run_balanced_masks_tmux.sh', 'run_balanced_masks.sh',
                             'Phase2_Multi_Source/configs/masks_balanced_finetune.json',
                             'Phase2_Multi_Source/research/implicit_slots/extend_evaluation.py',
                             'Phase2_Multi_Source/research/implicit_slots/boundary_refined_model.py',
                             'Phase2_Multi_Source/configs/masks_boundary_refined.json',
                             'Phase2_Multi_Source/configs/slot_search.json',
                             'Phase2_Multi_Source/research/implicit_slots/search_cache.py',
                             'Phase2_Multi_Source/research/implicit_slots/train_decomposition.py',
                             'Phase1_Single_Source_Single_Intensity/part_e/evaluate.py',
                             'Phase1_Single_Source_Single_Intensity/vendor/rind-dataset/pyproject.toml'):
                    self.assertIn(module.PREFIX + '/' + name, names)
                v5 = json.load(tar.extractfile(module.PREFIX + '/Phase2_Multi_Source/configs/masks_disentangled.json'))
                self.assertEqual(v5['model']['decomposition_architecture'],'disentangled_slots_v5')
                self.assertEqual(v5['loss']['objective'],'disentangled_v5')
                self.assertNotIn('training_expansion',v5)
                default = json.load(tar.extractfile(module.PREFIX + '/Phase2_Multi_Source/configs/masks_count_first.json'))
                self.assertEqual(default['training_expansion']['mode'], 'inventory')
                self.assertEqual(default['training_expansion']['additional_complex_windows'], 'all')
                template = json.load(tar.extractfile(module.PREFIX + '/Phase2_Multi_Source/configs/masks_count_first_train_template.json'))
                self.assertNotEqual(template['training_expansion'].get('mode'), 'inventory')
                for name in names:
                    self.assertFalse(set(Path(name).parts) & {'data', 'outputs', '.venv', '__pycache__', '.git', 'phase2_slot_update', 'server_bundle', 'pilot'})


if __name__ == '__main__':
    unittest.main()
