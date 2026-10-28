"""Validate packaged shell entry points without occupying any GPU or tmux job."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


PACKAGE = Path(__file__).resolve().parents[3]


class MasksLauncherTests(unittest.TestCase):
    def test_tmux_forwards_config_and_tracks_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="rind launch ") as temporary:
            root = Path(temporary)
            config = root / "custom config.json"
            output = root / "custom output"
            config.write_text(json.dumps({"output_root": str(output)}))
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "tmux"
            fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "RIND_PYTHON": sys.executable, "RIND_MASKS_CONFIG": str(config)}
            result = subprocess.run(["bash", str(PACKAGE / "run_masks_tmux.sh"), "start", "all", "--gpus", "8"],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn("rind_masks_only", result.stdout)
            run = Path((output / "tmux_runs/latest_run.txt").read_text().strip())
            command = shlex.split((run / "command.txt").read_text())
            self.assertIn(f"RIND_MASKS_CONFIG={config.resolve()}", command)
            self.assertIn(f"RIND_TMUX_RUNNER={PACKAGE / 'run_masks.sh'}", command)
            self.assertIn("RIND_TMUX_TASK=all", command)
            self.assertEqual(command[-2:], ["--gpus", "8"])

    def test_gpu_selection_and_separate_evaluation_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "bin"
            binary.mkdir()
            smi = binary / "nvidia-smi"
            smi.write_text("#!/usr/bin/env bash\nprintf '0, 0, 0\\n1, 20000, 99\\n2, 0, 0\\n'\n")
            smi.chmod(0o755)
            python = binary / "python-shim"
            python.write_text('#!/usr/bin/env bash\nif [[ "$1" == -c ]]; then exec "$REAL_PYTHON" "$@"; fi\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@" > "$GPU_DISPATCH"\n')
            python.chmod(0o755)
            capture = root / "arguments"
            env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "REAL_PYTHON": sys.executable, "RIND_PYTHON": str(python), "GPU_DISPATCH": str(capture)}
            for task, script in (("train", "train_decomposition.py"), ("evaluate", "evaluate_masks.py")):
                result = subprocess.run(["bash", str(PACKAGE / "run_masks.sh"), task, "--gpus", "8"],
                                        env=env, text=True, capture_output=True, check=True)
                lines = capture.read_text().splitlines()
                self.assertEqual(lines[0], "0,2")
                self.assertIn("--nproc-per-node=2", lines)
                self.assertIn("research/implicit_slots/" + script, lines)
                self.assertIn("Selected physical GPUs: 0,2", result.stdout)

    def test_invalid_config_stops_before_gpu_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "invalid.json"
            config.write_text('{"task":"decompositon_only"}')
            env = {**os.environ, "RIND_PYTHON": sys.executable, "RIND_MASKS_CONFIG": str(config)}
            result = subprocess.run(["bash", str(PACKAGE / "run_masks.sh"), "train", "--gpus", "8"],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("task must", result.stderr)
            self.assertNotIn("Selected physical GPUs", result.stdout)

    def test_expanded_tmux_keeps_a_separate_session_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "expanded.json"
            output = root / "output"
            config.write_text(json.dumps({"output_root": str(output)}))
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "tmux"
            fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == has-session ]]; then exit 1; fi\nexit 0\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
                   "RIND_PYTHON": sys.executable, "RIND_MASKS_CONFIG": str(config)}
            env.pop("RIND_TMUX_SESSION", None)
            result = subprocess.run(["bash", str(PACKAGE / "run_masks_expanded_tmux.sh"), "start", "all", "--gpus", "auto"],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn("rind_masks_expanded", result.stdout)
            self.assertIn("./run_masks_expanded_tmux.sh status", result.stdout)
            run = Path((output / "tmux_runs/latest_run.txt").read_text().strip())
            command = shlex.split((run / "command.txt").read_text())
            self.assertIn(f"RIND_MASKS_CONFIG={config.resolve()}", command)
            self.assertIn("RIND_TMUX_TASK=all", command)


if __name__ == "__main__":
    unittest.main()
