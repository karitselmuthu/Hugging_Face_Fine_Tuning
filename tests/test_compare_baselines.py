"""Offline checks that label comparisons include matching base-model results."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "src/compare_label_evaluations.py"


class ComparisonBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.first_dir = self.directory / "first"
        self.second_dir = self.directory / "second"
        self.first_dir.mkdir()
        self.second_dir.mkdir()
        self.first = self.write_result(self.first_dir, "test_evaluation.json", "trained_run", 0.8)
        self.first_base = self.write_result(self.first_dir, "base_test_evaluation.json", "starting_model", 0.2)
        self.second = self.write_result(self.second_dir, "test_evaluation.json", "trained_run", 0.9)
        self.second_base = self.write_result(self.second_dir, "base_test_evaluation.json", "starting_model", 0.3)

    def write_result(self, run_dir, filename, model, accuracy, prompt="same case"):
        result = {
            "task": "emotion_classification", "run_dir": str(run_dir), "model": model,
            "split": "test", "run_manifest_schema": 1,
            "prepared_artifacts": {"test.jsonl": {"sha256": "same", "rows": 1}},
            "decoding": {"temperature": 0, "max_new_tokens": 8},
            "examples": [{"prompt": prompt, "reference": "joy", "output": "joy"}],
            "generation_metric": {
                "metric": "label_accuracy", "accuracy": accuracy, "macro_f1": accuracy,
                "invalid_label_outputs": 0,
                "per_label": {"joy": {"support": 1, "recall": accuracy, "f1": accuracy}},
            },
        }
        path = run_dir / filename
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def compare(self, *args):
        return subprocess.run([sys.executable, str(COMPARE), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True)

    def test_two_trained_runs_include_both_baselines_and_lift(self):
        result = self.compare(self.first, self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("First baseline", result.stdout)
        self.assertIn("Second baseline", result.stdout)
        self.assertIn("lift over base: accuracy=+0.600", result.stdout)

    def test_missing_baseline_blocks_comparison(self):
        self.second_base.unlink()
        result = self.compare(self.first, self.second)
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing base-model baseline", result.stderr)

    def test_baseline_on_different_cases_blocks_comparison(self):
        self.write_result(self.second_dir, self.second_base.name, "starting_model", 0.3, prompt="other case")
        result = self.compare(self.first, self.second)
        self.assertEqual(result.returncode, 2)
        self.assertIn("same generated cases", result.stderr)

    def test_base_and_trained_result_from_same_run(self):
        custom_base = self.write_result(self.first_dir, "custom_base.json", "starting_model", 0.2)
        custom_trained = self.write_result(self.first_dir, "custom_trained.json", "trained_run", 0.8)
        result = self.compare(custom_base, custom_trained)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("baseline:"), 1)


if __name__ == "__main__":
    unittest.main()
