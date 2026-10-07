"""Repeated-run aggregation rejects unmatched evaluation settings."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SeedSummaryTests(unittest.TestCase):
    def test_two_matching_seeds_aggregate_and_decoding_drift_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed, accuracy in ((11, 0.6), (22, 0.8)):
                run = root / str(seed)
                run.mkdir()
                summary = {"method": "lora", "starting_model": "base", "starting_model_revision": "sha",
                           "train_examples": 10, "validation_examples": 2}
                (run / "run_summary.json").write_text(json.dumps(summary))
                result = {"task": "emotion", "split": "test", "model": "trained_run", "run_dir": str(run),
                          "training_seed": seed, "prepared_artifacts": {"test": "sha"},
                          "sampling": {"seed": 42}, "decoding": {"temperature": 0},
                          "generation_metric": {"metric": "label_accuracy", "accuracy": accuracy, "macro_f1": accuracy},
                          "examples": [{"prompt": "p", "reference": "joy"}]}
                path = run / "evaluation.json"
                path.write_text(json.dumps(result))
                paths.append(path)
            command = [sys.executable, str(ROOT / "src/summarize_seed_runs.py"), *(str(path) for path in paths)]
            passed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertAlmostEqual(json.loads(passed.stdout)["aggregate"]["accuracy"]["mean"], 0.7)
            altered = json.loads(paths[1].read_text())
            altered["decoding"]["temperature"] = 0.7
            paths[1].write_text(json.dumps(altered))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("decoding", failed.stderr)


if __name__ == "__main__":
    unittest.main()
