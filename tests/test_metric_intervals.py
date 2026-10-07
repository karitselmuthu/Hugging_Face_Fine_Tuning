"""Held-out intervals are deterministic and contain the observed score."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task_metrics import bootstrap_intervals, score_generations  # noqa: E402


class MetricIntervalTests(unittest.TestCase):
    def test_label_interval_is_deterministic(self):
        task = {"evaluation": {"metric": "label_accuracy", "labels": ["joy", "sadness"]}}
        examples = [{"reference": "joy", "output": "joy"},
                    {"reference": "sadness", "output": "sadness"},
                    {"reference": "joy", "output": "sadness"},
                    {"reference": "sadness", "output": "joy"}]
        result = bootstrap_intervals(task, examples, 42, resamples=100)
        self.assertEqual(result, bootstrap_intervals(task, examples, 42, resamples=100))
        point = score_generations(task, examples)
        for key in ("accuracy", "macro_f1"):
            self.assertLessEqual(result["intervals"][key]["lower"], point[key])
            self.assertGreaterEqual(result["intervals"][key]["upper"], point[key])


if __name__ == "__main__":
    unittest.main()
