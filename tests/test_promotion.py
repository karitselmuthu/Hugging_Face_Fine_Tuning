"""Promotion policy checks on matching test cases."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promote_run import evaluate_promotion  # noqa: E402
from task_config import load_task  # noqa: E402


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.task = load_task(ROOT / "tasks/emotion_classification.json")
        self.trained = self.result("trained_run", accuracy=0.75, macro_f1=0.60)
        self.base = self.result("starting_model", accuracy=0.40, macro_f1=0.30)

    def result(self, model, accuracy, macro_f1):
        return {"task": self.task["name"], "run_dir": str(self.run_dir), "model": model,
                "split": "test", "prepared_artifacts": {"test": "same"},
                "sampling": {"seed": 42}, "decoding": {"temperature": 0},
                "examples": [{"prompt": f"p-{index}", "reference": "joy"} for index in range(128)],
                "generation_metric": {"metric": "label_accuracy", "accuracy": accuracy,
                                      "macro_f1": macro_f1, "invalid_label_outputs": 0}}

    def test_example_thresholds_pass_with_base_lift(self):
        self.assertEqual(evaluate_promotion(self.task, self.trained, self.base, self.run_dir), [])

    def test_low_score_and_invalid_output_reject(self):
        self.trained["generation_metric"].update(accuracy=0.60, macro_f1=0.50, invalid_label_outputs=1)
        failures = evaluate_promotion(self.task, self.trained, self.base, self.run_dir)
        self.assertGreaterEqual(len(failures), 3)

    def test_mismatched_cases_reject(self):
        self.base["examples"][0]["prompt"] = "other"
        with self.assertRaisesRegex(ValueError, "same nonempty cases"):
            evaluate_promotion(self.task, self.trained, self.base, self.run_dir)


if __name__ == "__main__":
    unittest.main()
