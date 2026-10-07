"""Small fixtures for sensitive-data, near-duplicate, and length audits."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_task_data import audit_prepared_data  # noqa: E402
from data_integrity import snapshot_prepared_data  # noqa: E402
from data_quality import audit_sequence_lengths  # noqa: E402


class FakeTokenizer:
    eos_token = " <eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class DataQualityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        task = {"evaluation": {"metric": "label_accuracy", "labels": ["joy", "sadness"]}}
        (self.folder / "task_config.json").write_text(json.dumps(task))

    def write_splits(self, train, validation, test):
        for split, rows in (("train", train), ("validation", validation), ("test", test)):
            (self.folder / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.folder / "manifest.json").write_text(json.dumps({"artifacts": snapshot_prepared_data(self.folder)}))

    def row(self, text, response="joy"):
        return {"prompt": f"Classify: {text}", "response": response, "inputs": {"text": text}}

    def test_near_duplicate_and_label_distribution(self):
        self.write_splits(
            [self.row("I feel joyful after seeing my friends today")],
            [self.row("I feel joyful after seeing my friends tonight", "sadness")],
            [self.row("Something different happened")],
        )
        report = audit_prepared_data(self.folder)
        self.assertEqual(report["quality"]["near_duplicate_pairs"]["train_validation"], 1)
        self.assertEqual(report["quality"]["label_distribution"]["validation"]["sadness"], 1)
        with self.assertRaisesRegex(ValueError, "Near-duplicate"):
            audit_prepared_data(self.folder, fail_on_near_duplicate=True)

    def test_pii_is_reported_and_secrets_blocked(self):
        self.write_splits([self.row("Contact alice@example.com")],
                          [self.row("Another case")], [self.row("A third case")])
        report = audit_prepared_data(self.folder)
        self.assertEqual(report["quality"]["sensitive"]["train"]["pii_rows_by_type"]["email"], 1)
        self.assertNotIn("alice@example.com", json.dumps(report))
        with self.assertRaisesRegex(ValueError, "PII patterns"):
            audit_prepared_data(self.folder, fail_on_pii=True)
        self.write_splits([self.row("Token hf_" + "A" * 32)],
                          [self.row("Another case")], [self.row("A third case")])
        with self.assertRaisesRegex(ValueError, "Secret patterns"):
            audit_prepared_data(self.folder)

    def test_token_length_report_counts_oversized_rows(self):
        self.write_splits([self.row("one two three four five")],
                          [self.row("one")], [self.row("two")])
        report = audit_sequence_lengths(self.folder, FakeTokenizer(), max_length=6)
        self.assertEqual(report["train"]["over_max_length"], 1)
        self.assertEqual(report["validation"]["over_max_length"], 0)


if __name__ == "__main__":
    unittest.main()
