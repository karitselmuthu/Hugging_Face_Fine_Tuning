"""Offline checks for task configuration and prepared-data integrity."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_integrity import prompt_overlap, snapshot_prepared_data, verify_prepared_data  # noqa: E402
from task_core import load_task, preparation_spec  # noqa: E402


class TaskConfigTests(unittest.TestCase):
    def test_all_included_tasks_are_valid(self):
        for path in (ROOT / "tasks").glob("*.json"):
            with self.subTest(path=path.name):
                task = load_task(path)
                self.assertTrue(task["input_fields"])

    def test_invalid_settings_fail_before_training(self):
        original = json.loads((ROOT / "tasks/emotion_classification.json").read_text())
        changes = [
            (lambda value: value.update(max_length="192"), "max_length"),
            (lambda value: value["source"].update(dataset=""), "source.dataset"),
            (lambda value: value["source"].update(revision=""), "source.revision"),
            (lambda value: value["training"].update(batch_size=0), "training.batch_size"),
            (lambda value: value["evaluation"].update(labels=["joy", "joy"]), "evaluation.labels"),
            (lambda value: value.update(prompt_template="Bad {text"), "prompt_template"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            for change, message in changes:
                with self.subTest(message=message):
                    task = json.loads(json.dumps(original))
                    change(task)
                    path.write_text(json.dumps(task))
                    with self.assertRaisesRegex(ValueError, message):
                        load_task(path)

    def test_training_settings_do_not_change_preparation_spec(self):
        task = load_task(ROOT / "tasks/emotion_classification.json")
        other = json.loads(json.dumps(task))
        other["training"]["train_samples"] = 2000
        self.assertEqual(preparation_spec(task), preparation_spec(other))


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.task_path = ROOT / "tasks/emotion_classification.json"
        self.source_path = self.directory / "source.jsonl"
        rows = [{"text": f"example number {index}", "label": index % 6} for index in range(30)]
        self.source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.output_dir = self.directory / "prepared"

    def prepare(self, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "src/prepare_task.py"), "--task", str(self.task_path),
             "--local-jsonl", str(self.source_path), "--output-dir", str(self.output_dir), *extra],
            cwd=ROOT, text=True, capture_output=True,
        )

    def test_preparation_records_hashes_and_detects_changed_data(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["artifacts"], snapshot_prepared_data(self.output_dir))
        self.assertEqual(manifest["prompt_overlap"], prompt_overlap(self.output_dir))
        self.assertFalse(any(manifest["prompt_overlap"].values()))
        expected = verify_prepared_data(self.output_dir)
        with (self.output_dir / "test.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "differ from their manifest"):
            verify_prepared_data(self.output_dir)
        manifest.pop("artifacts")
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "differ from the training run"):
            verify_prepared_data(self.output_dir, expected)

    def test_failed_preparation_leaves_no_partial_output(self):
        rows = [{"text": f"example {index}", "label": index % 6} for index in range(30)]
        rows[7]["text"] = ""
        self.source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Empty prompt field", result.stderr)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(list(self.directory.glob(".prepared.tmp-*")), [])

    def test_strict_overlap_rejects_shared_prompts_without_partial_output(self):
        rows = [{"text": "the same input", "label": index % 6} for index in range(30)]
        self.source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.prepare("--fail-on-overlap")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cross-split prompt overlap", result.stderr)
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(list(self.directory.glob(".prepared.tmp-*")), [])

    def test_deduplication_preserves_heldout_rows_and_clears_overlap(self):
        rows = [{"text": "shared input" if index < 12 else f"unique {index}",
                 "label": index % 6} for index in range(30)]
        self.source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = self.prepare("--deduplicate-cross-split", "--fail-on-overlap")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((self.output_dir / "manifest.json").read_text())
        self.assertGreater(sum(manifest["removed_cross_split"].values()), 0)
        self.assertFalse(any(prompt_overlap(self.output_dir).values()))
        self.assertEqual(manifest["artifacts"], snapshot_prepared_data(self.output_dir))


if __name__ == "__main__":
    unittest.main()
