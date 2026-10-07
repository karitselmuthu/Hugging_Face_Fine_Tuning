"""Checkpoint guards reject changed or completed runs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_resume import resolve_checkpoint, write_guard  # noqa: E402


class ResumeTests(unittest.TestCase):
    def test_matching_incomplete_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            expected = {"task": "example", "max_steps": 3, "seed": 42}
            write_guard(run, expected)
            checkpoint = run / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
            self.assertEqual(resolve_checkpoint(run, "latest", expected), checkpoint)

    def test_changed_settings_and_completed_checkpoint_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            expected = {"task": "example", "max_steps": 1, "seed": 42}
            write_guard(run, expected)
            checkpoint = run / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
            with self.assertRaisesRegex(ValueError, "differ"):
                resolve_checkpoint(run, "latest", dict(expected, seed=17))
            with self.assertRaisesRegex(ValueError, "already reached"):
                resolve_checkpoint(run, "latest", expected)


if __name__ == "__main__":
    unittest.main()
