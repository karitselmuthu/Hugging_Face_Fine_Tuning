"""Fingerprint prepared task data and reject changes to recorded artifacts."""

import hashlib
import json
import warnings
from pathlib import Path


ARTIFACTS = ("train.jsonl", "validation.jsonl", "test.jsonl", "task_config.json")
SPLITS = ("train", "validation", "test")


def fingerprint_file(path):
    digest = hashlib.sha256()
    rows = 0
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows}


def snapshot_prepared_data(data_dir):
    data_dir = Path(data_dir)
    return {name: fingerprint_file(data_dir / name) for name in ARTIFACTS}


def prompt_overlap(data_dir):
    """Count exact prompts appearing in more than one prepared split."""
    data_dir = Path(data_dir)
    prompts = {}
    for split in SPLITS:
        values = set()
        with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                prompt = json.loads(line).get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError(f"{split}.jsonl:{number}: prompt must be a string")
                values.add(prompt)
        prompts[split] = values
    return {
        "train_validation": len(prompts["train"] & prompts["validation"]),
        "train_test": len(prompts["train"] & prompts["test"]),
        "validation_test": len(prompts["validation"] & prompts["test"]),
    }


def deduplicate_cross_split_prompts(data_dir):
    """Keep test, then validation, then train when a prompt spans splits."""
    data_dir = Path(data_dir)
    higher_priority_prompts = set()
    removed = {}
    for split in reversed(SPLITS):
        path = data_dir / f"{split}.jsonl"
        retained = []
        current_prompts = set()
        removed[split] = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row["prompt"]
                if prompt in higher_priority_prompts:
                    removed[split] += 1
                else:
                    retained.append(line)
                    current_prompts.add(prompt)
        path.write_text("".join(retained), encoding="utf-8")
        higher_priority_prompts.update(current_prompts)
    return removed


def verify_prepared_data(data_dir, expected=None):
    """Check the preparation manifest and, when available, the training snapshot."""
    data_dir = Path(data_dir)
    actual = snapshot_prepared_data(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("artifacts")
    if recorded is not None and actual != recorded:
        raise ValueError(f"Prepared files in {data_dir} differ from their manifest; prepare a new data directory")
    if recorded is None:
        counts = manifest.get("counts", {})
        mismatches = [name for name in ("train", "validation", "test")
                      if counts.get(name) != actual[f"{name}.jsonl"]["rows"]]
        if mismatches:
            warnings.warn(f"Legacy manifest in {data_dir} has stale row counts for: {', '.join(mismatches)}",
                          stacklevel=2)
    if expected is not None and actual != expected:
        raise ValueError(f"Prepared files in {data_dir} differ from the training run; restore the original data or use a matching run")
    return actual
