"""Prepare any configured Hugging Face text task as prompt/response JSONL."""

import argparse
import json
import random
import shutil
import tempfile
import warnings
from pathlib import Path

from datasets import Dataset, load_dataset

from data_integrity import deduplicate_cross_split_prompts, fingerprint_file, prompt_overlap, snapshot_prepared_data
from data_quality import check_quality_policy, scan_data_quality
from task_config import load_task
from task_io import prepared_dir
from task_prompts import format_prompt


def check_columns(dataset, task, split_name):
    required = set(task["input_fields"] + [task["response_field"]])
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{split_name} is missing columns: {', '.join(sorted(missing))}")


def group_key(value):
    if isinstance(value, list):
        return tuple(sorted(str(item).strip().lower() for item in value))
    return str(value).strip().lower()


def split_source(task, local_jsonl=None, heldout_fraction=None, seed=42):
    source = task["source"]
    if local_jsonl:
        with Path(local_jsonl).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        train = Dataset.from_list(rows)
        validation = test = None
    else:
        load_kwargs = {"data_files": source["data_files"]} if source.get("data_files") else {}
        if source.get("revision"):
            load_kwargs["revision"] = source["revision"]
        dataset = load_dataset(source["dataset"], source["config"], **load_kwargs) if source.get("config") else load_dataset(source["dataset"], **load_kwargs)
        train = dataset[source.get("train_split", "train")]
        validation = dataset[source["validation_split"]] if source.get("validation_split") else None
        test = dataset[source["test_split"]] if source.get("test_split") else None
    if validation is None:
        fraction = heldout_fraction if heldout_fraction is not None else source.get("heldout_fraction", 0.1)
        if not 0 < fraction < 1:
            raise ValueError("heldout_fraction must be between 0 and 1")
        group_field = source.get("group_field")
        if group_field:
            if group_field not in train.column_names:
                raise ValueError(f"Group field not in training data: {group_field}")
            if test is not None:
                test_keys = {group_key(value) for value in test[group_field]}
                train = train.select([i for i, value in enumerate(train[group_field]) if group_key(value) not in test_keys])
            groups = list(dict.fromkeys(group_key(value) for value in train[group_field]))
            if len(groups) < 2:
                raise ValueError(f"Need at least two distinct {group_field} groups for validation")
            random.Random(seed).shuffle(groups)
            heldout_count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
            heldout_groups = set(groups[:heldout_count])
            validation_indices = [i for i, value in enumerate(train[group_field]) if group_key(value) in heldout_groups]
            train_indices = [i for i, value in enumerate(train[group_field]) if group_key(value) not in heldout_groups]
            validation = train.select(validation_indices)
            train = train.select(train_indices)
        else:
            parts = train.train_test_split(test_size=fraction, seed=seed)
            train, validation = parts["train"], parts["test"]
    if test is None:
        group_field = source.get("group_field")
        if group_field:
            groups = list(dict.fromkeys(group_key(value) for value in validation[group_field]))
            if len(groups) < 2:
                raise ValueError(f"Need at least two distinct {group_field} groups for validation and test")
            random.Random(seed + 1).shuffle(groups)
            test_groups = set(groups[:max(1, len(groups) // 2)])
            test_indices = [i for i, value in enumerate(validation[group_field]) if group_key(value) in test_groups]
            validation_indices = [i for i, value in enumerate(validation[group_field]) if group_key(value) not in test_groups]
            test = validation.select(test_indices)
            validation = validation.select(validation_indices)
        else:
            heldout = validation.train_test_split(test_size=0.5, seed=seed)
            validation, test = heldout["train"], heldout["test"]
    return {"train": train, "validation": validation, "test": test}


def write_split(dataset, task, path):
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            raw_response = row[task["response_field"]]
            response = str(task.get("response_map", {}).get(str(raw_response), raw_response)).strip()
            if not response:
                continue
            prompt = format_prompt(task, row)
            inputs = {field: row[field] for field in task["input_fields"]}
            output.write(json.dumps({"prompt": prompt, "response": response, "inputs": inputs}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Task JSON configuration")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-jsonl", type=Path, help="Optional local JSONL source for an offline smoke test")
    parser.add_argument("--heldout-fraction", type=float, help="Used when source has no validation/test splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-on-overlap", action="store_true",
                        help="Reject exact prompts shared across train, validation, or test")
    parser.add_argument("--deduplicate-cross-split", action="store_true",
                        help="Keep held-out rows and drop lower-priority rows with the same prompt")
    args = parser.parse_args()
    task = load_task(args.task)
    output_dir = args.output_dir or prepared_dir(task)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    splits = split_source(task, args.local_jsonl, args.heldout_fraction, args.seed)
    for name, dataset in splits.items():
        check_columns(dataset, task, name)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        counts = {name: write_split(dataset, task, staging / f"{name}.jsonl")
                  for name, dataset in splits.items()}
        removed = deduplicate_cross_split_prompts(staging) if args.deduplicate_cross_split else {}
        if removed:
            counts = {name: counts[name] - removed[name] for name in counts}
        if min(counts.values()) == 0:
            raise ValueError("A prepared split is empty; choose a larger source dataset")
        (staging / "task_config.json").write_text(args.task.read_text(encoding="utf-8"), encoding="utf-8")
        quality = scan_data_quality(staging)
        check_quality_policy(quality, task)
        artifacts = snapshot_prepared_data(staging)
        overlap = prompt_overlap(staging)
        if any(overlap.values()):
            if args.fail_on_overlap:
                raise ValueError(f"Cross-split prompt overlap found: {overlap}")
            warnings.warn(f"Cross-split prompt overlap found: {overlap}; use --fail-on-overlap for a strict preparation",
                          stacklevel=1)
        manifest = {
            "task": task["name"],
            "source": str(args.local_jsonl) if args.local_jsonl else task["source"].get("repository", task["source"]["dataset"]),
            "source_sha256": fingerprint_file(args.local_jsonl)["sha256"] if args.local_jsonl else None,
            "seed": args.seed,
            "counts": counts,
            "format": "prompt/response JSONL",
            "artifacts": artifacts,
            "prompt_overlap": overlap,
            "quality": quality,
            "removed_cross_split": removed,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # Only an empty directory may be replaced.
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared {task['name']} in {output_dir}: {counts}")
    print("Example prompt:\n" + json.loads((output_dir / "train.jsonl").read_text().splitlines()[0])["prompt"])


if __name__ == "__main__":
    main()
