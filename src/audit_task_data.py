"""Audit prepared task files without loading a model or downloading data."""

import argparse
import json
from pathlib import Path

from data_integrity import prompt_overlap, verify_prepared_data
from data_quality import check_quality_policy, scan_data_quality


def audit_prepared_data(data_dir, *, fail_on_overlap=False, fail_on_near_duplicate=False, fail_on_pii=False):
    """Return verified artifacts and overlap counts before any model is loaded."""
    artifacts = verify_prepared_data(data_dir)
    overlap = prompt_overlap(data_dir)
    if fail_on_overlap and any(overlap.values()):
        raise ValueError(f"Cross-split prompt overlap {overlap}; prepare clean data or allow overlap for historical experiments")
    quality = scan_data_quality(data_dir)
    task = json.loads((Path(data_dir) / "task_config.json").read_text(encoding="utf-8"))
    check_quality_policy(quality, task, fail_on_near_duplicate=fail_on_near_duplicate,
                         fail_on_pii=fail_on_pii)
    return {"data_dir": str(data_dir), "artifacts": artifacts, "prompt_overlap": overlap,
            "quality": quality}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fail-on-overlap", action="store_true")
    parser.add_argument("--fail-on-near-duplicate", action="store_true")
    parser.add_argument("--fail-on-pii", action="store_true")
    parser.add_argument("--output", type=Path, help="Write a successful audit as JSON for a pipeline gate")
    args = parser.parse_args()
    try:
        report = audit_prepared_data(args.data_dir, fail_on_overlap=args.fail_on_overlap,
                                     fail_on_near_duplicate=args.fail_on_near_duplicate,
                                     fail_on_pii=args.fail_on_pii)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
