"""Gate a tested run against configured metrics and save a local registry record."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_lineage import file_sha256, load_run_manifest
from task_config import load_task
from task_io import ROOT


def evaluate_promotion(task, trained, base, run_dir):
    policy = task.get("promotion")
    if not policy:
        raise ValueError("Task has no promotion thresholds; define promotion in its task JSON")
    if trained.get("model") != "trained_run" or base.get("model") != "starting_model":
        raise ValueError("Provide a trained evaluation and a base-model evaluation")
    if trained.get("split") != "test" or base.get("split") != "test":
        raise ValueError("Promotion requires matching test-split evaluations")
    if any(Path(item.get("run_dir", "")).resolve() != Path(run_dir).resolve() for item in (trained, base)):
        raise ValueError("Both evaluations must belong to the requested run")
    if trained.get("task") != task["name"] or base.get("task") != task["name"]:
        raise ValueError("Evaluation task does not match the run")
    for key in ("prepared_artifacts", "sampling", "decoding"):
        if trained.get(key) != base.get(key):
            raise ValueError(f"Base and trained evaluations differ in {key}")
    cases = lambda item: [(row["prompt"], row["reference"]) for row in item["examples"]]
    if cases(trained) != cases(base) or not cases(trained):
        raise ValueError("Base and trained evaluations must use the same nonempty cases")
    trained_metric = trained.get("generation_metric") or {}
    base_metric = base.get("generation_metric") or {}
    if trained_metric.get("metric") != base_metric.get("metric"):
        raise ValueError("Base and trained metric types differ")
    failures = []
    if len(trained["examples"]) < policy.get("minimum_examples", 1):
        failures.append(f"generated cases={len(trained['examples'])} below {policy['minimum_examples']}")
    for key, minimum in policy["minimum_metrics"].items():
        value = trained_metric.get(key)
        if not isinstance(value, (int, float)) or value < minimum:
            failures.append(f"{key}={value} below {minimum}")
    for key, minimum in policy.get("minimum_lift", {}).items():
        value = trained_metric.get(key)
        starting = base_metric.get(key)
        if not isinstance(value, (int, float)) or not isinstance(starting, (int, float)) or value - starting < minimum:
            failures.append(f"{key} lift below {minimum}")
    invalid = trained_metric.get("invalid_label_outputs", 0)
    if invalid > policy.get("maximum_invalid_label_outputs", 0):
        failures.append(f"invalid_label_outputs={invalid} exceeds policy")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--base-evaluation", type=Path)
    args = parser.parse_args()
    manifest = load_run_manifest(args.run_dir)
    if manifest is None:
        parser.error("Promotion requires a versioned run manifest")
    task = load_task(args.run_dir / "task_config.json")
    evaluation_path = args.evaluation or args.run_dir / "test_evaluation.json"
    base_path = args.base_evaluation or args.run_dir / "base_test_evaluation.json"
    trained = json.loads(evaluation_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if trained.get("prepared_artifacts") != manifest["prepared_artifacts"] or base.get("prepared_artifacts") != manifest["prepared_artifacts"]:
        parser.error("Evaluations do not match the run's prepared-data fingerprint")
    if trained.get("run_manifest_schema") != manifest["schema_version"] or base.get("run_manifest_schema") != manifest["schema_version"]:
        parser.error("Evaluations do not match the run manifest schema")
    try:
        failures = evaluate_promotion(task, trained, base, args.run_dir)
    except ValueError as exc:
        parser.error(str(exc))
    decision = {"task": task["name"], "run_dir": str(args.run_dir.resolve()),
                "status": "rejected" if failures else "promoted", "failures": failures,
                "policy": task["promotion"], "metrics": trained["generation_metric"],
                "base_metrics": base["generation_metric"],
                "run_manifest_sha256": file_sha256(args.run_dir / "run_manifest.json"),
                "evaluation_sha256": file_sha256(evaluation_path), "base_evaluation_sha256": file_sha256(base_path),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    decision_path = args.run_dir / "promotion_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    if failures:
        parser.exit(2, f"Promotion rejected; see {decision_path}: {'; '.join(failures)}\n")
    registry_dir = ROOT / "models" / "registry" / task["name"]
    registry_dir.mkdir(parents=True, exist_ok=True)
    record_path = registry_dir / f"{args.run_dir.name}-{decision['run_manifest_sha256'][:12]}.json"
    if record_path.exists():
        parser.error(f"Registry record already exists: {record_path}")
    record_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"Promoted {args.run_dir} to local registry: {record_path}")


if __name__ == "__main__":
    main()
