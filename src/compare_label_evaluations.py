"""Compare label-task evaluations on matching cases, with a base for each run."""

import argparse
import json
from pathlib import Path


def load(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    metric = result.get("generation_metric") or {}
    if metric.get("metric") != "label_accuracy" or "per_label" not in metric:
        raise ValueError(f"{path} does not contain label accuracy and per-label scores")
    return result


def check_matching(first, second, parser, context):
    if first["task"] != second["task"]:
        parser.error(f"{context}: evaluations use different tasks")
    if first.get("split", "test") != second.get("split", "test"):
        parser.error(f"{context}: evaluations use different data splits")
    if first.get("run_manifest_schema") and second.get("run_manifest_schema"):
        if first.get("prepared_artifacts") != second.get("prepared_artifacts"):
            parser.error(f"{context}: evaluations use different prepared data snapshots")
        if first.get("decoding") != second.get("decoding"):
            parser.error(f"{context}: evaluations use different decoding settings")
    first_cases = [(case["prompt"], case["reference"]) for case in first["examples"]]
    second_cases = [(case["prompt"], case["reference"]) for case in second["examples"]]
    if first_cases != second_cases:
        parser.error(f"{context}: evaluations do not contain the same generated cases in the same order")
    if list(first["generation_metric"]["per_label"]) != list(second["generation_metric"]["per_label"]):
        parser.error(f"{context}: evaluations use different label sets")
    return len(first_cases)


def default_baseline_path(path):
    for split in ("validation", "test"):
        suffix = f"{split}_evaluation.json"
        if path.name.endswith(suffix):
            return path.with_name(path.name[:-len(suffix)] + f"base_{suffix}")
    return None


def baseline_for(path, result, other, explicit, parser, name):
    if result.get("model") == "starting_model":
        return result
    if other.get("model") == "starting_model" and Path(other["run_dir"]).resolve() == Path(result["run_dir"]).resolve():
        return other
    baseline_path = explicit or default_baseline_path(path)
    if baseline_path is None or not baseline_path.is_file():
        expected = baseline_path or "a baseline JSON file supplied with --first-base/--second-base"
        parser.error(f"{name}: missing base-model baseline {expected}; run evaluate_task.py "
                     f"--run-dir {result['run_dir']} --base-only with the same split and sampling settings")
    baseline = load(baseline_path)
    if baseline.get("model") != "starting_model":
        parser.error(f"{name}: {baseline_path} is not a base-model evaluation")
    if Path(baseline["run_dir"]).resolve() != Path(result["run_dir"]).resolve():
        parser.error(f"{name}: baseline belongs to a different run")
    check_matching(result, baseline, parser, f"{name} baseline")
    return baseline


def metric_line(name, result):
    metric = result["generation_metric"]
    return (f"{name}: {result['run_dir']} ({result.get('model', 'trained_run')}): "
            f"accuracy={metric['accuracy']:.3f}, macro-F1={metric['macro_f1']:.3f}, "
            f"invalid={metric['invalid_label_outputs']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--first-base", type=Path, help="Base evaluation for a custom first result path")
    parser.add_argument("--second-base", type=Path, help="Base evaluation for a custom second result path")
    parser.add_argument("--output", type=Path, help="Also save the comparison as a text report")
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    case_count = check_matching(first, second, parser, "Compared runs")
    first_base = baseline_for(args.first, first, second, args.first_base, parser, "First run")
    second_base = baseline_for(args.second, second, first, args.second_base, parser, "Second run")

    lines = [f"Task: {first['task']}; split: {first.get('split', 'test')}; same cases: {case_count}"]
    lines.append(metric_line("First baseline", first_base))
    if Path(first_base["run_dir"]).resolve() != Path(second_base["run_dir"]).resolve():
        lines.append(metric_line("Second baseline", second_base))
    for name, result, baseline in (("First", first, first_base), ("Second", second, second_base)):
        if result.get("model") == "starting_model":
            continue
        lines.append(metric_line(name, result))
        metric = result["generation_metric"]
        base_metric = baseline["generation_metric"]
        lines.append(f"  lift over base: accuracy={metric['accuracy'] - base_metric['accuracy']:+.3f}, "
                     f"macro-F1={metric['macro_f1'] - base_metric['macro_f1']:+.3f}")
    lines.append("\nLabel       Support  First recall  Second recall  First F1  Second F1")
    labels = list(first["generation_metric"]["per_label"])
    for label in labels:
        a = first["generation_metric"]["per_label"][label]
        b = second["generation_metric"]["per_label"][label]
        if a["support"] != b["support"]:
            parser.error(f"Different support for {label}")
        lines.append(f"{label:<11} {a['support']:>7}  {a['recall']:>12.3f}  {b['recall']:>13.3f}  "
                     f"{a['f1']:>8.3f}  {b['f1']:>9.3f}")
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
