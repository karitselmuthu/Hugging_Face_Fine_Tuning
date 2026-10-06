"""Compare two label-task evaluations on exactly the same generated cases."""

import argparse
import json
from pathlib import Path


def load(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    metric = result.get("generation_metric") or {}
    if metric.get("metric") != "label_accuracy" or "per_label" not in metric:
        raise ValueError(f"{path} does not contain label accuracy and per-label scores")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first, second = load(args.first), load(args.second)
    if first["task"] != second["task"]:
        parser.error("Evaluations use different tasks")
    if first.get("split", "test") != second.get("split", "test"):
        parser.error("Evaluations use different data splits")
    first_cases = [(case["prompt"], case["reference"]) for case in first["examples"]]
    second_cases = [(case["prompt"], case["reference"]) for case in second["examples"]]
    if first_cases != second_cases:
        parser.error("Evaluations do not contain the same generated test cases in the same order")
    labels = list(first["generation_metric"]["per_label"])
    if labels != list(second["generation_metric"]["per_label"]):
        parser.error("Evaluations use different label sets")

    print(f"Task: {first['task']}; split: {first.get('split', 'test')}; same cases: {len(first_cases)}")
    for result in (first, second):
        metric = result["generation_metric"]
        print(f"{result['run_dir']}: accuracy={metric['accuracy']:.3f}, "
              f"macro-F1={metric['macro_f1']:.3f}, invalid={metric['invalid_label_outputs']}")
    print("\nLabel       Support  First recall  Second recall  First F1  Second F1")
    for label in labels:
        a = first["generation_metric"]["per_label"][label]
        b = second["generation_metric"]["per_label"][label]
        if a["support"] != b["support"]:
            parser.error(f"Different test support for {label}")
        print(f"{label:<11} {a['support']:>7}  {a['recall']:>12.3f}  {b['recall']:>13.3f}  "
              f"{a['f1']:>8.3f}  {b['f1']:>9.3f}")


if __name__ == "__main__":
    main()
