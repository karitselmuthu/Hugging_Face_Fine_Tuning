"""Aggregate matching held-out evaluations from distinct training seeds."""

import argparse
import json
import statistics
from pathlib import Path


def signature(result):
    return (result["task"], result["split"], result["generation_metric"]["metric"],
            result.get("prepared_artifacts"), result.get("sampling"), result.get("decoding"),
            [(case["prompt"], case["reference"]) for case in result["examples"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.evaluations) < 2:
        parser.error("Provide at least two evaluations from distinct training seeds")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.evaluations]
    if any(result.get("model") != "trained_run" for result in results):
        parser.error("Use trained-run evaluations only")
    reference = signature(results[0])
    if any(signature(result) != reference for result in results[1:]):
        parser.error("Evaluations must use the same task, data, sampling, decoding, metric, and generated cases")
    seeds = [result.get("training_seed") for result in results]
    if None in seeds or len(set(seeds)) != len(seeds):
        parser.error("Evaluations need distinct recorded training seeds")
    summaries = [json.loads((Path(result["run_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
                 for result in results]
    settings = [(summary["method"], summary["starting_model"], summary.get("starting_model_revision"),
                 summary["train_examples"], summary["validation_examples"]) for summary in summaries]
    if any(setting != settings[0] for setting in settings[1:]):
        parser.error("Runs differ in method, starting model, or sample counts")
    metric = results[0]["generation_metric"]["metric"]
    keys = {"label_accuracy": ("accuracy", "macro_f1"),
            "concept_coverage_exact": ("mean_output_concept_fraction",),
            "rouge_l_f1": ("mean_f1",)}[metric]
    aggregate = {key: {"mean": statistics.mean(result["generation_metric"][key] for result in results),
                       "sample_std": statistics.stdev(result["generation_metric"][key] for result in results),
                       "values": [result["generation_metric"][key] for result in results]}
                 for key in keys}
    report = {"task": results[0]["task"], "split": results[0]["split"], "metric": metric,
              "training_seeds": seeds, "cases": len(results[0]["examples"]), "aggregate": aggregate,
              "evaluations": [str(path) for path in args.evaluations]}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
