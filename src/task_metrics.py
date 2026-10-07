"""Small, transparent task metrics for generated examples."""

import re
import random


def words(text):
    return re.findall(r"\b\w+\b", text.lower())


def rouge_l_f1(reference, output):
    a, b = words(reference), words(output)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0] * (len(b) + 1)
        for j, candidate in enumerate(b, start=1):
            current[j] = previous[j - 1] + 1 if token == candidate else max(previous[j], current[j - 1])
        previous = current
    overlap = previous[-1]
    return 2 * overlap / (len(a) + len(b))


def score_generations(task, examples):
    settings = task.get("evaluation", {})
    metric = settings.get("metric")
    if not metric:
        return None
    if metric == "label_accuracy":
        labels = settings["labels"]
        label_set = set(labels)
        predictions = []
        for case in examples:
            predicted = case["output"].strip().lower()
            predictions.append(predicted)
            case["predicted_label"] = predicted
        correct = sum(prediction == case["reference"].lower() for prediction, case in zip(predictions, examples))
        per_label = {}
        for label in labels:
            true_positives = sum(prediction == label and case["reference"].lower() == label
                                 for prediction, case in zip(predictions, examples))
            false_positives = sum(prediction == label and case["reference"].lower() != label
                                  for prediction, case in zip(predictions, examples))
            false_negatives = sum(prediction != label and case["reference"].lower() == label
                                  for prediction, case in zip(predictions, examples))
            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_label[label] = {"support": true_positives + false_negatives,
                                "predicted": true_positives + false_positives,
                                "precision": precision, "recall": recall, "f1": f1}
        return {"metric": metric, "examples": len(examples), "accuracy": correct / len(examples),
                "macro_f1": sum(values["f1"] for values in per_label.values()) / len(labels),
                "per_label": per_label,
                "invalid_label_outputs": sum(prediction not in label_set for prediction in predictions)}
    if metric == "concept_coverage_exact":
        output_coverage = []
        reference_coverage = []
        for case in examples:
            concepts = [str(x).lower() for x in case["inputs"]["concepts"]]
            output_words, reference_words = set(words(case["output"])), set(words(case["reference"]))
            output_coverage.append(sum(concept in output_words for concept in concepts) / len(concepts))
            reference_coverage.append(sum(concept in reference_words for concept in concepts) / len(concepts))
        return {"metric": metric, "examples": len(examples),
                "mean_output_concept_fraction": sum(output_coverage) / len(examples),
                "mean_reference_concept_fraction": sum(reference_coverage) / len(examples),
                "note": "Exact word forms only; inflections such as ski/skis count as different."}
    if metric == "rouge_l_f1":
        values = [rouge_l_f1(case["reference"], case["output"]) for case in examples]
        return {"metric": metric, "examples": len(examples), "mean_f1": sum(values) / len(values),
                "note": "Word-overlap diagnostic; does not detect invented facts."}
    raise ValueError(f"Unknown evaluation metric: {metric}")


def bootstrap_intervals(task, examples, seed, resamples=500):
    """Estimate descriptive 95% intervals by resampling generated cases."""
    if not examples or not task.get("evaluation", {}).get("metric"):
        return None
    keys = {"label_accuracy": ("accuracy", "macro_f1"),
            "concept_coverage_exact": ("mean_output_concept_fraction",),
            "rouge_l_f1": ("mean_f1",)}[task["evaluation"]["metric"]]
    rng = random.Random(seed)
    values = {key: [] for key in keys}
    for _ in range(resamples):
        sample = [dict(examples[rng.randrange(len(examples))]) for _ in examples]
        scored = score_generations(task, sample)
        for key in keys:
            values[key].append(scored[key])
    intervals = {}
    for key, samples in values.items():
        samples.sort()
        intervals[key] = {"lower": samples[int(0.025 * (resamples - 1))],
                          "upper": samples[int(0.975 * (resamples - 1))]}
    return {"method": "case_bootstrap_percentile", "confidence": 0.95,
            "resamples": resamples, "seed": seed, "intervals": intervals}
