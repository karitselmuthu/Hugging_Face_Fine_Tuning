"""Small, transparent task metrics for generated examples."""

import re


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
        labels = set(settings["labels"])
        predictions = []
        for case in examples:
            predicted = case["output"].strip().lower()
            predictions.append(predicted)
            case["predicted_label"] = predicted
        correct = sum(prediction == case["reference"].lower() for prediction, case in zip(predictions, examples))
        return {"metric": metric, "examples": len(examples), "accuracy": correct / len(examples),
                "invalid_label_outputs": sum(prediction not in labels for prediction in predictions)}
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
