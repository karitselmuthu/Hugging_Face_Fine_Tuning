"""Audit response-only training lengths and literal input mentions."""

import json
import math
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT / "models/smollm-horoscope-response-only-2000/final"
OUTPUT_PATH = ROOT / "results/training_data_audit.json"
LIMITS = (128, 256, 384, 512)
MARKER = "### Horoscope:\n"
FIELDS = re.compile(
    r"### Sign:\n(?P<sign>[^\n]+)\n\n"
    r"### Category:\n(?P<category>[^\n]+)\n\n"
    r"### Date:\n(?P<date>[^\n]+)"
)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def audit(dataset, tokenizer):
    records = []
    categories = Counter()
    mentions = Counter()
    for row in dataset:
        prompt, marker, response = row["text"].partition(MARKER)
        fields = FIELDS.search(prompt)
        if not marker or not fields:
            raise ValueError("Prepared data contains an unrecognized example")
        prompt += marker
        response = response.strip()
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        response_length = len(
            tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        )
        records.append({"prompt": prompt_length, "response": response_length})
        values = fields.groupdict()
        categories[values["category"]] += 1
        for field in ("sign", "category"):
            if re.search(rf"\b{re.escape(values[field])}\b", response, re.IGNORECASE):
                mentions[field] += 1
        if values["date"] in response:
            mentions["exact_date"] += 1
        if values["date"][:4] in response:
            mentions["year"] += 1

    total = len(records)
    lengths = {key: [record[key] for record in records] for key in ("prompt", "response")}
    lengths["total"] = [record["prompt"] + record["response"] for record in records]
    percentiles = {
        key: {str(p): percentile(values, p / 100) for p in (50, 90, 95, 99, 100)}
        for key, values in lengths.items()
    }
    limits = {}
    for limit in LIMITS:
        truncated = sum(length > limit for length in lengths["total"])
        retained_response = sum(
            min(record["response"], max(0, limit - record["prompt"]))
            for record in records
        )
        total_response = sum(lengths["response"])
        limits[str(limit)] = {
            "truncated_examples": truncated,
            "truncated_percent": 100 * truncated / total,
            "response_tokens_retained_percent": 100 * retained_response / total_response,
            "examples_without_response_tokens": sum(
                record["prompt"] >= limit for record in records
            ),
        }
    return {
        "examples": total,
        "length_percentiles": percentiles,
        "limits": limits,
        "categories": dict(sorted(categories.items())),
        "literal_mentions": {
            field: {"count": mentions[field], "percent": 100 * mentions[field] / total}
            for field in ("sign", "category", "exact_date", "year")
        },
    }


def write_report(path, results):
    lines = [
        "# Training data audit",
        "",
        "Tokenization matches the response-only training scripts: prompt and",
        "response are tokenized separately, with an end-of-sequence token on",
        "the response. The current training limit is 256 tokens.",
        "",
    ]
    for name, data in results.items():
        lines.extend([
            f"## {name} ({data['examples']:,} examples)",
            "",
            "| Length | Median | P90 | P95 | P99 | Maximum |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for label, values in data["length_percentiles"].items():
            lines.append(
                f"| {label.title()} | {values['50']} | {values['90']} | "
                f"{values['95']} | {values['99']} | {values['100']} |"
            )
        lines.extend([
            "",
            "| Token limit | Examples truncated | Response tokens retained | No response tokens |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for limit, values in data["limits"].items():
            lines.append(
                f"| {limit} | {values['truncated_examples']:,} "
                f"({values['truncated_percent']:.1f}%) | "
                f"{values['response_tokens_retained_percent']:.1f}% | "
                f"{values['examples_without_response_tokens']:,} |"
            )
        lines.extend([
            "",
            "Literal field mentions in the target horoscope:",
            "",
            "| Field | Mentions | Percent of examples |",
            "| --- | ---: | ---: |",
        ])
        for field, values in data["literal_mentions"].items():
            lines.append(f"| {field} | {values['count']:,} | {values['percent']:.1f}% |")
        lines.extend(["", "Categories: " + ", ".join(
            f"{category} {count:,}" for category, count in data["categories"].items()
        ), ""])
    lines.extend([
        "Literal mentions are a weak proxy for conditioning. A horoscope may",
        "follow a sign or category without repeating its name, and date details",
        "may be implicit. Inspect examples before treating these rates as errors.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    train = load_from_disk(str(ROOT / "data/processed/train"))
    test = load_from_disk(str(ROOT / "data/processed/test"))
    datasets = {
        "Training subset used for 2,000-example model": train.shuffle(seed=42).select(range(2000)),
        "Full training split": train,
        "Held-out test split": test,
    }
    results = {}
    for name, dataset in datasets.items():
        print(f"Analyzing {name}: {len(dataset):,} examples...", flush=True)
        results[name] = audit(dataset, tokenizer)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report_path = OUTPUT_PATH.with_suffix(".md")
    write_report(report_path, results)
    print(f"Saved {OUTPUT_PATH} and {report_path}")


if __name__ == "__main__":
    main()
