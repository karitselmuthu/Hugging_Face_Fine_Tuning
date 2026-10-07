"""Explainable prepared-data diagnostics; reports contain counts, never matched text."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from data_integrity import SPLITS
from task_io import read_jsonl


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)"),
}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "github_token": re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "assigned_secret": re.compile(r"\b(?:api[_-]?key|secret|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{20,}", re.I),
}


def source_text(row):
    inputs = row.get("inputs")
    if isinstance(inputs, dict) and inputs:
        raw = " ".join(json.dumps(inputs[key], ensure_ascii=False, sort_keys=True)
                       for key in sorted(inputs))
    else:
        raw = row["prompt"]
    return " ".join(re.findall(r"\w+", raw.lower()))


def simhash(value):
    tokens = value.split()
    features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    weights = [0] * 64
    for feature in set(features):
        bits = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if bits & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(weights) if weight > 0)


def near_duplicate_pairs(first, second):
    """Find likely cross-split input matches with SimHash bands and text similarity."""
    buckets = defaultdict(set)
    for index, (value, bits) in enumerate(first):
        for band in range(8):
            buckets[(band, (bits >> (band * 8)) & 255)].add(index)
        words = value.split()
        if len(words) >= 3:
            buckets[("prefix", tuple(words[:3]))].add(index)
    count = 0
    for value, bits in second:
        candidates = set()
        for band in range(8):
            candidates.update(buckets.get((band, (bits >> (band * 8)) & 255), ()))
        words = value.split()
        if len(words) >= 3:
            candidates.update(buckets.get(("prefix", tuple(words[:3])), ()))
        for index in candidates:
            earlier, earlier_bits = first[index]
            if (earlier_bits ^ bits).bit_count() > 12:
                continue
            if SequenceMatcher(None, earlier, value, autojunk=False).ratio() >= 0.9:
                count += 1
    return count


def scan_data_quality(data_dir):
    data_dir = Path(data_dir)
    task = json.loads((data_dir / "task_config.json").read_text(encoding="utf-8"))
    labels = (task.get("evaluation") or {}).get("labels") or []
    entries = {}
    distributions = {}
    sensitive = {}
    for split in SPLITS:
        rows = list(read_jsonl(data_dir / f"{split}.jsonl"))
        entries[split] = [(value, simhash(value)) for row in rows
                          if (value := source_text(row))]
        counts = Counter(row["response"].strip().lower() for row in rows)
        distributions[split] = {label: counts.get(label, 0) for label in labels} if labels else None
        if labels:
            distributions[split]["other"] = sum(count for label, count in counts.items() if label not in labels)
        pii_counts, secret_counts = Counter(), Counter()
        for row in rows:
            text = f"{row['prompt']}\n{row['response']}"
            for name, pattern in PII_PATTERNS.items():
                pii_counts[name] += bool(pattern.search(text))
            for name, pattern in SECRET_PATTERNS.items():
                secret_counts[name] += bool(pattern.search(text))
        sensitive[split] = {"pii_rows_by_type": dict(pii_counts), "secret_rows_by_type": dict(secret_counts)}
    overlap = {f"{a}_{b}": near_duplicate_pairs(entries[a], entries[b])
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    return {"near_duplicate_pairs": overlap, "near_duplicate_method": "input SimHash + 0.90 text similarity; heuristic",
            "label_distribution": distributions, "sensitive": sensitive}


def check_quality_policy(report, task, *, fail_on_near_duplicate=False, fail_on_pii=False):
    policy = task.get("data_quality", {})
    if (fail_on_near_duplicate or policy.get("fail_on_near_duplicate", False)) and any(report["near_duplicate_pairs"].values()):
        raise ValueError(f"Near-duplicate inputs across splits: {report['near_duplicate_pairs']}")
    pii = sum(sum(item["pii_rows_by_type"].values()) for item in report["sensitive"].values())
    if (fail_on_pii or policy.get("fail_on_pii", False)) and pii:
        raise ValueError(f"PII patterns found in {pii} split/type matches; review the local audit report")
    secrets = sum(sum(item["secret_rows_by_type"].values()) for item in report["sensitive"].values())
    if policy.get("fail_on_secret", True) and secrets:
        raise ValueError(f"Secret patterns found in {secrets} split/type matches; remove them before training")


def audit_sequence_lengths(data_dir, tokenizer, max_length):
    report = {}
    for split in SPLITS:
        total, over, largest = 0, 0, 0
        for row in read_jsonl(Path(data_dir) / f"{split}.jsonl"):
            prompt_tokens = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            response_tokens = len(tokenizer(row["response"] + tokenizer.eos_token,
                                            add_special_tokens=False)["input_ids"])
            length = prompt_tokens + response_tokens
            total += 1
            over += length > max_length
            largest = max(largest, length)
        report[split] = {"rows": total, "over_max_length": over, "fraction_over": over / total if total else 0,
                         "longest_tokens": largest}
    return report
