"""Audit explicit prompt-field cues in matching held-out generations.

These string checks are diagnostics, not a semantic adherence score.
"""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIGNS = (
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
    "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
)
CATEGORY_CUES = {
    "Birthday": r"\b(?:birthday|birthdays|born|party|parties|celebrat\w*)\b",
    "Career": r"\b(?:career|careers|work|working|job|jobs|office|business|boss|promotion|interview|professional)\b",
    "Love": r"\b(?:love|loving|romance|romantic|relationship|partner|dating|affection|intimacy|intimate|crush)\b",
    "Wellness": r"\b(?:health|healthy|wellness|diet|exercise|rest|sleep|stress|body|breathe|breathing|relax\w*|illness)\b",
}


def audit(case, output):
    sign_names = [name for name in SIGNS if re.search(rf"\b{name}\b", output, re.I)]
    cue = CATEGORY_CUES.get(case["category"])
    # This intentionally checks literal date reproduction only. A horoscope
    # can respond to a date without repeating it, so absence is not failure.
    date_mentioned = case["date"] in output or case["date"].replace("/", "-") in output
    return {
        "requested_sign_mentioned": case["sign"] in sign_names,
        "other_signs_mentioned": [name for name in sign_names if name != case["sign"]],
        "category_cue": bool(re.search(cue, output, re.I)) if cue else None,
        "exact_date_mentioned": date_mentioned,
    }


def summarize(rows):
    applicable = [row for row in rows if row["audit"]["category_cue"] is not None]
    return {
        "cases": len(rows),
        "requested_sign_mentioned": sum(row["audit"]["requested_sign_mentioned"] for row in rows),
        "other_sign_mentioned": sum(bool(row["audit"]["other_signs_mentioned"]) for row in rows),
        "category_cue_present": sum(row["audit"]["category_cue"] for row in applicable),
        "category_cue_applicable": len(applicable),
        "exact_date_mentioned": sum(row["audit"]["exact_date_mentioned"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch", type=Path, default=ROOT / "results/independent_lora_512.json")
    parser.add_argument("--mlx", type=Path, default=ROOT / "results/independent_mlx_qlora_512.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/independent_adherence.json")
    args = parser.parse_args()
    pytorch = json.loads(args.pytorch.read_text())
    mlx = json.loads(args.mlx.read_text())
    a, b = pytorch["generation_cases"], mlx["generation_cases"]
    if len(a) != len(b) or any(x["prompt"] != y["prompt"] for x, y in zip(a, b)):
        raise ValueError("PyTorch and MLX review prompts do not match")

    rows = {"reference": [], "peft_lora_512": [], "mlx_qlora_512": []}
    for x, y in zip(a, b):
        shared = {key: x[key] for key in ("sign", "category", "date")}
        outputs = {
            "reference": x["reference"],
            "peft_lora_512": x["outputs"]["lora-512"],
            "mlx_qlora_512": y["output"],
        }
        for name, output in outputs.items():
            rows[name].append(shared | {"output": output, "audit": audit(shared, output)})

    payload = {
        "purpose": "Explicit lexical cues only; these counts do not establish semantic adherence.",
        "category_cue_patterns": CATEGORY_CUES,
        "summary": {name: summarize(items) for name, items in rows.items()},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for name, values in payload["summary"].items():
        print(name, values)


if __name__ == "__main__":
    main()
