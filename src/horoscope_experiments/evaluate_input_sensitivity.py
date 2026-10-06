"""Compare how two saved models respond when one horoscope input changes."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = {
    "response-only-512": PROJECT_ROOT / "models/smollm-horoscope-response-only-512/final",
    "response-only-2000": PROJECT_ROOT / "models/smollm-horoscope-response-only-2000/final",
}
CASES = [
    {"name": "baseline", "sign": "Aries", "category": "General", "date": "2026/10/04"},
    {"name": "change-sign", "sign": "Taurus", "category": "General", "date": "2026/10/04"},
    {"name": "change-category", "sign": "Aries", "category": "Career", "date": "2026/10/04"},
    {"name": "change-date", "sign": "Aries", "category": "General", "date": "2026/11/04"},
]


def make_prompt(case):
    return f"""### Instruction:
Generate a horoscope using the following information.

### Sign:
{case['sign']}

### Category:
{case['category']}

### Date:
{case['date']}

### Horoscope:
"""


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_report(path, results, settings):
    lines = [
        "# Horoscope input sensitivity comparison",
        "",
        f"Device: `{settings['device']}`  ",
        f"Seed: `{settings['seed']}`  ",
        f"Maximum new tokens: `{settings['max_new_tokens']}`",
        "",
        "Each case uses the same generation settings and seed. Relative to the baseline,",
        "only the named input field changes. Inspect whether the content changes in a",
        "way that fits the sign, category, or date; different wording alone is not proof.",
        "",
    ]
    for result in results:
        case = result["case"]
        lines.extend([
            f"## {result['model']} — {case['name']}",
            "",
            f"Sign: {case['sign']} | Category: {case['category']} | Date: {case['date']}",
            "",
            result["output"] or "*(No text generated)*",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/input_sensitivity.json",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    device = choose_device(args.device)
    settings = {"device": device, "seed": args.seed, "max_new_tokens": args.max_new_tokens}
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.is_dir():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        print(f"Loading {model_name} on {device}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype="auto"
        ).to(device)
        model.eval()

        for case in CASES:
            prompt = make_prompt(case)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            results.append({"model": model_name, "case": case, "output": output})
            print(f"  {case['name']}: {output[:100]!r}", flush=True)

        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    payload = {"settings": settings, "results": results}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = args.output.with_suffix(".md")
    write_report(report_path, results, settings)
    print(f"Saved {args.output} and {report_path}")


if __name__ == "__main__":
    main()
