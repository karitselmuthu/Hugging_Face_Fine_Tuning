"""Score saved response-only models on held-out horoscopes and compare outputs."""

import argparse
import json
import math
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = PROJECT_ROOT / "data/processed/test"
MODEL_PATHS = {
    "response-only-512": PROJECT_ROOT / "models/smollm-horoscope-response-only-512/final",
    "response-only-2000": PROJECT_ROOT / "models/smollm-horoscope-response-only-2000/final",
}
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
LORA_PATH = PROJECT_ROOT / "models/smollm-horoscope-lora-512/final"
MAX_LENGTH = 256  # Match response-only training.
HOROSCOPE_MARKER = "### Horoscope:\n"
FIELD_PATTERN = re.compile(
    r"### Sign:\n(?P<sign>[^\n]+)\n\n"
    r"### Category:\n(?P<category>[^\n]+)\n\n"
    r"### Date:\n(?P<date>[^\n]+)"
)


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_example(text):
    prompt, marker, response = text.partition(HOROSCOPE_MARKER)
    if not marker:
        raise ValueError("Dataset example has no horoscope marker")
    fields = FIELD_PATTERN.search(prompt)
    if not fields:
        raise ValueError("Dataset example is missing a sign, category, or date")
    return {"prompt": prompt + marker, "reference": response.strip(), **fields.groupdict()}


def select_generation_cases(dataset, seed, count):
    """Cover signs first, with categories spread across the selected examples."""
    examples = [parse_example(row["text"]) for row in dataset.shuffle(seed=seed)]
    signs = sorted({example["sign"] for example in examples})
    categories = sorted({example["category"] for example in examples})
    chosen = []
    for index, sign in enumerate(signs[:count]):
        wanted_category = categories[index % len(categories)]
        case = next(
            example for example in examples
            if example["sign"] == sign and example["category"] == wanted_category
        )
        chosen.append(case)
    return chosen


def score_example(model, tokenizer, example, device):
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(
        example["reference"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    input_ids = (prompt_ids + response_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_LENGTH]
    token_count = sum(label != -100 for label in labels[1:])
    if token_count == 0:
        return None
    input_tensor = torch.tensor([input_ids], device=device)
    label_tensor = torch.tensor([labels], device=device)
    with torch.inference_mode():
        loss = model(input_ids=input_tensor, labels=label_tensor).loss.item()
    return loss, token_count


def generate(model, tokenizer, example, device, seed, max_new_tokens):
    inputs = tokenizer(
        example["prompt"], return_tensors="pt", add_special_tokens=False
    ).to(device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def write_report(path, payload):
    lines = [
        "# Held-out horoscope evaluation",
        "",
        f"Test examples scored: {payload['settings']['score_examples']}  ",
        f"Maximum sequence length: {MAX_LENGTH}  ",
        f"Device: `{payload['settings']['device']}`  ",
        f"Generation seed: {payload['settings']['seed']}",
        "",
        "| Model | Response tokens scored | Mean response NLL | Response perplexity |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in payload["metrics"].items():
        lines.append(
            f"| {name} | {metrics['response_tokens']} | "
            f"{metrics['mean_response_nll']:.4f} | {metrics['response_perplexity']:.2f} |"
        )
    lines.extend([
        "",
        "NLL measures prediction of held-out reference tokens under the same",
        "response-only objective and 256-token truncation used for training.",
        "Lower is better for that measure. It does not measure whether a generated",
        "horoscope follows the requested sign, category, or exact date.",
        "Generated text is sampled and may stop at the token limit.",
        "",
        "## Generation review",
        "",
    ])
    for case in payload["generation_cases"]:
        lines.extend([
            f"### {case['sign']} / {case['category']} / {case['date']}",
            "",
            "**Held-out reference:** " + case["reference"],
            "",
        ])
        for name, output in case["outputs"].items():
            lines.extend([f"**{name}:** {output or '*(No text generated)*'}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--score-examples", type=int, default=128)
    parser.add_argument("--score-offset", type=int, default=0,
                        help="Start index in the seed-shuffled held-out split")
    parser.add_argument("--generation-examples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-lora", action="store_true", help="Compare the saved 512-example LoRA adapter")
    parser.add_argument("--models", nargs="+", choices=["response-only-512", "response-only-2000", "lora-512"],
                        help="Only evaluate the named saved models")
    parser.add_argument(
        "--output", type=Path
    )
    args = parser.parse_args()
    if min(args.score_examples, args.generation_examples, args.max_new_tokens) < 1 or args.score_offset < 0:
        parser.error("example counts and --max-new-tokens must be positive")

    dataset = load_from_disk(str(TEST_DATA_PATH))
    if args.score_offset + args.score_examples > len(dataset) or args.generation_examples > 12:
        parser.error("requested more examples than available (maximum 12 generation signs)")
    shuffled = dataset.shuffle(seed=args.seed)
    scoring_cases = [
        parse_example(row["text"])
        for row in shuffled.select(range(args.score_offset, args.score_offset + args.score_examples))
    ]
    generation_pool = shuffled.select(range(args.score_offset, args.score_offset + args.score_examples))
    generation_cases = select_generation_cases(generation_pool, args.seed, args.generation_examples)
    device = choose_device(args.device)
    metrics = {}
    model_paths = dict(MODEL_PATHS)
    if args.include_lora:
        model_paths["lora-512"] = LORA_PATH
    if args.models:
        model_paths = {name: model_paths.get(name, LORA_PATH) for name in args.models}
    output_path = args.output or PROJECT_ROOT / "results" / (
        "heldout_lora_comparison.json" if args.include_lora else "heldout_evaluation.json"
    )

    for name, model_path in model_paths.items():
        if not model_path.is_dir():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        print(f"Loading {name} on {device}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if name == "lora-512":
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, local_files_only=True, dtype="auto"
            )
            model = PeftModel.from_pretrained(base, model_path, local_files_only=True).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, dtype="auto"
            ).to(device)
        model.eval()

        total_nll = 0.0
        total_tokens = 0
        scored_examples = 0
        for index, example in enumerate(scoring_cases, start=1):
            score = score_example(model, tokenizer, example, device)
            if score is not None:
                loss, token_count = score
                total_nll += loss * token_count
                total_tokens += token_count
                scored_examples += 1
            if index % 32 == 0:
                print(f"  scored {index}/{len(scoring_cases)}", flush=True)
        if not total_tokens:
            raise ValueError("No response tokens remained after truncation")
        mean_nll = total_nll / total_tokens
        metrics[name] = {
            "examples_scored": scored_examples,
            "response_tokens": total_tokens,
            "mean_response_nll": mean_nll,
            "response_perplexity": math.exp(mean_nll),
        }

        for index, example in enumerate(generation_cases):
            output = generate(
                model, tokenizer, example, device, args.seed + index, args.max_new_tokens
            )
            example.setdefault("outputs", {})[name] = output
            print(f"  generated {index + 1}/{len(generation_cases)}", flush=True)

        del model, tokenizer
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "settings": {
            "device": device,
            "seed": args.seed,
            "score_examples": len(scoring_cases),
            "score_offset": args.score_offset,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": metrics,
        "generation_cases": generation_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report_path = output_path.with_suffix(".md")
    write_report(report_path, payload)
    print(f"Saved {output_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
