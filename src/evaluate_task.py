"""Score a task validation or test split and save representative generations."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_core import choose_device, encode_example, load_saved_run, load_task, prepared_dir, read_jsonl
from task_metrics import score_generations


def select_balanced_rows(rows, labels, per_label, seed):
    """Select the same number of held-out rows for each configured label."""
    rng = random.Random(seed)
    selected = []
    for label in labels:
        candidates = [row for row in rows if row["response"].strip().lower() == label]
        if len(candidates) < per_label:
            raise ValueError(f"Need {per_label} test examples for {label}; found {len(candidates)}")
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--test-samples", type=int, help="Number of rows for random sampling (default: 128)")
    parser.add_argument("--generation-examples", type=int)
    parser.add_argument("--balanced-per-label", type=int,
                        help="For label tasks, score and generate this many test examples per label")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-only", action="store_true", help="Score the starting model without the saved adapter")
    args = parser.parse_args()
    task = load_task(args.run_dir / "task_config.json")
    evaluation = task.get("evaluation", {})
    balanced = args.balanced_per_label is not None
    if balanced:
        if evaluation.get("metric") != "label_accuracy" or not evaluation.get("labels"):
            parser.error("--balanced-per-label requires a label_accuracy task with labels")
        if args.balanced_per_label < 1:
            parser.error("--balanced-per-label must be positive")
        if args.test_samples is not None:
            parser.error("Use --balanced-per-label without --test-samples")
        test_limit = args.balanced_per_label * len(evaluation["labels"])
        if args.generation_examples is not None and args.generation_examples != test_limit:
            parser.error("Balanced evaluation must generate every selected example")
        generation_examples = test_limit
    else:
        test_limit = args.test_samples if args.test_samples is not None else 128
        generation_examples = args.generation_examples if args.generation_examples is not None else evaluation.get("generation_examples", 5)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else evaluation.get("max_new_tokens", 80)
    if min(test_limit, generation_examples, max_new_tokens) < 1:
        parser.error("sample counts and max-new-tokens must be positive")
    data_dir = args.data_dir or prepared_dir(task)
    prepared = load_task(data_dir / "task_config.json")
    if prepared["name"] != task["name"] or prepared["prompt_template"] != task["prompt_template"]:
        parser.error("Prepared data and saved run use different prompts")
    if args.base_only:
        summary = json.loads((args.run_dir / "run_summary.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "final", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(summary["starting_model"], dtype="auto")
        model = model.to(choose_device(args.device))
        model.eval()
    else:
        model, tokenizer, summary = load_saved_run(args.run_dir, args.device)
    if summary["task"] != task["name"]:
        parser.error("Saved run and task config do not match")
    device = choose_device(args.device)
    test_rows = list(read_jsonl(data_dir / f"{args.split}.jsonl"))
    if balanced:
        try:
            test_rows = select_balanced_rows(test_rows, evaluation["labels"], args.balanced_per_label, summary["seed"])
        except ValueError as exc:
            parser.error(str(exc))
    else:
        random.Random(summary["seed"]).shuffle(test_rows)
    total_loss, total_tokens, scored, skipped = 0.0, 0, 0, 0
    examples = []
    for row in test_rows:
        encoded = encode_example(row, tokenizer, task["max_length"])
        if encoded is None:
            skipped += 1
            continue
        ids = torch.tensor([encoded["input_ids"]], device=device)
        labels = torch.tensor([encoded["labels"]], device=device)
        with torch.inference_mode():
            loss = model(input_ids=ids, labels=labels).loss.item()
        tokens = sum(value != -100 for value in encoded["labels"][1:])
        total_loss += loss * tokens
        total_tokens += tokens
        scored += 1
        if len(examples) < generation_examples:
            torch.manual_seed(summary["seed"] + len(examples))
            prompt_ids = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **prompt_ids,
                    max_new_tokens=min(max_new_tokens, task["max_length"] - prompt_ids["input_ids"].shape[1]),
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            response_ids = output[0][prompt_ids["input_ids"].shape[1]:]
            examples.append({"prompt": row["prompt"], "reference": row["response"], "inputs": row.get("inputs", {}),
                             "output": tokenizer.decode(response_ids, skip_special_tokens=True).strip()})
        if scored == test_limit:
            break
    if not total_tokens:
        raise ValueError("No test examples fit within max_length")
    if balanced and scored != test_limit:
        raise ValueError("Balanced evaluation lost examples to the context limit; reduce --balanced-per-label or raise max_length")
    result = {
        "task": task["name"], "run_dir": str(args.run_dir), "device": device,
        "split": args.split,
        "model": "starting_model" if args.base_only else "trained_run",
        "sampling": {"method": "balanced_per_label", "per_label": args.balanced_per_label, "seed": summary["seed"]}
                    if balanced else {"method": "random", "seed": summary["seed"]},
        f"{args.split}_examples": scored, "skipped_for_length": skipped,
        "response_tokens": total_tokens, "response_loss": total_loss / total_tokens,
        "response_perplexity": math.exp(total_loss / total_tokens),
        "generation_metric": score_generations(task, examples),
        "examples": examples,
    }
    if balanced:
        default_name = f"balanced_{args.balanced_per_label}_" + ("base_" if args.base_only else "") + f"{args.split}_evaluation.json"
    else:
        default_name = ("base_" if args.base_only else "") + f"{args.split}_evaluation.json"
    output_path = args.output or args.run_dir / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.split.title()} loss: {result['response_loss']:.4f}; perplexity: {result['response_perplexity']:.2f}; examples: {scored}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
