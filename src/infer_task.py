"""Generate an answer with a saved task run and named prompt inputs."""

import argparse
import json
from pathlib import Path

import torch

from task_core import choose_device, format_prompt, load_saved_run, load_task


def parse_inputs(items):
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Input must be field=value: {item}")
        field, value = item.split("=", 1)
        values[field] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", action="append", default=[], help="Repeat field=value for every prompt field")
    parser.add_argument("--input-json", type=Path, help="JSON object with prompt field values")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty <= 0:
        parser.error("Check generation settings: max tokens > 0, temperature >= 0, 0 < top-p <= 1, repetition penalty > 0")
    task = load_task(args.run_dir / "task_config.json")
    try:
        values = json.loads(args.input_json.read_text()) if args.input_json else {}
        values.update(parse_inputs(args.input))
        prompt = format_prompt(task, values)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    model, tokenizer, summary = load_saved_run(args.run_dir, args.device)
    if summary["task"] != task["name"]:
        parser.error("Saved run and task config do not match")
    device = choose_device(args.device)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    if inputs["input_ids"].shape[1] >= task["max_length"]:
        parser.error(f"Prompt exceeds the task's {task['max_length']}-token context limit")
    torch.manual_seed(args.seed)
    kwargs = {
        "max_new_tokens": min(args.max_new_tokens, task["max_length"] - inputs["input_ids"].shape[1]),
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(response_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
