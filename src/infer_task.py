"""Generate an answer with a saved task run and named prompt inputs."""

import argparse
import json
from pathlib import Path

from run_lineage import load_run_manifest
from task_core import choose_device, format_prompt, load_saved_run, load_task
from task_generation import generate_text


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
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    manifest = load_run_manifest(args.run_dir)
    settings = dict(manifest["decoding_defaults"]) if manifest else {
        "max_new_tokens": 80, "temperature": 0.8, "top_p": 0.9,
        "repetition_penalty": 1.1, "seed": 42}
    for name in settings:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    if settings["max_new_tokens"] < 1 or settings["temperature"] < 0 or not 0 < settings["top_p"] <= 1 or settings["repetition_penalty"] <= 0:
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
    try:
        print(generate_text(model, tokenizer, prompt, task["max_length"], settings, device))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
