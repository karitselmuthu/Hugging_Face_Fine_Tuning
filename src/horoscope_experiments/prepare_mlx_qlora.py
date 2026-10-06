"""Export the saved response-only split for MLX QLoRA on Apple Silicon."""

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/smollm2-135m-mlx-4bit"
MARKER = "### Horoscope:\n"
# Keep the prompt format used by the PyTorch experiments. MLX-LM's
# completions loader calls apply_chat_template, so this template simply
# concatenates the prompt and response and appends EOS to the response.
RAW_COMPLETION_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)


def export_rows(dataset, path):
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            prompt, marker, completion = row["text"].partition(MARKER)
            if not marker:
                raise ValueError("Prepared example has no horoscope marker")
            output.write(
                json.dumps(
                    {"prompt": prompt + marker, "completion": completion.strip()},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--valid-samples", type=int, default=8)
    parser.add_argument("--test-samples", type=int, default=None,
                        help="Held-out examples for final scoring (defaults to validation count)")
    parser.add_argument("--test-offset", type=int, default=0,
                        help="Start index in the seed-42 shuffled held-out split")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mlx-qlora-smoke")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if args.test_samples is None:
        args.test_samples = args.valid_samples
    if min(args.train_samples, args.valid_samples, args.test_samples) < 1 or args.test_offset < 0:
        parser.error("sample counts must be positive")

    model_config = args.model_dir / "config.json"
    tokenizer_config = args.model_dir / "tokenizer_config.json"
    if not model_config.is_file() or not tokenizer_config.is_file():
        parser.error(f"converted MLX model not found at {args.model_dir}")
    quantization = json.loads(model_config.read_text())["quantization"]
    if quantization.get("bits") != 4:
        parser.error("the MLX model must have 4-bit quantized weights")
    tokenizer_settings = json.loads(tokenizer_config.read_text())
    tokenizer_settings["chat_template"] = RAW_COMPLETION_TEMPLATE
    tokenizer_config.write_text(json.dumps(tokenizer_settings, indent=2) + "\n")

    train = load_from_disk(str(ROOT / "data/processed/train"))
    valid = load_from_disk(str(ROOT / "data/processed/test"))
    if args.train_samples > len(train) or max(args.valid_samples, args.test_offset + args.test_samples) > len(valid):
        parser.error("requested more rows than the saved splits contain")
    train = train.shuffle(seed=42).select(range(args.train_samples))
    heldout = valid.shuffle(seed=42)
    valid = heldout.select(range(args.valid_samples))
    test = heldout.select(range(args.test_offset, args.test_offset + args.test_samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_rows(train, args.output_dir / "train.jsonl")
    export_rows(valid, args.output_dir / "valid.jsonl")
    export_rows(test, args.output_dir / "test.jsonl")
    print(f"Wrote {len(train)} train, {len(valid)} validation, and {len(test)} test examples to {args.output_dir}")


if __name__ == "__main__":
    main()
