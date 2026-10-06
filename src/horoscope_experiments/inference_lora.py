"""Generate a horoscope with a saved LoRA or QLoRA adapter."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_ADAPTER = ROOT / "models/smollm-horoscope-lora-512/final"


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--sign", default="Aries")
    parser.add_argument("--category", default="General")
    parser.add_argument("--date", default="2026/10/04")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not args.adapter_path.joinpath("adapter_config.json").is_file():
        parser.error(f"no saved adapter found at {args.adapter_path}")

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path, local_files_only=True).to(device)
    model.eval()
    prompt = f"""### Instruction:
Generate a horoscope using the following information.

### Sign:
{args.sign}

### Category:
{args.category}

### Date:
{args.date}

### Horoscope:
"""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(response_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
