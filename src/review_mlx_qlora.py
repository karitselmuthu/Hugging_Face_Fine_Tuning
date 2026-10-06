"""Generate the same held-out review prompts used for the PyTorch models."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path,
                        default=ROOT / "models/smollm-horoscope-mlx-qlora-512")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/heldout_mlx_qlora_512.json")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "results/heldout_lora_comparison.json",
                        help="PyTorch evaluation JSON providing the matching prompts")
    args = parser.parse_args()

    prior = json.loads(args.source.read_text())
    model, tokenizer = load(str(ROOT / "models/smollm2-135m-mlx-4bit"),
                            adapter_path=str(args.adapter_path))
    sampler = make_sampler(temp=0.8, top_p=0.9)
    processors = make_logits_processors(repetition_penalty=1.1)
    cases = []
    for case in prior["generation_cases"]:
        mx.random.seed(42)
        prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
        output = generate(model, tokenizer, prompt_ids, max_tokens=80,
                          sampler=sampler, logits_processors=processors).strip()
        cases.append({key: case[key] for key in ("sign", "category", "date", "prompt", "reference")}
                     | {"output": output})
        print(f"{case['sign']} / {case['category']}: {output[:100]!r}")

    payload = {
        "model": "mlx-qlora-512",
        "adapter_path": str(args.adapter_path.relative_to(ROOT)),
        "source_cases": str(args.source.resolve().relative_to(ROOT)),
        "generation_settings": {"seed": 42, "max_tokens": 80, "temperature": 0.8,
                                "top_p": 0.9, "repetition_penalty": 1.1},
        "generation_cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
