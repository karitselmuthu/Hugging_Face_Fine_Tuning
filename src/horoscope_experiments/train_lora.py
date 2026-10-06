"""Train a response-only LoRA adapter, or QLoRA on a supported CUDA GPU."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_LENGTH = 256
SEED = 42
MARKER = "### Horoscope:\n"


def tokenize_example(example, tokenizer):
    prompt_part, marker, response_part = example["text"].partition(MARKER)
    if not marker:
        raise ValueError("Prepared example has no horoscope marker")
    prompt_ids = tokenizer(prompt_part + marker, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(
        response_part.strip() + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]
    input_ids = (prompt_ids + response_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_LENGTH]
    if all(label == -100 for label in labels):
        raise ValueError("No response tokens remain after truncation")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class ResponseOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        inputs = [
            {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}
            for row in features
        ]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [row["labels"] + [-100] * (width - len(row["labels"])) for row in features],
            dtype=torch.long,
        )
        return batch


def load_model(method):
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QLoRA in this script requires a supported CUDA GPU and bitsandbytes. "
                "Use --method lora on Apple Silicon/MPS."
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install requirements-qlora.txt in the CUDA environment") from exc
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
            dtype=compute_dtype,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype="auto")

    model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=-1, help="Use a small value for a smoke run")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.train_samples, args.eval_samples, args.gradient_accumulation_steps) < 1:
        parser.error("sample counts and gradient accumulation must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps must be -1 or positive")
    if args.method == "qlora" and not torch.cuda.is_available():
        parser.error("QLoRA requires a supported CUDA GPU; use --method lora on this Mac")

    run_name = f"smollm-horoscope-{args.method}-{args.train_samples}"
    if args.max_steps != -1:
        run_name += f"-steps-{args.max_steps}"
    output_dir = args.output_dir or ROOT / "models" / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    train = load_from_disk(str(ROOT / "data/processed/train"))
    test = load_from_disk(str(ROOT / "data/processed/test"))
    if args.train_samples > len(train) or args.eval_samples > len(test):
        parser.error("requested more examples than the saved splits contain")
    train = train.shuffle(seed=SEED).select(range(args.train_samples))
    test = test.shuffle(seed=SEED).select(range(args.eval_samples))
    train = train.map(lambda row: tokenize_example(row, tokenizer), remove_columns=train.column_names)
    test = test.map(lambda row: tokenize_example(row, tokenizer), remove_columns=test.column_names)

    model = load_model(args.method)
    model.config.pad_token_id = tokenizer.pad_token_id
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        logging_steps=max(1, min(25, args.train_samples // 4)),
        eval_strategy="no",
        save_strategy="no",
        dataloader_pin_memory=False,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=ResponseOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    print(f"Training {args.method} on {len(train)} examples; device={training_args.device}", flush=True)
    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()
    adapter_dir = output_dir / "final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "method": args.method,
        "base_model": BASE_MODEL,
        "train_samples": len(train),
        "eval_samples": len(test),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "adapter_dir": str(adapter_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved adapter to {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
