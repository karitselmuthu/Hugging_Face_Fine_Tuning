"""Fine-tune a configured text task using full training or PEFT LoRA."""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from task_core import ROOT, ResponseOnlyCollator, encode_example, load_task, prepared_dir, read_jsonl


def select_encoded(path, tokenizer, max_length, limit, seed):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No examples in {path}")
    import random
    random.Random(seed).shuffle(rows)
    selected = []
    skipped = 0
    for row in rows:
        encoded = encode_example(row, tokenizer, max_length)
        if encoded is None:
            skipped += 1
            continue
        selected.append(encoded)
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError(f"No examples fit in {max_length} tokens; increase max_length")
    return Dataset.from_list(selected), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--starting-model", help="Base or saved full model; defaults to task base_model")
    parser.add_argument("--method", choices=["full", "lora"])
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value for a short smoke run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    task = load_task(args.task)
    settings = task["training"]
    method = args.method or settings["method"]
    train_limit = args.train_samples if args.train_samples is not None else settings["train_samples"]
    validation_limit = args.validation_samples if args.validation_samples is not None else settings["validation_samples"]
    if min(train_limit, validation_limit) < 1 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("sample limits must be positive and max-steps must be -1 or positive")
    data_dir = args.data_dir or prepared_dir(task)
    run_name = f"{method}-{train_limit}"
    if args.max_steps != -1:
        run_name += f"-steps-{args.max_steps}"
    output_dir = args.output_dir or ROOT / "models" / "tasks" / task["name"] / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    for name in ("train", "validation", "test"):
        if not (data_dir / f"{name}.jsonl").is_file():
            parser.error(f"Missing prepared {name} data: {data_dir}")
    prepared = load_task(data_dir / "task_config.json")
    if prepared["prompt_template"] != task["prompt_template"] or prepared["name"] != task["name"]:
        parser.error("Prepared data uses a different task prompt; rerun prepare_task.py")

    starting_model = args.starting_model or task["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(starting_model)
    tokenizer.pad_token = tokenizer.eos_token
    train, train_skipped = select_encoded(data_dir / "train.jsonl", tokenizer, task["max_length"], train_limit, args.seed)
    validation, validation_skipped = select_encoded(data_dir / "validation.jsonl", tokenizer, task["max_length"], validation_limit, args.seed)
    model = AutoModelForCausalLM.from_pretrained(starting_model, dtype="auto")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if method == "lora":
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=settings["lora_rank"],
            lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"],
            target_modules=settings["lora_targets"],
            bias="none",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        gradient_checkpointing=settings.get("gradient_checkpointing", False),
        torch_empty_cache_steps=settings.get("torch_empty_cache_steps"),
        num_train_epochs=settings["epochs"],
        max_steps=args.max_steps,
        learning_rate=settings["learning_rate"],
        logging_steps=max(1, min(25, len(train) // 4)),
        eval_strategy="no",
        save_strategy="no",
        dataloader_pin_memory=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=ResponseOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    print(f"Training {task['name']} with {method}: {len(train)} train, {len(validation)} validation; device={training_args.device}", flush=True)
    train_metrics = trainer.train().metrics
    validation_metrics = trainer.evaluate()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {
        "task": task["name"], "method": method, "starting_model": starting_model,
        "train_examples": len(train), "validation_examples": len(validation),
        "train_skipped_for_length": train_skipped,
        "validation_skipped_for_length": validation_skipped,
        "max_length": task["max_length"], "seed": args.seed,
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "final_dir": str(final_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "task_config.json").write_text(args.task.read_text(encoding="utf-8"))
    print(f"Saved {method} model to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
