"""Fine-tune a configured text task using full training or PEFT LoRA."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from audit_task_data import audit_prepared_data
from data_quality import audit_sequence_lengths
from run_lineage import write_run_manifest
from task_config import load_task, preparation_spec
from task_io import ROOT, prepared_dir, read_jsonl
from task_model import ResponseOnlyCollator
from task_prompts import encode_example
from training_resume import checkpoint_spec, resolve_checkpoint, write_guard


def select_encoded(path, tokenizer, max_length, limit, seed, balanced_labels=None):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No examples in {path}")
    selected = []
    skipped = 0
    label_counts = Counter()
    if balanced_labels is None:
        random.Random(seed).shuffle(rows)
        for row in rows:
            encoded = encode_example(row, tokenizer, max_length)
            if encoded is None:
                skipped += 1
                continue
            selected.append(encoded)
            if len(selected) == limit:
                break
    else:
        rng = random.Random(seed)
        per_label, extra = divmod(limit, len(balanced_labels))
        for index, label in enumerate(balanced_labels):
            required = per_label + (index < extra)
            candidates = [row for row in rows if row["response"].strip().lower() == label]
            rng.shuffle(candidates)
            if len(candidates) < required:
                raise ValueError(f"Need {required} training examples for {label}; found {len(candidates)}")
            for row in candidates:
                if label_counts[label] == required:
                    break
                encoded = encode_example(row, tokenizer, max_length)
                if encoded is None:
                    skipped += 1
                    continue
                selected.append(encoded)
                label_counts[label] += 1
            if label_counts[label] < required:
                raise ValueError(f"Only {label_counts[label]} examples for {label} fit within {max_length} tokens; need {required}")
        rng.shuffle(selected)
    if not selected:
        raise ValueError(f"No examples fit in {max_length} tokens; increase max_length")
    return Dataset.from_list(selected), skipped, dict(label_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--starting-model", help="Base or saved full model; defaults to task base_model")
    parser.add_argument("--starting-model-revision", help="Immutable commit when overriding a Hub starting model")
    parser.add_argument("--method", choices=["full", "lora"])
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value for a short smoke run")
    parser.add_argument("--resume-from-checkpoint", help="Use latest or an existing checkpoint path inside --output-dir")
    parser.add_argument("--save-steps", type=int, help="Checkpoint interval (default: task setting or 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-train", action="store_true",
                        help="For label tasks, draw nearly equal training counts per label without replacement")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="Allow exact prompts shared across splits after showing audit counts")
    args = parser.parse_args()
    task = load_task(args.task)
    settings = task["training"]
    method = args.method or settings["method"]
    train_limit = args.train_samples if args.train_samples is not None else settings["train_samples"]
    validation_limit = args.validation_samples if args.validation_samples is not None else settings["validation_samples"]
    save_steps = args.save_steps if args.save_steps is not None else settings.get("save_steps", 100)
    if min(train_limit, validation_limit, save_steps) < 1 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("sample limits and save-steps must be positive; max-steps must be -1 or positive")
    labels = None
    if args.balanced_train:
        evaluation = task.get("evaluation", {})
        labels = evaluation.get("labels")
        if evaluation.get("metric") != "label_accuracy" or not labels:
            parser.error("--balanced-train requires a label_accuracy task with labels")
        if train_limit < len(labels):
            parser.error("--balanced-train requires at least one training example per label")
    data_dir = args.data_dir or prepared_dir(task)
    run_name = f"{method}-{train_limit}"
    if args.balanced_train:
        run_name += "-balanced"
    if args.max_steps != -1:
        run_name += f"-steps-{args.max_steps}"
    output_dir = args.output_dir or ROOT / "models" / "tasks" / task["name"] / run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_from_checkpoint:
        parser.error(f"output directory is not empty: {output_dir}")
    if args.resume_from_checkpoint and not output_dir.is_dir():
        parser.error(f"cannot resume: output directory does not exist: {output_dir}")
    for name in ("train", "validation", "test"):
        if not (data_dir / f"{name}.jsonl").is_file():
            parser.error(f"Missing prepared {name} data: {data_dir}")
    prepared = load_task(data_dir / "task_config.json")
    if preparation_spec(prepared) != preparation_spec(task):
        parser.error("Prepared data uses different source, fields, or prompt settings; rerun prepare_task.py")
    try:
        audit = audit_prepared_data(data_dir, fail_on_overlap=not args.allow_overlap)
    except ValueError as exc:
        parser.error(str(exc))
    prepared_artifacts = audit["artifacts"]
    if any(audit["prompt_overlap"].values()):
        print(f"Allowed cross-split prompt overlap: {audit['prompt_overlap']}", flush=True)

    starting_model = args.starting_model or task["base_model"]
    revision = args.starting_model_revision or (task.get("base_model_revision") if args.starting_model is None else None)
    config = AutoConfig.from_pretrained(starting_model, revision=revision, trust_remote_code=False)
    model_commit = getattr(config, "_commit_hash", None) or revision
    if model_commit is None and not Path(starting_model).exists():
        parser.error("Hub model revision could not be resolved; set --starting-model-revision to an immutable commit")
    tokenizer = AutoTokenizer.from_pretrained(starting_model, revision=model_commit,
                                              trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    token_lengths = audit_sequence_lengths(data_dir, tokenizer, task["max_length"])
    max_fraction = task.get("data_quality", {}).get("max_overlength_fraction", 1)
    exceeded = {split: item for split, item in token_lengths.items() if item["fraction_over"] > max_fraction}
    if exceeded:
        parser.error(f"Rows above {task['max_length']} tokens exceed data_quality.max_overlength_fraction "
                     f"{max_fraction}: {exceeded}")
    print(f"Sequence-length audit: {json.dumps(token_lengths)}", flush=True)
    train, train_skipped, train_label_counts = select_encoded(
        data_dir / "train.jsonl", tokenizer, task["max_length"], train_limit, args.seed, labels)
    validation, validation_skipped, _ = select_encoded(
        data_dir / "validation.jsonl", tokenizer, task["max_length"], validation_limit, args.seed)
    model = AutoModelForCausalLM.from_pretrained(starting_model, dtype="auto", revision=model_commit,
                                                trust_remote_code=False, use_safetensors=True)
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
    guard = checkpoint_spec(task, prepared_artifacts, model=starting_model, revision=model_commit,
                            method=method, train_limit=train_limit, validation_limit=validation_limit,
                            max_steps=args.max_steps, seed=args.seed, balanced=args.balanced_train,
                            save_steps=save_steps)
    if args.resume_from_checkpoint:
        try:
            checkpoint = resolve_checkpoint(output_dir, args.resume_from_checkpoint, guard)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        checkpoint = None
        output_dir.mkdir(parents=True, exist_ok=True)
        write_guard(output_dir, guard)
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
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
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
    train_metrics = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None).metrics
    validation_metrics = trainer.evaluate()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    weights = list(final_dir.glob("*.safetensors"))
    unsafe_weights = [path for path in final_dir.glob("*.bin") if path.name != "training_args.bin"]
    unsafe_weights += list(final_dir.glob("*.pt"))
    if not weights or unsafe_weights:
        raise RuntimeError(f"Final model must contain safetensors weights only: {final_dir}")
    summary = {
        "task": task["name"], "method": method, "starting_model": starting_model,
        "starting_model_revision": revision,
        "tokenizer_revision": model_commit,
        "train_examples": len(train), "validation_examples": len(validation),
        "train_skipped_for_length": train_skipped,
        "validation_skipped_for_length": validation_skipped,
        "train_sampling": {"method": "balanced_per_label" if args.balanced_train else "random",
                           "label_counts": train_label_counts} if args.balanced_train else {"method": "random"},
        "max_length": task["max_length"], "seed": args.seed,
        "max_steps": args.max_steps,
        "save_steps": save_steps,
        "resumed_from_checkpoint": str(checkpoint) if checkpoint else None,
        "prepared_artifacts": prepared_artifacts,
        "data_audit": {"quality": audit["quality"], "token_lengths": token_lengths},
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "final_dir": str(final_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "task_config.json").write_text(args.task.read_text(encoding="utf-8"))
    write_run_manifest(output_dir, task, summary, data_dir, model_commit)
    print(f"Saved {method} model to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
