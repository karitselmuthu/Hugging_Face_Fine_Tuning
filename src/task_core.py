"""Shared prompt, data, and model helpers for reusable text tasks."""

import json
import re
import string
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_task(path):
    task = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise ValueError("Task config must be a JSON object")
    required = ("name", "source", "prompt_template", "response_field", "base_model", "max_length", "training")
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(f"Task config is missing: {', '.join(missing)}")
    if not isinstance(task["name"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", task["name"]):
        raise ValueError("Task name must use lowercase letters, digits, and underscores")
    if not isinstance(task["source"], dict) or not isinstance(task["source"].get("dataset"), str) or not task["source"]["dataset"].strip():
        raise ValueError("source.dataset must be a nonempty string")
    source = task["source"]
    for key in ("train_split", "validation_split", "test_split", "config", "group_field", "revision"):
        if source.get(key) is not None and (not isinstance(source[key], str) or not source[key].strip()):
            raise ValueError(f"source.{key} must be a nonempty string or null")
    if "heldout_fraction" in source and (not isinstance(source["heldout_fraction"], (int, float))
                                         or isinstance(source["heldout_fraction"], bool)
                                         or not 0 < source["heldout_fraction"] < 1):
        raise ValueError("source.heldout_fraction must be between 0 and 1")
    if "data_files" in source and not isinstance(source["data_files"], (str, list, dict)):
        raise ValueError("source.data_files must be a path, list, or split-to-path mapping")
    for key in ("prompt_template", "response_field", "base_model"):
        if not isinstance(task[key], str) or not task[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if type(task["max_length"]) is not int or task["max_length"] < 32:
        raise ValueError("max_length must be at least 32")
    fields = []
    try:
        for _, field, spec, conversion in string.Formatter().parse(task["prompt_template"]):
            if field is not None:
                if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", field) or spec or conversion:
                    raise ValueError("Prompt placeholders must be simple field names")
                fields.append(field)
    except ValueError as exc:
        raise ValueError(f"Invalid prompt_template: {exc}") from exc
    if not fields:
        raise ValueError("Prompt template needs at least one input field")
    task["input_fields"] = list(dict.fromkeys(fields))
    transforms = task.get("field_transforms", {})
    if not isinstance(transforms, dict) or any(field not in task["input_fields"] or mode not in ("strip", "title", "lower", "join_comma")
                                               for field, mode in transforms.items()):
        raise ValueError("field_transforms must map prompt fields to strip, title, lower, or join_comma")
    if not isinstance(task.get("response_map", {}), dict):
        raise ValueError("response_map must be a JSON object")
    settings = task["training"]
    if not isinstance(settings, dict) or settings.get("method") not in ("full", "lora"):
        raise ValueError("training.method must be full or lora")
    for key in ("train_samples", "validation_samples", "batch_size", "gradient_accumulation_steps"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "epochs"):
        if type(settings.get(key)) not in (int, float) or settings[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if settings["method"] == "lora":
        for key in ("lora_rank", "lora_alpha"):
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(f"training.{key} must be a positive integer")
        if type(settings.get("lora_dropout")) not in (int, float) or not 0 <= settings["lora_dropout"] < 1:
            raise ValueError("training.lora_dropout must be between 0 and 1")
        targets = settings.get("lora_targets")
        if not isinstance(targets, list) or not targets or any(not isinstance(value, str) or not value.strip() for value in targets):
            raise ValueError("training.lora_targets must be a nonempty list of module names")
    if "gradient_checkpointing" in settings and type(settings["gradient_checkpointing"]) is not bool:
        raise ValueError("training.gradient_checkpointing must be true or false")
    if "torch_empty_cache_steps" in settings and (type(settings["torch_empty_cache_steps"]) is not int or settings["torch_empty_cache_steps"] < 1):
        raise ValueError("training.torch_empty_cache_steps must be a positive integer")
    evaluation = task.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("metric") not in (None, "label_accuracy", "concept_coverage_exact", "rouge_l_f1"):
        raise ValueError("evaluation.metric is unsupported")
    if evaluation.get("metric") == "label_accuracy":
        labels = evaluation.get("labels")
        if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("evaluation.labels must be a nonempty list of unique labels")
    for key in ("generation_examples", "max_new_tokens"):
        if key in evaluation and (type(evaluation[key]) is not int or evaluation[key] < 1):
            raise ValueError(f"evaluation.{key} must be a positive integer")
    return task


def preparation_spec(task):
    """Fields that determine the prepared prompts, responses, and splits."""
    return {key: task.get(key) for key in ("name", "source", "prompt_template", "response_field", "response_map", "field_transforms")}


def format_prompt(task, values):
    missing = [field for field in task["input_fields"] if field not in values]
    if missing:
        raise ValueError(f"Missing prompt fields: {', '.join(missing)}")
    cleaned = {}
    for field in task["input_fields"]:
        raw = values[field]
        transform = task.get("field_transforms", {}).get(field)
        if transform == "join_comma":
            if not isinstance(raw, list):
                raise ValueError(f"Expected a list for {field}")
            value = ", ".join(str(item).strip() for item in raw)
        else:
            value = str(raw).strip()
        if not value:
            raise ValueError(f"Empty prompt field: {field}")
        if transform == "title":
            value = value.title()
        elif transform == "lower":
            value = value.lower()
        elif transform not in (None, "strip", "join_comma"):
            raise ValueError(f"Unsupported transform for {field}: {transform}")
        cleaned[field] = value
    return task["prompt_template"].format_map(cleaned)


def prepared_dir(task):
    return ROOT / "data" / "tasks" / task["name"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                    raise ValueError(f"{path}:{number}: expected prompt and response strings")
                yield row


def encode_example(row, tokenizer, max_length):
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(row["response"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(response_ids) > max_length:
        return None
    input_ids = prompt_ids + response_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
    }


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_saved_run(run_dir, device="auto"):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    final_dir = run_dir / "final"
    tokenizer = AutoTokenizer.from_pretrained(final_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if summary["method"] == "lora":
        base = AutoModelForCausalLM.from_pretrained(summary["starting_model"], dtype="auto")
        model = PeftModel.from_pretrained(base, final_dir, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(final_dir, local_files_only=True, dtype="auto")
    model = model.to(choose_device(device))
    model.eval()
    return model, tokenizer, summary


class ResponseOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        inputs = [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [row["labels"] + [-100] * (width - len(row["labels"])) for row in features],
            dtype=torch.long,
        )
        return batch
