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
    required = ("name", "source", "prompt_template", "response_field", "base_model", "max_length", "training")
    missing = [key for key in required if key not in task]
    if missing:
        raise ValueError(f"Task config is missing: {', '.join(missing)}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task["name"]):
        raise ValueError("Task name must use lowercase letters, digits, and underscores")
    if int(task["max_length"]) < 32:
        raise ValueError("max_length must be at least 32")
    fields = []
    for _, field, spec, conversion in string.Formatter().parse(task["prompt_template"]):
        if field is not None:
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", field) or spec or conversion:
                raise ValueError("Prompt placeholders must be simple field names")
            fields.append(field)
    if not fields:
        raise ValueError("Prompt template needs at least one input field")
    task["input_fields"] = list(dict.fromkeys(fields))
    return task


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
