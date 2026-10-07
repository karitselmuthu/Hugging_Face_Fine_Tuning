"""Validate task configuration and identify preparation-affecting settings."""

import json
import re
import string
from pathlib import Path


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
    if task.get("base_model_revision") is not None and (not isinstance(task["base_model_revision"], str)
                                                        or not task["base_model_revision"].strip()):
        raise ValueError("base_model_revision must be a nonempty string")
    if not isinstance(task.get("prompt_version", "v1"), str) or not task.get("prompt_version", "v1").strip():
        raise ValueError("prompt_version must be a nonempty string")
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
    if "save_steps" in settings and (type(settings["save_steps"]) is not int or settings["save_steps"] < 1):
        raise ValueError("training.save_steps must be a positive integer")
    quality = task.get("data_quality", {})
    if not isinstance(quality, dict):
        raise ValueError("data_quality must be an object")
    for key in ("fail_on_near_duplicate", "fail_on_pii", "fail_on_secret"):
        if key in quality and type(quality[key]) is not bool:
            raise ValueError(f"data_quality.{key} must be true or false")
    fraction = quality.get("max_overlength_fraction", 1)
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
        raise ValueError("data_quality.max_overlength_fraction must be between 0 and 1")
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
    promotion = task.get("promotion")
    if promotion is not None:
        if not isinstance(promotion, dict) or not promotion.get("minimum_metrics"):
            raise ValueError("promotion.minimum_metrics must define at least one threshold")
        if "minimum_examples" in promotion and (type(promotion["minimum_examples"]) is not int or promotion["minimum_examples"] < 1):
            raise ValueError("promotion.minimum_examples must be a positive integer")
        for group in ("minimum_metrics", "minimum_lift"):
            values = promotion.get(group, {})
            if not isinstance(values, dict) or any(not isinstance(name, str) or type(value) not in (int, float)
                                                    or not -1 <= value <= 1 for name, value in values.items()):
                raise ValueError(f"promotion.{group} must map metric names to thresholds between -1 and 1")
        invalid = promotion.get("maximum_invalid_label_outputs", 0)
        if type(invalid) is not int or invalid < 0:
            raise ValueError("promotion.maximum_invalid_label_outputs must be a nonnegative integer")
    return task


def preparation_spec(task):
    """Fields that determine the prepared prompts, responses, and splits."""
    spec = {key: task.get(key) for key in ("name", "source", "prompt_template", "response_field", "response_map", "field_transforms")}
    spec["prompt_version"] = task.get("prompt_version", "v1")
    return spec
