"""Format versioned prompts and encode response-only training examples."""


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
