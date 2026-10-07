"""One generation path for saved-run inference and evaluation."""

import torch


def generate_text(model, tokenizer, prompt, max_length, settings, device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length >= max_length:
        raise ValueError(f"Prompt exceeds the task's {max_length}-token context limit")
    torch.manual_seed(settings["seed"])
    kwargs = {
        "max_new_tokens": min(settings["max_new_tokens"], max_length - prompt_length),
        "do_sample": settings["temperature"] > 0,
        "repetition_penalty": settings["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs.update(temperature=settings["temperature"], top_p=settings["top_p"])
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
