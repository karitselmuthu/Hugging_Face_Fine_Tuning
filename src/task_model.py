"""Load saved models and collate response-only batches."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_lineage import load_run_manifest


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
    manifest = load_run_manifest(run_dir)
    final_dir = run_dir / "final"
    tokenizer = AutoTokenizer.from_pretrained(final_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if summary["method"] == "lora":
        revision = manifest["starting_model"]["resolved_revision"] if manifest else None
        base = AutoModelForCausalLM.from_pretrained(summary["starting_model"], dtype="auto", revision=revision,
                                                   trust_remote_code=False, use_safetensors=True)
        model = PeftModel.from_pretrained(base, final_dir, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(final_dir, local_files_only=True, dtype="auto",
                                                     trust_remote_code=False, use_safetensors=True)
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
