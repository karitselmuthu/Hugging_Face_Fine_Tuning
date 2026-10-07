"""Bind Trainer checkpoints to the same task, data, and training settings."""

import json
from pathlib import Path


def checkpoint_spec(task, artifacts, *, model, revision, method, train_limit, validation_limit, max_steps, seed, balanced, save_steps):
    return {"task": task, "prepared_artifacts": artifacts, "model": model, "revision": revision,
            "method": method, "train_limit": train_limit, "validation_limit": validation_limit,
            "max_steps": max_steps, "seed": seed, "balanced": balanced, "save_steps": save_steps}


def resolve_checkpoint(output_dir, request, expected):
    output_dir = Path(output_dir)
    guard = output_dir / "resume_guard.json"
    if not guard.is_file() or json.loads(guard.read_text(encoding="utf-8")) != expected:
        raise ValueError("Checkpoint task, data, or training settings differ from this run")
    if request == "latest":
        candidates = sorted(output_dir.glob("checkpoint-*"),
                            key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1)
        if not candidates:
            raise ValueError(f"No Trainer checkpoint in {output_dir}")
        checkpoint = candidates[-1]
    else:
        checkpoint = Path(request)
    if checkpoint.resolve().parent != output_dir.resolve() or not checkpoint.is_dir():
        raise ValueError("Resume checkpoint must be an existing child of the output directory")
    if not (checkpoint / "trainer_state.json").is_file():
        raise ValueError(f"Not a complete Trainer checkpoint: {checkpoint}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if expected["max_steps"] > 0 and state.get("global_step", 0) >= expected["max_steps"]:
        raise ValueError(f"Checkpoint already reached max_steps={expected['max_steps']}; start a new run")
    return checkpoint


def write_guard(output_dir, expected):
    (Path(output_dir) / "resume_guard.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
