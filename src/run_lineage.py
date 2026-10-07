"""Versioned run metadata and checks shared by training and consumers."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path


SCHEMA_VERSION = 2


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty():
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def package_versions():
    names = ("torch", "transformers", "datasets", "accelerate", "peft", "huggingface-hub")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_run_manifest(run_dir, task, summary, data_dir, model_commit):
    run_dir = Path(run_dir)
    prepared_manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    remote_source = task["source"].get("repository", task["source"]["dataset"])
    resolved = {
        "task": task,
        "training": {
            "method": summary["method"],
            "train_examples": summary["train_examples"],
            "validation_examples": summary["validation_examples"],
            "sampling": summary["train_sampling"],
            "max_steps": summary["max_steps"],
            "save_steps": summary.get("save_steps"),
            "resumed_from_checkpoint": summary.get("resumed_from_checkpoint"),
            "seed": summary["seed"],
        },
    }
    resolved_path = run_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task["name"],
        "resolved_config_sha256": file_sha256(resolved_path),
        "task_config_sha256": file_sha256(run_dir / "task_config.json"),
        "run_summary_sha256": file_sha256(run_dir / "run_summary.json"),
        "final_artifacts": snapshot_files(run_dir / "final"),
        "prepared_data_dir": str(Path(data_dir).resolve()),
        "prepared_artifacts": summary["prepared_artifacts"],
        "prepared_manifest_sha256": file_sha256(Path(data_dir) / "manifest.json"),
        "prepared_source": prepared_manifest.get("source"),
        "prepared_source_sha256": prepared_manifest.get("source_sha256"),
        "dataset_revision": (task["source"].get("revision") or task["source"].get("repository_revision"))
                            if prepared_manifest.get("source") == remote_source else None,
        "prompt_version": task.get("prompt_version", "v1"),
        "prompt_sha256": sha256_bytes(task["prompt_template"].encode("utf-8")),
        "starting_model": {
            "identifier": summary["starting_model"],
            "requested_revision": summary.get("starting_model_revision"),
            "resolved_revision": model_commit,
            "local_artifacts": snapshot_files(summary["starting_model"])
                               if Path(summary["starting_model"]).is_dir() else None,
        },
        "tokenizer": {"identifier": summary["starting_model"],
                      "resolved_revision": summary.get("tokenizer_revision") or model_commit},
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "seed": summary["seed"],
        "decoding_defaults": {"max_new_tokens": 80, "temperature": 0.8, "top_p": 0.9,
                              "repetition_penalty": 1.1, "seed": 42},
        "evaluation_decoding": {"max_new_tokens": task.get("evaluation", {}).get("max_new_tokens", 80),
                                "temperature": 0, "top_p": 1, "repetition_penalty": 1,
                                "seed": summary["seed"]},
        "library_versions": package_versions(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_run_manifest(run_dir):
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return None  # Historical runs predate schema 1.
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError(f"Unsupported run manifest schema: {manifest.get('schema_version')}")
    checks = (("resolved_config.json", "resolved_config_sha256"),
              ("task_config.json", "task_config_sha256"),
              ("run_summary.json", "run_summary_sha256"))
    for filename, key in checks:
        if file_sha256(Path(run_dir) / filename) != manifest[key]:
            raise ValueError(f"Saved run {filename} differs from its run manifest")
    resolved = json.loads((Path(run_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    if sha256_bytes(resolved["task"]["prompt_template"].encode("utf-8")) != manifest["prompt_sha256"]:
        raise ValueError("Saved prompt differs from its run manifest")
    if snapshot_files(Path(run_dir) / "final") != manifest["final_artifacts"]:
        raise ValueError("Saved model artifacts differ from the run manifest")
    local_artifacts = manifest["starting_model"].get("local_artifacts")
    if local_artifacts is not None and snapshot_files(manifest["starting_model"]["identifier"]) != local_artifacts:
        raise ValueError("Local starting model differs from the run manifest")
    return manifest


def verify_run_data(manifest, data_dir, artifacts):
    if manifest is None:
        return
    if artifacts != manifest["prepared_artifacts"]:
        raise ValueError("Prepared files differ from the run manifest")
    if file_sha256(Path(data_dir) / "manifest.json") != manifest["prepared_manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from the run manifest")
