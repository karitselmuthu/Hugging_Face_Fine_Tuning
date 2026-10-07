# Internal training toolkit safeguards

This project is currently an internal, command-line training toolkit. Its
reusable path is `src/prepare_task.py` → `src/train_task.py` →
`src/evaluate_task.py` → `src/infer_task.py`. The horoscope learning scripts
are separate in `src/horoscope_experiments/`.

## Prepare and audit data

Task JSON files are checked before a dataset download or model load. A
prepared directory is written through a temporary directory; failure does
not leave a partial final directory. Its `manifest.json` records row counts,
SHA-256 hashes of the three split files and task config, and counts of exact
prompts shared across splits.

To inspect an existing prepared dataset without loading a model:

```bash
.venv/bin/python src/audit_task_data.py --data-dir data/tasks/emotion_classification
```

Add `--fail-on-overlap` when an audit should return a failing exit status.
The current historical emotion data contains 5 distinct prompts shared
between train and validation, 11 between train and test, and 3 between
validation and test. The audit additionally reports likely near duplicates
using input similarity. This heuristic does not detect every paraphrase or
semantic duplicate. It also reports label counts and counts of patterns
resembling email addresses, phone numbers, or common secrets, without
printing matched text. Detected secrets block preparation and training by
default. PII and near duplicates are report-only unless the task sets
`data_quality.fail_on_pii` or `data_quality.fail_on_near_duplicate`, or an
individual audit uses `--fail-on-pii` or `--fail-on-near-duplicate`.

For a new, clean preparation, use a **new** output directory:

```bash
.venv/bin/python src/prepare_task.py --task tasks/emotion_classification.json --output-dir data/tasks/emotion_classification_clean --deduplicate-cross-split --fail-on-overlap
```

Deduplication preserves test rows first, then validation rows, and removes
matching prompts from lower-priority splits. Recheck the row counts and
label distribution before training. Do not compare scores from this clean
dataset directly with the project's historical emotion results: the
evaluation rows may differ. To pin a Hugging Face dataset version, set
`source.revision` in the task JSON to a commit identifier before preparing.
The included task files pin their Hub revisions; a local JSONL source is
bound by its SHA-256 in the preparation manifest.

## Train and evaluate the same data

New training runs record hashes of the exact prepared files they used.
Evaluation checks those hashes and the task's preparation settings before
loading model weights. Training calls `audit_task_data.audit_prepared_data`
before loading a model and rejects exact cross-split prompt overlap;
use `--allow-overlap` only to repeat historical experiments against the old
prepared data. New runs also write `run_manifest.json` and
`resolved_config.json` with model and tokenizer revision, saved artifact hashes, decoding
defaults, package versions, and Git state. Inference verifies the saved
run files before model loading. If the data changed, restore the original prepared
directory or train a new run against the new data. Existing historical runs
remain readable, but they lack this retroactive fingerprint guarantee.

New runs count token lengths for every prepared row before loading model
weights. Set `data_quality.max_overlength_fraction` to stop training when too
many rows exceed `max_length`; otherwise the count is diagnostic and the
response-only tokenizer skips those rows. Trainer checkpoints are saved every
100 steps by default, or at `training.save_steps`. Resume with
`--resume-from-checkpoint latest` using the same output directory and
settings; changed task, data, model, limits, or seed are rejected. The
governed path loads model weights from safetensors with remote code disabled.

For a run on a separately prepared directory, pass the same directory to
both commands:

```bash
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json --data-dir data/tasks/emotion_classification_clean --output-dir models/tasks/emotion_classification/clean-lora-512
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/clean-lora-512 --data-dir data/tasks/emotion_classification_clean
```

These hashes catch accidental file changes; they are not signatures or an
access-control system. Dataset licenses, private data handling, shared
artifact storage, and model approval remain organization-specific decisions.
See [ADR-001](adr/001-run-lineage-and-pipeline-orchestration.md) for the
optional DVC graph and remaining lineage limits.

For private Hub resources, pass `HF_TOKEN` through the shell environment.
Keep it out of task JSON, command arguments, and Git. See the
[Hugging Face token documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

## Checks before a change

The offline suite uses tiny local JSONL fixtures and downloads no model or
dataset. Run it locally:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on pushes and pull requests. It also runs
the audit command against a checked-in, clean prepared-data fixture; real
downloaded datasets remain local and are audited before training. Direct
dependency versions are pinned in `requirements.txt`; a platform-specific
lock of all transitive dependencies and a GPU training gate remain future
work. The offline checks establish code and data-handling behavior, not
model quality or MPS/CUDA performance.
