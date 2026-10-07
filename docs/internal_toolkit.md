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
validation and test. These are exact prompt matches; the audit does not
detect paraphrases or other semantic duplicates.

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

## Train and evaluate the same data

New training runs record hashes of the exact prepared files they used.
Evaluation checks those hashes and the task's preparation settings before
loading model weights. Training now rejects exact cross-split prompt overlap;
use `--allow-overlap` only to repeat historical experiments against the old
prepared data. New runs also write `run_manifest.json` and
`resolved_config.json` with model revision, saved artifact hashes, decoding
defaults, package versions, and Git state. Inference verifies the saved
run files before model loading. If the data changed, restore the original prepared
directory or train a new run against the new data. Existing historical runs
remain readable, but they lack this retroactive fingerprint guarantee.

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
optional DVC graph and the limits of unpinned upstream revisions.

## Checks before a change

The offline suite uses tiny local JSONL fixtures and downloads no model or
dataset. Run it locally:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on pushes and pull requests. Direct
dependency versions are pinned in `requirements.txt`; a platform-specific
lock of all transitive dependencies and a GPU training gate remain future
work. The offline checks establish code and data-handling behavior, not
model quality or MPS/CUDA performance.
