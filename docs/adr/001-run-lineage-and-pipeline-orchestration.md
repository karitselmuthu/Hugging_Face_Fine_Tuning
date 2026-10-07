# ADR-001: Run lineage and pipeline orchestration

Status: implemented for new runs; optional DVC path awaits a local end-to-end run.

## Context

Preparation, training, evaluation, and inference have been separate commands.
The prepared-data manifest and training summary existed, but model revision,
resolved settings, decoding defaults, and package versions were not bound in
one run record.

## Decision

Each new training run writes `resolved_config.json` and a versioned
`run_manifest.json`. Evaluation checks the run files and prepared-data
fingerprints before model loading; inference checks the saved run files.
Both commands use `task_generation.generate_text()`. Training rejects exact
cross-split prompt overlap unless explicitly allowed for historical data.

`dvc.yaml` offers an optional prepare → audit → train → evaluate graph, with
a second evaluation of the base model and a paired comparison. `params.yaml` selects the task and
separate output paths. DVC is optional and is not part of `requirements.txt`.

## Manifest schema 1

The run manifest records the task name, SHA-256 of the saved task and resolved
config, prompt SHA-256, hashes and row counts of prepared files, hash of the
preparation manifest, saved summary and model artifact hashes, dataset revision when configured, starting model ID,
requested and resolved model revision when available, Git commit and dirty state, seed,
inference and evaluation decoding defaults, and installed library versions.
`resolved_config.json` records the validated task and effective method, sample
counts, sampling mode, step limit, and seed. Evaluations record their own
decoding settings and whether schema 1 was available. Older runs remain usable
without retroactive lineage claims.

For a new run, set `source.revision` and `base_model_revision` to immutable
Hugging Face commit IDs in the task JSON. Without them, a later prepare may
fetch changed data, or the loader may not report a resolved model commit.
Local starting-model directories also need external artifact versioning.
Git commit alone does not describe uncommitted source changes.

## Options considered

| Option | Strength | Limit |
| --- | --- | --- |
| DVC + JSON manifest | Local dependency graph and artifact cache; fits files | DVC setup and limited comparison UI |
| MLflow | Tracking and registry for teams | Does not orchestrate or version data by itself |
| Weights & Biases | Fast experiment UI and Trainer integration | SaaS dependency and data residency review |

DVC and MLflow can be combined later. The current internal, single-machine
workflow does not need a tracking server.

## Operating path

```bash
.venv/bin/python -m pip install dvc
dvc init
dvc repro
```

Review `params.yaml` first. Choose unused output paths; the commands refuse
to overwrite nonempty directories. `prepare` uses exact-prompt deduplication
and strict overlap checks; `audit` writes a report only after passing. DVC
then gates training on that report. `evaluate` and `base_evaluate` use the
same held-out rows and generation settings; `compare` is configured for the
emotion label task in `params.yaml`. DVC itself has not been installed
or run in this checkout, so the pipeline definition is not yet validated
end to end. CI runs only offline fixtures because the full datasets and model
weights are intentionally local.

## Consequences and remaining work

Schema 1 catches changes to saved config and prepared files, and it records
the starting model revision when the loader exposes it. The manifest is not
a signed attestation, and outputs still require an artifact store for shared
team use. The DVC graph needs a local `dvc repro` check and an immutable
dataset/model revision before it should be treated as a reproducible run.
Before adding MLflow or a registry, collect several run manifests and decide
which metrics and promotion rules the team actually needs.
