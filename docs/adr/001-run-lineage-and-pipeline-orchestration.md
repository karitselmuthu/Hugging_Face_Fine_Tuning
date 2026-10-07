# ADR-001: Run lineage and pipeline orchestration

Status: implemented for new runs; optional DVC graph awaits a local end-to-end run.

## Context

Preparation, training, evaluation, and inference were separate commands. The
prepared-data manifest and training summary did not bind the exact model,
prompt, source, and decoding settings into one run record.

## Decision

Each new training run writes `resolved_config.json` and a schema 2
`run_manifest.json`. Evaluation checks saved files and prepared-data
fingerprints before loading model weights; inference checks the saved run.
Both commands use `task_generation.generate_text()`. The task JSON copied
into the run is their configuration source.

The manifest records the task and resolved-config hashes, prompt hash and
version, prepared file hashes and counts, preparation manifest hash, saved
model hashes, remote source revision or local source SHA, base model and
tokenizer commit, Git state, seed, decoding defaults, and library versions.
Local starting models are bound by file hashes. Schema 1 and historical runs
remain readable, but they do not gain retroactive guarantees.

Preparation and training use the shared audit. Exact cross-split prompt
overlap fails training unless `--allow-overlap` is used for historical data.
Likely near duplicates, label distribution, possible sensitive data, and
overlength rows are reported. Secret patterns fail by default; other quality
thresholds are task policies. CI audits a checked-in clean fixture.

`dvc.yaml` defines prepare → audit → train → trained evaluation and base
evaluation → comparison. `params.yaml` selects the emotion example and
separate output paths. DVC is optional and is not part of `requirements.txt`.
The label comparison requires a base result. The emotion task also has an
example threshold gate in `promote_run.py` that writes a local registry record
only when the held-out trained result passes.

## Options considered

| Option | Strength | Limit |
| --- | --- | --- |
| DVC + JSON manifest | Local dependency graph and artifact cache | Setup and limited comparison UI |
| MLflow | Team tracking and registry | Does not orchestrate or version data alone |
| Weights & Biases | Experiment UI and Trainer integration | SaaS dependency and data residency review |

The internal single-machine workflow uses local files without a tracking
server. Team use may later need a shared artifact store and approval process.

## Operating path

```bash
.venv/bin/python -m pip install dvc
dvc init
dvc repro
```

Review `params.yaml` and choose unused output paths first. Preparation uses
exact-prompt deduplication and strict overlap checks; the audit report gates
training. Trained and base evaluations use the same held-out rows and
decoding. The comparison stage is configured for emotion labels. The full
DVC graph has not been run in this checkout; CI uses offline fixtures because
the full datasets and weights are local.

## Consequences and limits

Schema 2 detects accidental changes in saved config, data, and model files.
It is not a signed attestation. The near-duplicate and sensitive-data scans
are heuristics. The example promotion policy is not organizational approval.
Bootstrap intervals cover held-out case sampling uncertainty, while training
variation requires separately trained seeds. A local `dvc repro` and repeated
training runs remain validation work.
