# Reusable text fine-tuning toolkit

Prepare, fine-tune, evaluate, and run inference for text tasks with Hugging Face causal language models. Each task JSON file defines the dataset, prompt, starting model, training settings, and evaluation. The included tasks default to `HuggingFaceTB/SmolLM2-135M`; experiments also use SmolLM2-360M. The original horoscope scripts remain available as learning examples.

## Project layout

```text
.github/workflows/        Offline checks on pushes and pull requests
tasks/                   Dataset and experiment configuration (four examples)
src/                     Reusable command-line pipeline
  prepare_task.py         Prepare prompt/response train, validation, and test files
  train_task.py           Train a full model or LoRA adapter
  evaluate_task.py        Score validation or test examples
  infer_task.py           Generate from a saved run
  audit_task_data.py      Check dataset integrity and split overlap
  compare_label_evaluations.py  Compare classification runs on the same cases
  task_config.py, task_prompts.py, task_io.py, task_model.py  Focused shared helpers
  task_core.py            Compatibility imports for older scripts
  data_integrity.py, data_quality.py  Fingerprints and quality audit
  run_lineage.py, task_generation.py  Run manifest and shared generation
  training_resume.py      Checkpoint settings and resume checks
  promote_run.py          Metric gate and local registry record
  summarize_seed_runs.py  Compare repeated runs with different seeds
  horoscope_experiments/ Earlier step-by-step horoscope and QLoRA scripts
tests/                   Offline checks using small local fixtures
docs/                    Pipeline guide, safeguards, and experiment notes
hugging_face_fine_tuning_learnings.md  Step-by-step learning journal
configs/                 MLX QLoRA experiment settings
examples/                Small inference inputs
requirements*.txt        Main and optional MLX/CUDA dependencies
dvc.yaml, params.yaml     Optional prepare → audit → train → evaluate pipeline
LICENSE                  MIT license for project code
data/                    Downloaded and prepared datasets (local only)
models/                  Checkpoints and adapters (local only)
results/                 Generated evaluations (local only)
```

Use the four `*_task.py` commands in `src/` for new scenarios. The scripts in [src/horoscope_experiments](src/horoscope_experiments/README.md) preserve the earlier, horoscope-specific sequence; [this example](examples/horoscope_governed.md) runs horoscope through the governed pipeline. Run commands from the project root. See the [architecture diagram](docs/architecture.md), [reusable pipeline guide](docs/reusable_pipeline.md), [internal toolkit safeguards](docs/internal_toolkit.md), and [learning journal](hugging_face_fine_tuning_learnings.md).

## Set up

From the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The reusable PyTorch path was last run with Python 3.14.7, PyTorch 2.14.0, Transformers 5.17.0, Datasets 5.0.1, Accelerate 1.15.0, and PEFT 0.21.2. These direct dependency versions are pinned in `requirements.txt`; transitive dependencies are not fully locked. Apple Silicon can use MPS when PyTorch reports it available; CPU also works. MLX QLoRA and CUDA bitsandbytes have separate requirement files and are described in the [experiment guide](docs/horoscope_experiments.md).

## Internal toolkit checks

Run the offline checks before changing task preparation or training code:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/audit_task_data.py --data-dir data/tasks/emotion_classification
```

New preparations record file hashes, row counts, and exact-prompt overlap in
their manifest. New training runs write `run_manifest.json` and a resolved
configuration; evaluation rejects changed data or saved configuration before
loading model weights. The data audit also reports likely near duplicates,
label counts, and possible PII or secrets; training checks token lengths with
the resolved tokenizer. Potential secrets stop
training by default; PII and near-duplicate findings are reported for review
unless the task policy says to fail. CI audits a checked-in prepared-data fixture. Training
calls the same audit code and stops on cross-split prompt overlap unless
`--allow-overlap` is set for historical experiments. For a clean new data
directory, add `--deduplicate-cross-split --fail-on-overlap` to
`src/prepare_task.py`. Keep historical prepared data and model runs in their
existing directories; a clean re-preparation changes the comparison set.
See [internal toolkit safeguards](docs/internal_toolkit.md) for commands,
limitations, and the [run lineage ADR](docs/adr/001-run-lineage-and-pipeline-orchestration.md). Pull requests run the offline
checks in GitHub Actions.

## Run a first task

Emotion classification is a small-output example. Prepare it once, then run a two-step check:

```bash
.venv/bin/python src/prepare_task.py --task tasks/emotion_classification.json --output-dir data/tasks/emotion_classification_clean --deduplicate-cross-split --fail-on-overlap
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json --data-dir data/tasks/emotion_classification_clean --train-samples 32 --validation-samples 8 --max-steps 2 --output-dir models/tasks/emotion_classification/smoke-2-clean
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smoke-2-clean --test-samples 16 --generation-examples 8
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smoke-2-clean --test-samples 16 --generation-examples 8 --base-only
.venv/bin/python src/compare_label_evaluations.py models/tasks/emotion_classification/smoke-2-clean/base_test_evaluation.json models/tasks/emotion_classification/smoke-2-clean/test_evaluation.json
.venv/bin/python src/infer_task.py --run-dir models/tasks/emotion_classification/smoke-2-clean --input 'text=I am excited to see my friends.' --temperature 0
```

The two-step run checks that the workflow executes; it does not establish useful classification quality. To train the configured 512-example LoRA run, use a new output directory:

```bash
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json --data-dir data/tasks/emotion_classification_clean --output-dir models/tasks/emotion_classification/clean-lora-512
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/clean-lora-512
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/clean-lora-512 --base-only
.venv/bin/python src/promote_run.py --run-dir models/tasks/emotion_classification/clean-lora-512
```

Promotion checks the **example** emotion policy on at least 128 generated test cases: test accuracy at least 0.70,
macro-F1 at least 0.55, accuracy lift over the same base model at least 0.05,
and no invalid labels. A short smoke run is expected to fail this quality
gate. Passing writes a local registry record under `models/registry/`; it does
not publish or deploy a model. The other three tasks need their own promotion
policy before they can use this command.

Preparation and training refuse to overwrite nonempty output directories. If a dataset is already prepared, skip preparation. To repeat a run, pass a new `--output-dir`.

For optional stage orchestration, install DVC separately, run `dvc init`, review
the paths in `params.yaml`, then run `dvc repro`. The DVC pipeline writes a
separate clean dataset and evaluates both the adapter and its base model.
Manual label comparisons require a matching base-model evaluation for each
trained run; generate one with `evaluate_task.py --base-only` using the same
split and sampling flags.
See the [run lineage ADR](docs/adr/001-run-lineage-and-pipeline-orchestration.md)
for the manifest fields and current limits.

## Included tasks

| Task | Dataset | Input → output | Generation check |
| --- | --- | --- | --- |
| [Horoscope](tasks/horoscope.json) | [astro_horoscope](https://huggingface.co/datasets/karthiksagarn/astro_horoscope) | Sign, category, date → horoscope | Manual adherence review |
| [Conversation summary](tasks/conversation_summary.json) | [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) | Dialogue → summary | ROUGE-L word overlap and factual review |
| [Concept sentence](tasks/concept_sentence.json) | [CommonGen](https://huggingface.co/datasets/GEM/common_gen) | Concept list → sentence | Exact concept word coverage |
| [Emotion classification](tasks/emotion_classification.json) | [emotion](https://huggingface.co/datasets/dair-ai/emotion) | Text → one label | Exact-label accuracy |

For concept-list inference, pass [the example JSON file](examples/concept_sentence_input.json) with `--input-json`. For all commands and the task-file schema, see [the reusable pipeline guide](docs/reusable_pipeline.md).

## Experiment snapshot

The emotion-classification adapters were compared on the same 128 test examples:

| Model | Training examples | Exact accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| SmolLM2-135M + LoRA | 512 | 66.4% | 0.382 |
| SmolLM2-360M + LoRA | 512 | 72.7% | 0.539 |
| SmolLM2-360M + LoRA | 2,000 | 78.1% | 0.568 |

Equal-count training improved the 360M model on a balanced 150-example validation slice (60.7% to 70.7% accuracy), while the original shuffled training subset performed better on an ordinary 128-example validation slice (73.4% versus 70.3%). The [learning journal](hugging_face_fine_tuning_learnings.md) records the full comparisons and limits; the [validation error review](docs/emotion_validation_error_review.md) explains the rare-label tradeoff. Each other task still needs its own quality evaluation.

## Add another dataset

Copy a task file and change its `name`, `source`, `prompt_template`, `response_field`, and training settings. Each `{placeholder}` in the prompt must match a source column. `response_map` converts numeric labels to text; `field_transforms` handles simple string and list formatting. Prepare the data, inspect its splits, then train and evaluate. Each task writes to separate directories under `data/tasks/` and `models/tasks/`.

Datasets, checkpoints, virtual environments, and generated evaluations are intentionally excluded from Git. They are recreated locally.

## License

The project code is released under the [MIT License](LICENSE). Dataset and base-model licenses are separate; check their terms before redistributing data, weights, or derivative artifacts.
