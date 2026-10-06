# Reusable SmolLM2 fine-tuning pipeline

Fine-tune `HuggingFaceTB/SmolLM2-135M` on a text task by changing a JSON task file. The shared workflow prepares data, trains a response-only model or LoRA adapter, evaluates it on a separate test split, and runs inference. The original horoscope experiments remain available as learning examples.

## Project layout

```text
configs/                 MLX QLoRA experiment settings
examples/                Small inference inputs
tasks/                   Dataset, prompt, training, and evaluation settings
src/                     Reusable task pipeline
  prepare_task.py         Build prompt/response train, validation, and test files
  train_task.py           Train a full model or LoRA adapter
  evaluate_task.py        Score validation or test examples and save generations
  infer_task.py           Generate from a saved run
  compare_label_evaluations.py  Compare classification runs on matching cases
  task_core.py            Shared task and model helpers
  task_metrics.py         Task-specific generation checks
  horoscope_experiments/ Earlier step-by-step horoscope and QLoRA scripts
docs/                    Workflow and experiment notes
data/                    Downloaded and prepared datasets (local only)
models/                  Checkpoints and adapters (local only)
results/                 Generated evaluations (local only)
```

The scripts in [src/horoscope_experiments](src/horoscope_experiments/README.md) preserve the step-by-step horoscope and QLoRA experiments. Run commands from the project root; see [the experiment guide](docs/horoscope_experiments.md) and [learning journal](hugging_face_fine_tuning_learnings.md).

## Set up

From the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The reusable PyTorch path was last run with Python 3.14.7, PyTorch 2.14.0, Transformers 5.17.0, Datasets 5.0.1, Accelerate 1.15.0, and PEFT 0.21.2. Apple Silicon can use MPS when PyTorch reports it available; CPU also works. MLX QLoRA and CUDA bitsandbytes have separate requirement files and are described in the [experiment guide](docs/horoscope_experiments.md).

## Run a first task

Emotion classification is a small-output example. Prepare it once, then run a two-step check:

```bash
.venv/bin/python src/prepare_task.py --task tasks/emotion_classification.json
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json --train-samples 32 --validation-samples 8 --max-steps 2 --output-dir models/tasks/emotion_classification/smoke-2
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smoke-2 --test-samples 16 --generation-examples 8
.venv/bin/python src/infer_task.py --run-dir models/tasks/emotion_classification/smoke-2 --input 'text=I am excited to see my friends.' --temperature 0
```

The two-step run checks that the workflow executes; it does not establish useful classification quality. To train the configured 512-example LoRA run, use a new output directory:

```bash
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/lora-512
```

Preparation and training refuse to overwrite nonempty output directories. If a dataset is already prepared, skip preparation. To repeat a run, pass a new `--output-dir`.

## Included tasks

| Task | Dataset | Input → output | Generation check |
| --- | --- | --- | --- |
| [Horoscope](tasks/horoscope.json) | [astro_horoscope](https://huggingface.co/datasets/karthiksagarn/astro_horoscope) | Sign, category, date → horoscope | Manual adherence review |
| [Conversation summary](tasks/conversation_summary.json) | [SAMSum](https://huggingface.co/datasets/knkarthick/samsum) | Dialogue → summary | ROUGE-L word overlap and factual review |
| [Concept sentence](tasks/concept_sentence.json) | [CommonGen](https://huggingface.co/datasets/GEM/common_gen) | Concept list → sentence | Exact concept word coverage |
| [Emotion classification](tasks/emotion_classification.json) | [emotion](https://huggingface.co/datasets/dair-ai/emotion) | Text → one label | Exact-label accuracy |

For concept-list inference, pass [the example JSON file](examples/concept_sentence_input.json) with `--input-json`. For all commands and the task-file schema, see [the reusable pipeline guide](docs/reusable_pipeline.md).

The first full emotion-classification experiment used 512 training examples. On the same 128 held-out examples, SmolLM2-135M got 85 correct (66.4% accuracy; 0.382 macro-F1) and SmolLM2-360M got 93 correct (72.7%; 0.539 macro-F1). The 360M run still made one invalid prediction and missed every `surprise` case. Both untuned base models scored 0% under the strict one-label output rule because their generations contained extra text. A second, balanced 150-example test comparison gave the trained 135M model 40.0% accuracy and 0.306 macro-F1, versus 52.0% and 0.469 for 360M; the latter recognized only 2 of 25 `surprise` cases. Increasing the 360M adapter's training subset from 512 to 2,000 examples raised balanced validation accuracy from 44.7% to 60.7% and macro-F1 from 0.410 to 0.598 on the same 150 cases. On the matching balanced test slice, accuracy rose from 52.0% to 64.0% and macro-F1 from 0.469 to 0.624. On the matching ordinary 128-case test sample, accuracy rose from 72.7% to 78.1% and macro-F1 from 0.539 to 0.568; both adapters missed all five `surprise` cases in that sample. Results and limits are recorded in the [learning journal](hugging_face_fine_tuning_learnings.md). The [balanced evaluation guide](docs/reusable_pipeline.md#balanced-emotion-evaluation) shows how to check rare labels on matching cases. The [validation error review](docs/emotion_validation_error_review.md) motivated an equal-count 2,000-example run through `train_task.py --balanced-train`; on the same balanced validation slice, accuracy rose from 60.7% to 70.7% and macro-F1 from 0.598 to 0.707. On the ordinary 128-example validation sample, the original adapter led 73.4% to 70.3% in accuracy and 0.670 to 0.612 in macro-F1, showing a tradeoff between common-label accuracy and rare-label recall.

## Add another dataset

Copy a task file and change its `name`, `source`, `prompt_template`, `response_field`, and training settings. Each `{placeholder}` in the prompt must match a source column. `response_map` converts numeric labels to text; `field_transforms` handles simple string and list formatting. Prepare the data, inspect its splits, then train and evaluate. Each task writes to separate directories under `data/tasks/` and `models/tasks/`.

Datasets, checkpoints, virtual environments, and generated evaluations are intentionally excluded from Git. They are recreated locally.

## License

The project code is released under the [MIT License](LICENSE). Dataset and base-model licenses are separate; check their terms before redistributing data, weights, or derivative artifacts.
