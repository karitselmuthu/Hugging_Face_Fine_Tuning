# Reusable fine-tuning pipeline

The project has two parts:

- `src/prepare_dataset.py`, `src/train_response_only.py`, and the other
  horoscope scripts preserve the completed learning experiments.
- `tasks/*.json` plus `src/{prepare,train,evaluate,infer}_task.py` provide a
  reusable path for a new text input → text output task.

The shared prepared format is one JSON object per line. `inputs` retains the
original prompt fields for task-specific checks:

```json
{"prompt":"### Instruction:\nSummarize the conversation...\n### Summary:\n","response":"A and B agreed to meet on Friday.","inputs":{"dialogue":"A: ..."}}
```

Training tokenizes these two fields separately. Prompt token labels are `-100`,
so the loss is applied only to the response and EOS token. The train,
validation, and test splits stay separate. The trainer **skips** examples
whose full prompt and response exceed the configured context length; it does
not silently cut off a summary target. Check the skipped counts in the run
summary and increase the context length or change the data policy if too many
examples are excluded.

## Included scenarios

| Task | Source columns | Split policy | Generation check |
| --- | --- | --- | --- |
| Horoscope | `sign`, `category`, `date` → `horoscope` | Split source train data | Inspect sign, category, and date use |
| Conversation summary | `dialogue` → `summary` | Source train, validation, test | ROUGE-L word-overlap F1 plus factual review |
| Concept sentence | `concepts` list → `target` | Keep identical concept sets together; public validation is test | Fraction of concepts appearing as exact words |
| Emotion classification | `text` → numeric `label` | Source train, validation, test | Exact generated-label accuracy and macro-F1 |

The concept check counts exact word forms, so `ski` and `skis` differ. ROUGE-L
does not detect invented facts. Generation checks use greedy decoding on the
configured sample count. Response-only loss and perplexity are also reported
for each task. The public CommonGen validation split supplies labelled test
examples because its official test targets are unavailable to this workflow.

## Balanced emotion evaluation

The standard evaluation shuffles the test set and scores the first 128 fitting
examples. To inspect rare labels, `--balanced-per-label 25` deterministically
selects 25 examples for each of the six emotion labels (150 total) and
generates an answer for every selected example. It writes a separate file,
preserving the earlier random-sample evaluation. Run this from the project
root after training both adapters:

```bash
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/lora-512 --balanced-per-label 25 --device mps
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smollm2-360m-lora-512 --balanced-per-label 25 --device mps
.venv/bin/python src/compare_label_evaluations.py models/tasks/emotion_classification/lora-512/balanced_25_test_evaluation.json models/tasks/emotion_classification/smollm2-360m-lora-512/balanced_25_test_evaluation.json
```

The comparison command verifies that both files contain identical prompts
and references, then prints strict accuracy, macro-F1, invalid outputs, and
per-label recall and F1. This sample deliberately changes the class mix, so
its overall accuracy is not an estimate of accuracy on the original test-set
distribution. Use the ordinary evaluation or the full test set for that.
The prepared emotion test split has 66 `surprise` examples, so 25 per label
fits. These commands load each model and generate 150 answers, which takes
longer than the original 128-example check; they do not retrain either model.

If rare-label recall remains low, change only the training subset size first.
The following run uses the same 360M base, task settings, seed, validation
sample count, and prepared data as the 512-example run. Use the validation
split for this next model-selection decision; the test slice above has already
been inspected repeatedly:

```bash
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json --starting-model HuggingFaceTB/SmolLM2-360M --train-samples 2000 --validation-samples 64 --output-dir models/tasks/emotion_classification/smollm2-360m-lora-2000
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smollm2-360m-lora-512 --split validation --balanced-per-label 25 --device mps
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smollm2-360m-lora-2000 --split validation --balanced-per-label 25 --device mps
.venv/bin/python src/compare_label_evaluations.py models/tasks/emotion_classification/smollm2-360m-lora-512/balanced_25_validation_evaluation.json models/tasks/emotion_classification/smollm2-360m-lora-2000/balanced_25_validation_evaluation.json
```

The 2,000-example run took 441.7 seconds on MPS. On the same 150 balanced
validation cases, its accuracy was 60.7% and macro-F1 was 0.598, compared
with 44.7% and 0.410 for the 512-example adapter. `love` recall rose from
3/25 to 11/25 and `surprise` from 3/25 to 10/25. Both models produced valid
labels for every case. Eight cases overlap the 64 validation examples
monitored during training; they were never used for weight updates. This
balanced sample has equal class counts, so its accuracy answers a different
question from accuracy under the original class distribution.

The 2,000-example setup is the selected candidate for a final held-out test
check. The 512-example balanced test file already exists; run the new adapter
on those same 150 test cases and compare them:

```bash
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smollm2-360m-lora-2000 --balanced-per-label 25 --device mps
.venv/bin/python src/compare_label_evaluations.py models/tasks/emotion_classification/smollm2-360m-lora-512/balanced_25_test_evaluation.json models/tasks/emotion_classification/smollm2-360m-lora-2000/balanced_25_test_evaluation.json
```

For a fixed sample with the original test-set class mix, generate all 128
answers for the new adapter. Compare with the saved 512-example test file only
if it also contains the same 128 generated cases:

```bash
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/smollm2-360m-lora-2000 --test-samples 128 --generation-examples 128 --device mps
.venv/bin/python src/compare_label_evaluations.py models/tasks/emotion_classification/smollm2-360m-lora-512/test_evaluation.json models/tasks/emotion_classification/smollm2-360m-lora-2000/test_evaluation.json
```

Use these test results to report performance, not to choose another training
configuration on the same test examples.

## Prepare and run

From the project root, with `requirements.txt` installed in `.venv`:

```bash
.venv/bin/python src/prepare_task.py --task tasks/concept_sentence.json
.venv/bin/python src/prepare_task.py --task tasks/emotion_classification.json
.venv/bin/python src/prepare_task.py --task tasks/horoscope.json
```

Preparation rejects a nonempty output directory. To rebuild a dataset, use
another `--output-dir` and supply it to training with `--data-dir`.

Use one task at a time for training. For example:

```bash
.venv/bin/python src/train_task.py --task tasks/emotion_classification.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/emotion_classification/lora-512
.venv/bin/python src/infer_task.py --run-dir models/tasks/emotion_classification/lora-512 --input 'text=I am excited to see my friends.' --temperature 0
```

For a concept list, use a JSON input file:

```bash
.venv/bin/python src/train_task.py --task tasks/concept_sentence.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/concept_sentence/lora-512
.venv/bin/python src/infer_task.py --run-dir models/tasks/concept_sentence/lora-512 --input-json examples/concept_sentence_input.json --temperature 0
```

The generic horoscope task uses the same commands:

```bash
.venv/bin/python src/train_task.py --task tasks/horoscope.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/horoscope/lora-512
.venv/bin/python src/infer_task.py --run-dir models/tasks/horoscope/lora-512 --input sign=Aries --input category=Career --input date=2026/10/06
```

These full training commands are optional learning experiments. A short
pipeline check uses `--train-samples 32 --validation-samples 8 --max-steps 2
--output-dir models/tasks/<task>/smoke-2` on `train_task.py`; evaluate with a
small `--test-samples` and `--generation-examples`. A two-step adapter is not a
quality result.

## Conversation summarization example

From the project root, with `requirements.txt` installed in `.venv`:

```bash
.venv/bin/python src/prepare_task.py --task tasks/conversation_summary.json
.venv/bin/python src/train_task.py --task tasks/conversation_summary.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/conversation_summary/lora-512
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/conversation_summary/lora-512 --base-only
.venv/bin/python src/infer_task.py --run-dir models/tasks/conversation_summary/lora-512 --input $'dialogue=A: Can we meet on Friday?\nB: Yes. Please bring the notes.\nA: I will.'
```

The task uses the [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum)
(`dialogue` → `summary`) and starts from `HuggingFaceTB/SmolLM2-135M`.
The dataset card describes a noncommercial research license; review it before
using the data outside a learning experiment. Prepared files go to
`data/tasks/conversation_summary/`. The trained adapter, tokenizer, run
summary, and task snapshot go to
`models/tasks/conversation_summary/lora-512/`. The test command writes
`test_evaluation.json` there; `--base-only` writes
`base_test_evaluation.json` for a same-slice baseline. The saved adapter is
loaded with the same base model for inference. The earlier learning experiment
is summarized in [the journal](../hugging_face_fine_tuning_learnings.md).
Both scripts reject a nonempty output directory to protect saved
experiments. To repeat training, pass a new `--output-dir`; to rebuild data,
also pass a new preparation `--output-dir` and point training at it with
`--data-dir`.

For a short pipeline check, use distinct output directories:

```bash
.venv/bin/python src/train_task.py --task tasks/conversation_summary.json --train-samples 32 --validation-samples 8 --max-steps 2 --output-dir models/tasks/conversation_summary/smoke-2
```

Two training steps check that the pipeline runs; they do not measure useful
summarization quality. The configured 512-example run is a first learning
experiment. Review generated summaries against the held-out references and
measure factual coverage before treating the adapter as useful.

## Add another dataset or task

1. Copy `tasks/conversation_summary.json` to a new file under `tasks/`.
2. Give it a unique `name`, a Hugging Face `source.dataset`, and the source
   split names. If the source has only one split, set validation and test to
   `null`; preparation creates disjoint splits with `heldout_fraction`.
3. Write `prompt_template` with `{column_name}` placeholders and set
   `response_field` to the target column. The placeholders must match source
   columns. Optional `field_transforms` support `strip`, `title`, `lower`, and
   `join_comma` for list inputs. `response_map` converts numeric labels to
   answer text. `source.data_files` supports Parquet files when a dataset
   repository script is incompatible with the installed `datasets` version.
   `source.group_field` keeps repeated inputs out of different splits.
4. Set `max_length` and training options. `method` can be `lora` or `full`.
   `lora_targets` are model architecture dependent; the example targets fit
   SmolLM2's attention layers.
5. Run prepare, train, evaluate, and infer with the new task file. Each task
   gets its own data and model directory, so previous experiments remain
   available.

Use `--starting-model` when you intentionally want to start from a different
base or a saved **full** model checkpoint. The default starts each task from
SmolLM2-135M and saves a separate adapter. Continuing from a horoscope LoRA
adapter is a different multi-task experiment and is not implemented by this
flag.

The generic trainer currently supports PyTorch full fine-tuning and PEFT
LoRA. The project's MLX QLoRA and CUDA bitsandbytes scripts remain separate
learning paths; they are not yet connected to these task files. The generic
evaluator reports response loss, sample text, and the configured task metric
where one exists. Generated text still needs human review where factual
accuracy matters.
