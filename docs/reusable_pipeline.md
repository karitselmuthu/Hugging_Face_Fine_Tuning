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
| Emotion classification | `text` → numeric `label` | Source train, validation, test | Exact generated-label accuracy |

The concept check counts exact word forms, so `ski` and `skis` differ. ROUGE-L
does not detect invented facts. Generation checks use greedy decoding on the
configured sample count. Response-only loss and perplexity are also reported
for each task. The public CommonGen validation split supplies labelled test
examples because its official test targets are unavailable to this workflow.

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
