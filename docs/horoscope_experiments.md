# Horoscope experiment commands

These historical scripts live in
[`src/horoscope_experiments/`](../src/horoscope_experiments/README.md) and show
the project progression before the reusable task pipeline. Run commands from
the repository root. They require locally prepared data or saved models;
neither is included in Git. Generated evaluation files under `results/` are
local too. For the public starting path, see the [README](../README.md) and
[reusable pipeline guide](reusable_pipeline.md). For a new horoscope run with
the shared audit and run manifest, use the
[governed horoscope example](../examples/horoscope_governed.md).

## Run the saved model

From the project root:

```bash
.venv/bin/python src/horoscope_experiments/inference_response_only.py
```

This script uses the 2,000-example response-only model and a prompt configured
in the script.

## Compare input sensitivity

```bash
.venv/bin/python src/horoscope_experiments/evaluate_input_sensitivity.py
```

The comparison runs four prompts against each of the 512-example and
2,000-example response-only models. It holds all inputs fixed except one of
sign, category, or date, and resets the generation seed for each prompt. It
writes full outputs to `results/input_sensitivity.md` and machine-readable
results to `results/input_sensitivity.json`.

To change the output length or run without a GPU:

```bash
.venv/bin/python src/horoscope_experiments/evaluate_input_sensitivity.py --max-new-tokens 100
.venv/bin/python src/horoscope_experiments/evaluate_input_sensitivity.py --device cpu
```

These four prompts are a first diagnostic, not a quality score. Review whether
the output follows the requested sign, category, and exact date before drawing
conclusions or starting another training run.

## Evaluate on held-out examples

```bash
.venv/bin/python src/horoscope_experiments/evaluate_heldout.py
```

This scores both response-only models on the same 128 examples from the saved
test split. It also generates 12 outputs, one for each sign, with categories
spread across the examples. The results are saved to
`results/heldout_evaluation.md` and `results/heldout_evaluation.json`.

The response-only loss measures prediction of reference horoscope tokens.
Read the generated examples separately to judge whether the model follows the
requested sign, category, and date. The first run is summarized in
`results/heldout_review.md`.

## Audit data and sequence length

```bash
.venv/bin/python src/horoscope_experiments/analyze_training_data.py
```

The audit writes `results/training_data_audit.md` and its JSON companion.
The current 256-token limit retains every prepared example in the train and
test splits. The audit also counts literal mentions of sign, category, and
date in target text; those counts are diagnostic and do not measure semantic
instruction following.

## LoRA experiment

Install the project dependencies, including PEFT:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Train an adapter on the same 512-example subset used by the earlier full
fine-tuning experiment:

```bash
.venv/bin/python src/horoscope_experiments/train_lora.py --method lora --train-samples 512 --eval-samples 64
```

The adapter is saved in `models/smollm-horoscope-lora-512/final`. It contains
only the adapter weights; loading it for inference also loads the original
`HuggingFaceTB/SmolLM2-135M` base model.

```bash
.venv/bin/python src/horoscope_experiments/inference_lora.py --sign Taurus --category Career --date 2026/10/04
.venv/bin/python src/horoscope_experiments/evaluate_heldout.py --include-lora
```

The comparison is in `results/heldout_lora_comparison.md`. On the same 128
held-out examples, response-only NLL was 3.105 for LoRA trained on 512 examples,
3.274 for the 512-example full fine-tune, and 3.203 for the 2,000-example full
fine-tune. The LoRA run used a different learning rate, so these results do not
isolate the adapter method as the cause of the difference. Generated text still
needs review for sign, category, and date accuracy. See
`results/lora_review.md` for the first qualitative review.

## QLoRA on Apple Silicon with MLX

The Mac-native path uses MLX-LM, a separate Python 3.13 environment, and a
4-bit MLX conversion of the original SmolLM2-135M model. It is separate from
the PyTorch/PEFT adapter format. From the project root:

```bash
python3.13 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx-qlora.txt
.venv-mlx/bin/python -m mlx_lm convert --hf-path HuggingFaceTB/SmolLM2-135M --mlx-path models/smollm2-135m-mlx-4bit --quantize --q-bits 4 --q-group-size 64
.venv-mlx/bin/python src/horoscope_experiments/prepare_mlx_qlora.py --train-samples 32 --valid-samples 8
.venv-mlx/bin/python -m mlx_lm lora --config configs/mlx_qlora_smoke.yaml
```

The data preparation step exports prompt/completion JSONL files and sets a
raw-text tokenizer template in the converted model. `mask_prompt: true` makes
MLX-LM train on the horoscope response rather than the prompt. The YAML file
sets batch size 1, two training iterations, 16 adapted layers, and a 256-token
limit. The test completed on this Mac and saved a 5 MB adapter in
`models/smollm-horoscope-mlx-qlora-smoke-config`. Its output after two steps
is not a quality result; see `results/mlx_qlora_smoke.md`.

A 512-example follow-up completed on this Mac. To reproduce its training,
held-out scoring, and 12-prompt review:

```bash
.venv-mlx/bin/python src/horoscope_experiments/prepare_mlx_qlora.py --train-samples 512 --valid-samples 64 --test-samples 128 --output-dir data/mlx-qlora-512
.venv-mlx/bin/python -m mlx_lm lora --config configs/mlx_qlora_512.yaml
.venv-mlx/bin/python -m mlx_lm lora --model models/smollm2-135m-mlx-4bit --data data/mlx-qlora-512 --adapter-path models/smollm-horoscope-mlx-qlora-512 --test --mask-prompt --batch-size 1 --max-seq-length 256 --test-batches 128
.venv-mlx/bin/python src/horoscope_experiments/review_mlx_qlora.py
```

The adapter's held-out loss was 2.998 versus 3.902 for the 4-bit base.
Generated text still has category and date errors. See
`results/heldout_mlx_qlora_512.md` for the comparison and its limitations.

For a test slice that was not used for validation, run the commands in
`results/independent_adherence.md`. That report compares response loss on the
same 128 examples and audits explicit sign, category, and date cues in 12
matching generations. The cue counts are diagnostics, not semantic adherence
scores.

To generate from the saved MLX adapter, pass the full prompt with
`--ignore-chat-template`:

```bash
.venv-mlx/bin/python -m mlx_lm generate --model models/smollm2-135m-mlx-4bit --adapter-path models/smollm-horoscope-mlx-qlora-smoke-config --prompt $'### Instruction:\nGenerate a horoscope using the following information.\n\n### Sign:\nAries\n\n### Category:\nCareer\n\n### Date:\n2026/10/04\n\n### Horoscope:\n' --ignore-chat-template --max-tokens 40 --seed 42
```

## QLoRA on a CUDA machine with bitsandbytes

The separate PyTorch QLoRA option in `src/horoscope_experiments/train_lora.py` uses bitsandbytes NF4
loading and requires a supported CUDA GPU. On that machine, run:

```bash
python -m pip install -r requirements-qlora.txt
python src/horoscope_experiments/train_lora.py --method qlora --train-samples 512 --eval-samples 64
```

The CUDA bitsandbytes path has not been executed in this project. The Mac MLX
path has completed a two-step smoke test and a 512-step training run.
