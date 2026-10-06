# Hugging Face Fine-Tuning Project --- What We Have Learned So Far

## Project Goal

This project started as a hands-on Hugging Face fine-tuning workflow using a
horoscope dataset. It now has a reusable task-driven path, first applied to
conversation summarization with SAMSum.

The focus so far has been on learning the application and training
pipeline step by step:

-   Loading datasets from Hugging Face
-   Preparing and transforming data
-   Tokenizing text
-   Loading pretrained causal language models
-   Fine-tuning with `Trainer`
-   Running training on Apple Silicon using MPS
-   Saving and loading local models
-   Performing inference
-   Moving from plain language-model training to instruction-style
    fine-tuning
-   Implementing response-only supervised fine-tuning
-   Comparing the effect of increasing training data

The current base model is:

`HuggingFaceTB/SmolLM2-135M`

The original dataset is:

`karthiksagarn/astro_horoscope`

------------------------------------------------------------------------

## 1. Project Structure

The horoscope experiments remain as separate scripts. New text-to-text tasks
use JSON task definitions and shared preparation, training, evaluation, and
inference scripts.

``` text
Hugging_Face_Fine_Tuning/
│
├── data/
│   ├── processed/                 # original horoscope split
│   └── tasks/<task_name>/         # separate prompt/response splits
│
├── models/
│   ├── smollm-horoscope-.../      # saved learning experiments
│   └── tasks/<task_name>/<run>/   # separate task runs
│
├── tasks/
│   ├── horoscope.json
│   └── conversation_summary.json
│
├── src/
│   ├── task_core.py
│   ├── prepare_task.py
│   ├── train_task.py
│   ├── evaluate_task.py
│   ├── infer_task.py
│   └── ...                         # original experiment scripts
│
├── docs/reusable_pipeline.md
└── requirements.txt
```

### Why keep separate scripts?

Keeping each experiment separate makes it easier to:

-   Compare implementations
-   Reproduce previous experiments
-   Avoid accidentally overwriting working code
-   Compare model outputs
-   Understand how each training technique changes the pipeline

This is especially useful while learning.

------------------------------------------------------------------------

## 2. Loading a Hugging Face Dataset

The original dataset is loaded with:

``` python
from datasets import load_dataset

dataset = load_dataset(
    "karthiksagarn/astro_horoscope",
    split="train",
)
```

The dataset contains 21,959 examples.

The original columns are:

``` text
sign
category
date
horoscope
```

This introduced the Hugging Face `datasets` library and the `Dataset`
abstraction.

Useful operations learned include:

``` python
dataset.map(...)
dataset.shuffle(...)
dataset.select(...)
dataset.train_test_split(...)
```

------------------------------------------------------------------------

## 3. Train/Test Splitting

The original dataset was divided into training and evaluation data:

``` python
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)
```

Result:

``` text
Training examples:   19,763
Evaluation examples:  2,196
```

The purpose of the evaluation dataset is to measure model behavior on
examples that were not directly used for training.

Using a fixed seed:

``` python
seed=42
```

makes the split reproducible.

------------------------------------------------------------------------

## 4. Dataset Preparation as a Separate Stage

Instead of doing all processing inside the training script, dataset
preparation was moved into:

``` text
src/prepare_dataset.py
```

The raw dataset is converted into instruction-formatted text and saved
locally.

The processed dataset is stored at:

``` text
data/processed/train
data/processed/test
```

Using:

``` python
dataset.save_to_disk(...)
```

The training scripts then use:

``` python
from datasets import load_from_disk

train_dataset = load_from_disk(
    "./data/processed/train"
)
```

### Application design lesson

Data preparation and model training are different responsibilities.

Separating them gives a cleaner pipeline:

``` text
Raw Dataset
     ↓
prepare_dataset.py
     ↓
Processed Dataset
     ↓
train_*.py
     ↓
Fine-Tuned Model
     ↓
inference_*.py
```

It also avoids repeating dataset preparation every time a model is
trained.

------------------------------------------------------------------------

## 5. Tokenization

Models do not directly process Python strings. Text must first be
converted into token IDs.

The tokenizer is loaded with:

``` python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

For the causal language model, the EOS token is also used as the padding
token:

``` python
tokenizer.pad_token = tokenizer.eos_token
```

Tokenization looks like:

``` python
tokens = tokenizer(
    text,
    truncation=True,
    max_length=256,
)
```

The important outputs include:

``` text
input_ids
attention_mask
```

### `input_ids`

These are integer IDs representing tokens.

Conceptually:

``` text
"Aries will have a great day"

        ↓ tokenizer

[1234, 891, 341, 27, 982, ...]
```

### `attention_mask`

The attention mask identifies which positions contain actual input
tokens versus padding.

Typically:

``` text
1 = real token
0 = padding
```

------------------------------------------------------------------------

## 6. Sequence Length

The initial horoscope dataset had relatively short examples.

Observed statistics for the original horoscope text were approximately:

``` text
Shortest: 37 tokens
Longest:  197 tokens
Average:   84 tokens
```

Therefore:

``` python
MAX_LENGTH = 256
```

was chosen instead of unnecessarily using 512.

This reduces memory usage and computation.

As the dataset format became larger due to instructions and metadata,
token statistics became important again.

A useful check was added:

``` python
truncated_examples = sum(
    1
    for length in lengths
    if length >= MAX_LENGTH
)
```

This helps detect whether examples are reaching the maximum sequence
length.

------------------------------------------------------------------------

## 7. Loading the Model

The current model is loaded using:

``` python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    torch_dtype="auto",
)
```

`AutoModelForCausalLM` is appropriate because the task is text
generation.

The model contains roughly 135 million parameters.

A larger Qwen model was initially tested, but full fine-tuning was too
slow and resource-intensive on an 8 GB Apple Silicon machine.

Switching to SmolLM2-135M made experimentation practical.

------------------------------------------------------------------------

## 8. Apple Silicon and MPS

PyTorch detected Apple's Metal Performance Shaders backend:

``` text
Trainer device:
mps
```

No manual CUDA configuration is required.

The Hugging Face `Trainer` automatically selected MPS.

One MPS-specific adjustment was:

``` python
dataloader_pin_memory=False
```

because pinned memory is not useful/supported in the same way on MPS.

### Hardware lesson

Model size matters significantly during full fine-tuning.

A smaller model made it possible to:

-   Train repeatedly
-   Debug quickly
-   Compare experiments
-   Avoid excessive memory pressure
-   Keep the Mac responsive

------------------------------------------------------------------------

## 9. Hugging Face Trainer

Training is managed with:

``` python
from transformers import (
    TrainingArguments,
    Trainer,
)
```

Example configuration:

``` python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    dataloader_pin_memory=False,
    report_to="none",
    seed=42,
)
```

The `Trainer` handles:

-   Forward passes
-   Loss calculation
-   Backpropagation
-   Optimizer steps
-   Learning-rate scheduling
-   Evaluation
-   Checkpointing
-   Logging

Example:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()
```

------------------------------------------------------------------------

## 10. Phase 1 --- Plain Causal Language Modeling

The first training approach used only the horoscope text.

Conceptually:

``` text
Horoscope text
      ↓
Causal Language Model
      ↓
Predict next token
```

The model learned patterns in horoscope-like text.

A language-modeling collator was used:

``` python
DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

`mlm=False` means causal language modeling rather than masked language
modeling.

### Limitation

The model learned the style/domain of horoscope text but was not
explicitly trained to map structured inputs such as sign, category, and
date to a horoscope.

------------------------------------------------------------------------

## 11. Phase 2 --- Instruction-Formatted Training

The dataset was changed from plain horoscope text into a structured
training record.

The application now creates examples containing:

``` text
Instruction
Sign
Category
Date
Horoscope
```

This teaches the model that metadata is related to the horoscope
response.

The major application-level change was therefore not the model
architecture.

It was the **representation of the training data**.

### Important lesson

Fine-tuning quality depends heavily on how the training task is
represented.

The same pretrained model can learn very different behavior depending on
the structure of its training examples.

------------------------------------------------------------------------

## 12. Phase 3 --- Response-Only Supervised Fine-Tuning

Phase 3 introduced the most important training-code change so far:
custom labels.

Instead of calculating loss across the entire prompt and response, only
response tokens contribute to the training loss.

Conceptually:

``` text
Prompt Tokens
     ↓
labels = -100
     ↓
Ignored by loss


Response Tokens
     ↓
labels = actual token IDs
     ↓
Included in loss
```

The labels are constructed like:

``` python
labels = (
    [-100] * len(prompt_tokens)
    + response_tokens
)
```

PyTorch loss functions used by Transformers treat:

``` text
-100
```

as an ignore index.

This means the model still receives the prompt as context but is trained
specifically on predicting the desired response.

------------------------------------------------------------------------

## 13. Custom Data Collator

Once labels were created manually, `DataCollatorForLanguageModeling` was
no longer appropriate for the response-only experiment.

A custom collator was implemented.

Its responsibilities are:

1.  Collect examples into a batch
2.  Pad `input_ids`
3.  Pad `attention_mask`
4.  Pad labels
5.  Use `-100` for label padding
6.  Return PyTorch tensors

Example concept:

``` python
batch["labels"] = torch.tensor(
    padded_labels,
    dtype=torch.long,
)
```

This was an important step toward understanding what high-level training
libraries normally handle automatically.

------------------------------------------------------------------------

## 14. Verifying Labels Before Training

A debugging/verification stage was added before starting expensive
training.

The script calculates:

``` text
Total tokens
Prompt tokens
Response tokens
```

and decodes prompt and response token groups separately.

This verifies that:

``` text
Prompt
→ ignored by loss

Response
→ included in loss
```

### Engineering lesson

Do not assume preprocessing is correct just because training starts
successfully.

Inspect transformed examples before spending time training.

------------------------------------------------------------------------

## 15. Saving Fine-Tuned Models

After training:

``` python
trainer.save_model(
    FINAL_MODEL_PATH
)

tokenizer.save_pretrained(
    FINAL_MODEL_PATH
)
```

The resulting directory contains files such as:

``` text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

These files allow the fine-tuned model to be loaded later without
retraining.

------------------------------------------------------------------------

## 16. Loading Local Models for Inference

A saved model is loaded using:

``` python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
)
```

The model is moved to MPS:

``` python
model = model.to("mps")
model.eval()
```

Inference is performed without gradient calculation:

``` python
with torch.no_grad():
    outputs = model.generate(...)
```

This reduces unnecessary memory and computation.

------------------------------------------------------------------------

## 17. Local Path Validation

An important debugging issue occurred when a model directory did not
exist.

If this path is supplied:

``` python
"./models/some-model/final"
```

but does not exist, Hugging Face may attempt to interpret it as a Hub
repository identifier.

This resulted in a confusing:

``` text
HFValidationError
```

A better application-level check was added:

``` python
from pathlib import Path

model_path = Path(MODEL_PATH)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path.resolve()}"
    )
```

### Engineering lesson

Validate local resources explicitly before passing them into libraries
that accept both local paths and remote identifiers.

This produces much clearer errors.

------------------------------------------------------------------------

## 18. Generation and Inference

Generation uses:

``` python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

A fixed random seed was added:

``` python
torch.manual_seed(42)
```

This makes comparisons between experiments more reproducible.

------------------------------------------------------------------------

## 19. Extracting Only Newly Generated Tokens

`model.generate()` returns both the original prompt tokens and newly
generated tokens.

The prompt length is calculated:

``` python
prompt_length = inputs["input_ids"].shape[1]
```

Then only generated tokens are selected:

``` python
generated_tokens = outputs[0][prompt_length:]
```

Finally:

``` python
horoscope = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)
```

This gives the application only the generated response instead of
repeating the entire input.

------------------------------------------------------------------------

## 20. Experiment Scaling: 512 → 2,000 Examples

Phase 3 trained response-only SFT on:

``` text
512 training examples
64 evaluation examples
```

The next experiment kept the same training method but increased the
dataset to:

``` text
2,000 training examples
200 evaluation examples
```

This is important experimental design.

Instead of changing:

-   Model
-   Learning rate
-   Dataset format
-   Masking
-   Generation parameters
-   Training technique

all at once, the experiment mainly changed the amount of training data.

That makes it easier to understand the effect of dataset size.

------------------------------------------------------------------------

## 21. Observable Result from Scaling

The 512-example response-only model produced weak output involving
unrelated calendar/lunar statements.

The 2,000-example response-only model generated text much closer to the
intended horoscope domain and also reflected the supplied year more
appropriately.

The important application lesson is:

> A correct training implementation does not guarantee useful behavior
> if the model has not seen enough representative training examples.

Increasing the training data while keeping the pipeline mostly unchanged
produced noticeably better task behavior.

------------------------------------------------------------------------

## 22. Training Loss vs Generation Quality

Several experiments produced different loss values.

However, loss values from different training objectives cannot always be
compared directly.

For example:

``` text
Phase 2
Loss calculated on:
Prompt + Response

Phase 3
Loss calculated on:
Response only
```

Therefore:

``` text
lower numerical loss
```

does not automatically mean:

``` text
better model
```

when the loss is being calculated over different token sets.

Model evaluation should include both:

-   Quantitative metrics
-   Actual generation behavior

------------------------------------------------------------------------

## 23. Reproducible Experiments

Several practices were introduced to make experiments reproducible:

``` python
SEED = 42
```

for dataset shuffling and training.

And:

``` python
torch.manual_seed(42)
```

for generation.

Each experiment also uses its own model directory.

Example:

``` text
smollm-horoscope-512
smollm-horoscope-instruction-512
smollm-horoscope-response-only-512
smollm-horoscope-response-only-2000
```

This prevents experiments from overwriting one another.

------------------------------------------------------------------------

## 24. Current Application Pipeline

The project now implements a complete small-scale LLM fine-tuning
workflow:

``` text
Hugging Face Dataset
        ↓
prepare_dataset.py
        ↓
Clean / Transform
        ↓
Train/Test Split
        ↓
save_to_disk()
        ↓
Processed Dataset
        ↓
Training Script
        ↓
Tokenizer
        ↓
input_ids / attention_mask
        ↓
Response-Only Labels
        ↓
Custom Data Collator
        ↓
SmolLM2-135M
        ↓
Hugging Face Trainer
        ↓
PyTorch / MPS
        ↓
Fine-Tuned Model
        ↓
save_model()
        ↓
Local Model Directory
        ↓
Inference Script
        ↓
model.generate()
        ↓
Generated Horoscope
```

------------------------------------------------------------------------

## 25. Key Libraries Used

### `datasets`

Used for:

-   Downloading datasets
-   Mapping transformations
-   Train/test splitting
-   Shuffling
-   Selecting subsets
-   Saving processed datasets
-   Loading processed datasets

### `transformers`

Used for:

-   Tokenizers
-   Pretrained models
-   Training arguments
-   Trainer
-   Data collators
-   Text generation

### `torch`

Used for:

-   MPS execution
-   Tensor creation
-   Random seeds
-   Gradient-free inference

### `pathlib`

Used for:

-   Local filesystem paths
-   Checking whether model directories exist
-   Cleaner path handling

------------------------------------------------------------------------

## 26. Important Debugging Lessons

### Model path errors

A missing local model path can be interpreted as a Hugging Face Hub
repository ID.

Always validate paths.

### Training success does not equal task success

A model can finish training with a reasonable loss while still
generating poor responses.

Always test inference.

### Dataset formatting matters

Changing the dataset representation significantly changes what the model
learns.

### Verify labels

Response masking should be inspected before training.

### Preserve experiments

Separate scripts and output directories make comparisons much easier.

### Hardware affects model choice

A theoretically better/larger model is not always the best model for
learning and iteration.

Smaller models enable faster experimentation.

------------------------------------------------------------------------

## 27. Experiments Completed

### Experiment 1 --- Smoke Test

``` text
Model: SmolLM2-135M
Examples: 32
Goal: Verify the complete training pipeline works
```

### Experiment 2 --- Plain Fine-Tuning

``` text
Model: SmolLM2-135M
Examples: 512
Training: Plain causal language modeling
Goal: Learn horoscope-style text
```

### Experiment 3 --- Instruction Fine-Tuning

``` text
Model: SmolLM2-135M
Examples: 512
Training: Instruction + response
Goal: Introduce structured task conditioning
```

### Experiment 4 --- Response-Only Fine-Tuning

``` text
Model: SmolLM2-135M
Examples: 512
Training: Response-only labels
Goal: Calculate loss only on the target horoscope
```

### Experiment 5 --- Scaled Response-Only Fine-Tuning

``` text
Model: SmolLM2-135M
Examples: 2,000
Training: Response-only labels
Goal: Measure the effect of increasing training data
```

------------------------------------------------------------------------

## 28. Current Technical Understanding

At this point the project has moved beyond simply calling:

``` python
trainer.train()
```

We now understand the major pieces underneath the training workflow:

``` text
Dataset
   ↓
Formatting
   ↓
Tokenizer
   ↓
input_ids
   ↓
attention_mask
   ↓
labels
   ↓
batching / padding
   ↓
model forward pass
   ↓
loss
   ↓
backpropagation
   ↓
optimizer
   ↓
updated model weights
```

The response-only experiment also introduced direct control over **which
tokens contribute to the loss**, which is an important foundation for
understanding supervised fine-tuning.

------------------------------------------------------------------------

## 29. Current State

The largest full fine-tuning experiment so far is the response-only model
trained on 2,000 examples:

``` text
models/smollm-horoscope-response-only-2000/final
```

The current inference pipeline successfully:

1.  Loads the local tokenizer
2.  Loads the fine-tuned model
3.  Moves it to MPS
4.  Tokenizes structured input
5.  Generates new tokens
6.  Removes prompt tokens from the returned sequence
7.  Decodes only the generated response

Work completed after this initial summary:

-   Input-sensitivity testing: four controlled prompts for each of the 512-
    and 2,000-example response-only models. See
    `results/input_sensitivity.md`.
-   Better evaluation: response-only loss on the same 128 held-out examples,
    plus 12 generated examples covering all signs and categories. The
    2,000-example model had lower held-out loss (3.203 versus 3.274), while
    generated text still showed category and date errors. See
    `results/heldout_review.md`.
-   Sequence-length analysis: the current 256-token limit truncates none of
    the prepared train or test examples. Increasing it is not the next useful
    experiment for this dataset. See `results/training_data_audit.md`.
-   Data audit: target horoscopes never repeat the exact date or year in the
    prepared data. Literal sign and category mentions are also uncommon;
    these counts alone do not establish whether the text follows those inputs.

Current priorities after the 5 October evaluations, in a useful order:

1.  Define what correct sign, category, and date use looks like in generated
    text. Review and improve training targets, then use a human-reviewed rubric
    on prompts outside the validation slice. Literal word counts alone are not
    an adherence score; see `results/independent_adherence.md`.
2.  Scale an adapter run beyond 512 training examples while holding the prompt,
    training method, and independent evaluation fixed. The full response-only
    model has already been trained on 2,000 examples; the LoRA and MLX QLoRA
    adapters have not.
3.  Compare batch size and gradient accumulation for training speed, memory,
    and quality while keeping the effective batch size and data fixed.
4.  Add reproducible inference controls such as sampling temperature, top-p,
    repetition penalty, and seed to the command-line inference path.
5.  If supported CUDA hardware becomes available, run and verify the separate
    bitsandbytes QLoRA path. Mac MLX QLoRA has already completed a 512-step
    run and an independent held-out check; see sections 32–33.

------------------------------------------------------------------------

## 30. LoRA and QLoRA experiment

`src/train_lora.py` now trains a response-only LoRA adapter on the same
512-example subset used for the earlier full fine-tuning experiment. It adapts
the attention query and value projections (`q_proj` and `v_proj`) while leaving
the base model frozen. The run updated 460,800 of 134,975,808 parameters
(about 0.34%) and saved a 1.8 MB adapter at
`models/smollm-horoscope-lora-512/final`.

The held-out comparison scored the same 128 test examples for all models:

| Model | Response-only NLL |
| --- | ---: |
| Full fine-tune, 512 examples | 3.274 |
| Full fine-tune, 2,000 examples | 3.203 |
| LoRA, 512 examples | 3.105 |

This LoRA configuration did better on the measured held-out loss, but the
experiment also changed the learning rate. The comparison therefore does not
show that LoRA is inherently better than full fine-tuning. Generated outputs
still need review for sign, category, and date control. Full outputs are in
`results/heldout_lora_comparison.md`.

`src/inference_lora.py` loads the base model and adapter and accepts sign,
category, date, device, and output-length options. A QLoRA option using
bitsandbytes NF4 is present in the PyTorch training script for a supported
CUDA GPU; that CUDA route has not been run here.

For Apple Silicon, `mlx-lm` provides a separate QLoRA route. The base model
was converted to 4-bit MLX format, the existing training split was exported
as prompt/completion JSONL, and `configs/mlx_qlora_smoke.yaml` completed a
two-iteration run with prompt masking. The adapter was saved and loaded for
generation. That short run verified the Mac workflow but did not produce a
useful horoscope. See `results/mlx_qlora_smoke.md` for the run details.

------------------------------------------------------------------------

## 31. 4 October 2026 --- Testing QLoRA on Apple Silicon

The earlier CUDA requirement applied to the project's **PyTorch +
bitsandbytes** QLoRA implementation. It was not a general limitation of Mac
fine-tuning. We tested a separate **MLX-LM** workflow on this Mac's Metal GPU.
MLX-LM calls the run QLoRA when a LoRA adapter is trained over a quantized
base model. That is why the saved YAML has `fine_tune_type: lora` even though
the experiment uses 4-bit base weights. These MLX adapters have a different
format from the PyTorch/PEFT adapters.

### Environment and model conversion

We kept MLX separate from the project's Python 3.14 PyTorch environment:

``` bash
python3.13 -m venv .venv-mlx
.venv-mlx/bin/python -m pip install -r requirements-mlx-qlora.txt
.venv-mlx/bin/python -m mlx_lm convert \
    --hf-path HuggingFaceTB/SmolLM2-135M \
    --mlx-path models/smollm2-135m-mlx-4bit \
    --quantize --q-bits 4 --q-group-size 64
```

The tested environment used MLX-LM 0.32.0 and MLX 0.32.3. Conversion produced
a 4-bit affine model of about 76 MB. MLX reported approximately 4.5 bits per
weight after quantization overhead.

### Preserve the response-only training objective

`src/prepare_mlx_qlora.py` exported the same prepared train/test split to
MLX-LM's prompt/completion JSONL format. It also wrote a tokenizer template
that concatenates the existing raw prompt and target response without adding
chat role markers. A tokenization check showed the prompt token IDs matched
the original prompt. With `mask_prompt: true`, the training loss is applied
only to the response. The first checked example had 52 prompt tokens and 121
response tokens, including the end-of-sequence token.

``` bash
.venv-mlx/bin/python src/prepare_mlx_qlora.py \
    --train-samples 32 --valid-samples 8
.venv-mlx/bin/python -m mlx_lm lora \
    --config configs/mlx_qlora_smoke.yaml
```

To repeat the smoke test without replacing its saved adapter, add
`--adapter-path models/smollm-horoscope-mlx-qlora-rerun` to the last command.

The YAML sets batch size 1, 256 maximum tokens, 16 adapted layers, rank 8,
learning rate `2e-4`, and two training iterations. The quantized model
determines that this is QLoRA; the YAML's LoRA setting selects the adapter
method.

### What the smoke test established

The two-step run trained 1.303 million parameters (0.968% of the model) and
saved an adapter of about 5 MB. Reported training loss changed from 4.290 to
4.058. Validation loss was 3.595 at the first report and 4.161 at the second,
so this tiny run gives no evidence of quality improvement. The adapter loaded
for generation, but the output was repeated section labels rather than a
useful horoscope.

This was a **pipeline test**: 4-bit conversion, response-only data preparation,
training, validation, saving, and loading all worked on the Mac. A meaningful
quality comparison needs a longer MLX run and held-out evaluation. The MLX
adapter must be evaluated through MLX tooling; it cannot be loaded by
`src/inference_lora.py`, which expects a PyTorch/PEFT adapter. See
`results/mlx_qlora_smoke.md` and the [MLX-LM LoRA/QLoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

------------------------------------------------------------------------

## 32. 5 October 2026 --- A longer Mac QLoRA run

The two-step test only established that the workflow ran. We next trained a
4-bit MLX base plus LoRA adapter on 512 examples for 512 iterations, using
batch size 1, rank 8, 16 adapted layers, response-only masking, and a
256-token limit. The adapter is saved at
`models/smollm-horoscope-mlx-qlora-512`.

``` bash
.venv-mlx/bin/python src/prepare_mlx_qlora.py --train-samples 512 --valid-samples 64 --test-samples 128 --output-dir data/mlx-qlora-512
.venv-mlx/bin/python -m mlx_lm lora --config configs/mlx_qlora_512.yaml
.venv-mlx/bin/python -m mlx_lm lora --model models/smollm2-135m-mlx-4bit --data data/mlx-qlora-512 --adapter-path models/smollm-horoscope-mlx-qlora-512 --test --mask-prompt --batch-size 1 --max-seq-length 256 --test-batches 128
.venv-mlx/bin/python src/review_mlx_qlora.py
```

Validation loss fell from 3.893 to 2.970. On the same 128 held-out examples
used for the earlier PyTorch comparison, the 4-bit base scored 3.902 loss
(49.521 perplexity), while the trained MLX adapter scored 2.998 loss (20.037
perplexity). Both evaluations covered 10,770 response tokens. The PyTorch
LoRA-512 result was 3.1047 loss, but the adapter settings and implementations
differ, so these scores do not prove MLX QLoRA is inherently better.

The first 64 of the 128 test examples also served as validation cases. This
mirrors the earlier comparison, but a future final test should use examples
that were never used for validation. The 12 generated review cases showed
horoscope-like prose after training, with lingering category and date errors
and occasional incomplete endings. Reference loss improved substantially
without resolving instruction adherence. The next learning step is to define
an independent test slice and measure sign, category, and date adherence
separately from response loss. See `results/heldout_mlx_qlora_512.md` and
`results/heldout_mlx_qlora_512.json`.

------------------------------------------------------------------------

## 33. 5 October 2026 --- Independent test and prompt-field cues

The first MLX comparison reused 64 validation examples in its 128-example
score. We now reserved indices 128–255 from the same seed-42 shuffled test
split for a separate final check. These examples were neither training nor
validation data. All three models scored the same 10,754 response tokens:

| Model | Response loss | Perplexity |
| --- | ---: | ---: |
| MLX 4-bit base | 3.935 | 51.155 |
| MLX QLoRA-512 | 3.015 | 20.387 |
| PyTorch PEFT LoRA-512 | 3.1193 | 22.63 |

This confirms a substantial within-MLX improvement over the quantized base
on untouched examples. The numerical MLX/PEFT difference still does not
isolate quantization because the adapters differ in adapted layers and
implementation.

We generated 12 matching independent prompts and audited explicit sign,
category, and date cues. The references named the requested sign in only
2/12 cases and repeated the exact date in 0/12. The PEFT outputs did so in
2/12 and 0/12; the MLX outputs in 1/12 and 0/12. All three sets contained an
explicit category cue in 5/10 non-General cases. These counts cannot measure
semantic adherence by themselves: a horoscope can address a date without
printing it, and category words can appear in off-topic text. A few outputs
were clearly on topic (both Libra/Career and Capricorn/Love), while others
were generic or drifted (both Aries/Career and Sagittarius/Love).

The important evaluation lesson is to define an observable target before
assigning a percentage. Sign and date matching need a rubric that says what
correct use looks like, not just a string search. See
`results/independent_adherence.md`, `results/independent_evaluation.json`,
and `results/independent_adherence.json` for commands, metrics, full output,
and per-case cues.

------------------------------------------------------------------------

## 34. 5–6 October 2026 --- Generalizing beyond horoscopes

The earlier scripts embedded horoscope columns, prompts, and paths directly
in Python. Changing only the Hugging Face dataset name would therefore fail
for another task. We added `tasks/*.json` definitions and a shared
`prompt`/`response` JSONL format. The generic scripts now prepare separate
train/validation/test splits, mask prompt labels during training, save a
separate adapter or full model per task, evaluate on untouched test examples,
and generate from named input fields. The original horoscope scripts and
models remain available.

The first new task is SAMSum conversation summarization:

``` bash
.venv/bin/python src/prepare_task.py --task tasks/conversation_summary.json
.venv/bin/python src/train_task.py --task tasks/conversation_summary.json
.venv/bin/python src/evaluate_task.py --run-dir models/tasks/conversation_summary/lora-512
.venv/bin/python src/infer_task.py --run-dir models/tasks/conversation_summary/lora-512 --input $'dialogue=A: Can we meet on Friday?\nB: Yes. Please bring the notes.\nA: I will.'
```

Preparation produced 14,731 training, 818 validation, and 819 test examples.
The initial 512-step Mac MPS run exhausted memory at step 230. Enabling
gradient checkpointing and periodic cache clearing allowed the retry to
finish. Its saved training runtime was 198.8 seconds. The LoRA adapter
updated 460,800 parameters; the trainer selected 512 fitting training
examples and 64 fitting validation examples under a 512-token limit.

On the same 128 test conversations, the untouched base had response loss
2.2946 and the adapter had 1.7667. This shows improvement in predicting
reference summary text. Sample outputs still changed names and invented
events, so factual summary quality remains a separate problem. See
`results/conversation_summary_first_run.md` and
`docs/reusable_pipeline.md` for the run and the reusable workflow.

For a future dataset, create a new task file with its source split names,
prompt placeholders, and response column. The reusable path currently covers
PyTorch full fine-tuning and PEFT LoRA; the older MLX QLoRA route is still
separate. Starting each task from SmolLM2-135M yields an independent adapter.
Continuing from a saved full checkpoint is available with `--starting-model`;
continuing from a previous LoRA adapter is not part of this path.

------------------------------------------------------------------------

## 35. 6 October 2026 --- One pipeline for four scenarios

The reusable task path now has definitions for horoscope generation,
conversation summarization, CommonGen concept-to-sentence generation, and
emotion classification. It maps numeric emotion labels to their names and
turns CommonGen concept lists into comma-separated prompt text. CommonGen's
published Parquet files are used because the installed `datasets` version
cannot execute its older repository loading script.

Each task has prepared data under `data/tasks/<task>/`. The prepared counts
are 19,763/1,098/1,098 for horoscope, 14,731/818/819 for conversation
summary, 60,674/6,714/993 for concept sentence, and 16,000/2,000/2,000
for emotion classification (train/validation/test). CommonGen keeps repeated
concept sets together; an overlap check found zero matching concept sets
across its three prepared splits. Its published validation split is used as
the labelled test set.

Two-step LoRA smoke runs completed on CPU for concept sentence, emotion
classification, and the generic horoscope task. Small test evaluations
completed for all three; saved-model inference completed for the two new
scenarios. These runs verify
the software path, not learned task quality. The concept
model often omits required concepts or emits extra explanation. The emotion
model can produce extra text instead of one label. The evaluator now reports
exact concept word coverage, strict emotion-label accuracy, and ROUGE-L word
overlap for new summary evaluations. These metrics are diagnostics; direct
review is still required, especially for summary facts and naturalness.

See `docs/reusable_pipeline.md` for commands to train and evaluate each
scenario. The original SAMSum adapter still has its prior evaluation; its
saved task snapshot predates the new ROUGE-L setting, so a new run is needed
to use that metric without changing the historic record.

------------------------------------------------------------------------

## 36. 6 October 2026 --- First full emotion-classification run

The reusable pipeline trained a SmolLM2-135M LoRA adapter on 512 emotion
examples and validated on 64. Training took 67.3 seconds in the recorded run.
The adapter was evaluated on 128 held-out examples, with zero examples
skipped for length. The untouched starting model was evaluated on the same
slice and generation settings.

| Model | Response loss | Strict label accuracy | Macro-F1 | Invalid label outputs |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 4.0100 | 0/128 | 0.000 | 128/128 |
| 512-example LoRA | 0.3958 | 85/128 (66.4%) | 0.382 | 0/128 |

Strict accuracy accepts only an output that is exactly one of the six labels.
The starting model often emitted a label followed by extra text, so its zero
score is a format failure under this rule; it is not a general semantic
classification comparison. The adapter produced valid labels, but its
per-class results show uneven performance:

| Label | Test examples | Correct | Recall |
| --- | ---: | ---: | ---: |
| sadness | 45 | 31 | 68.9% |
| joy | 45 | 43 | 95.6% |
| love | 6 | 0 | 0% |
| anger | 17 | 10 | 58.8% |
| fear | 10 | 1 | 10.0% |
| surprise | 5 | 0 | 0% |

The adapter never predicted `love` or `surprise` in these 128 examples.
Most `love`, `surprise`, and `fear` cases were classified as `joy` or
`sadness`. This small test slice has only five or six examples for the
rarest labels, so full-test or stratified evaluation is needed before making
a stable per-class claim. The saved evaluation JSON files are under
`models/tasks/emotion_classification/lora-512/` and remain local.

The selected 512 training examples contained 158 `joy`, 145 `sadness`,
85 `anger`, 59 `fear`, 43 `love`, and 22 `surprise` labels. The model saw
every class, but `love` and `surprise` had far fewer examples than the two
most common labels. This imbalance may contribute to the missing predictions;
the current run alone does not establish the cause.

The next experiment should address class coverage before trying a larger
base model: evaluate on a larger balanced or complete test set, then compare
a larger or class-balanced training subset using the same held-out examples.

------------------------------------------------------------------------

## 37. 6 October 2026 --- Comparing SmolLM2-135M and SmolLM2-360M

The same 512 prepared emotion training examples and 64 validation examples
were used to train a LoRA adapter on SmolLM2-360M. The recorded MPS training
runtime was 109.0 seconds, compared with 67.3 seconds for SmolLM2-135M.
Both runs were evaluated on the exact same 128 prompts and references. The
saved tokenizers have identical files, and both scored 361 response tokens.

| Model | Test loss | Strict accuracy | Macro-F1 | Invalid outputs |
| --- | ---: | ---: | ---: | ---: |
| Untuned SmolLM2-135M | 4.0100 | 0/128 | 0.000 | 128 |
| SmolLM2-135M + LoRA | 0.3958 | 85/128 (66.4%) | 0.382 | 0 |
| Untuned SmolLM2-360M | 2.6325 | 0/128 | 0.000 | 128 |
| SmolLM2-360M + LoRA | 0.3823 | 93/128 (72.7%) | 0.539 | 1 |

The 360M adapter was correct on 14 examples the 135M adapter missed; the
135M adapter was correct on six examples the 360M adapter missed. The larger
adapter identified eight of ten `fear` cases versus one of ten for 135M.
It identified one of six `love` cases, but still identified none of the five
`surprise` cases. The single invalid 360M output was `trustworthiness`.
Both untuned models failed the strict output-format check on every sampled
case; they often continued the prompt or wrote explanatory text instead of
returning one exact label. The 360M starting model had lower response loss
than the 135M starting model on the same 361 reference tokens, but neither
base model produced a valid classifier output under this prompt and decoding
rule. All four evaluations used identical prompts and references.

This 128-example slice is small for rare-class conclusions. The next
comparison should use a larger or stratified held-out sample before changing
model size again.

------------------------------------------------------------------------

## 38. 6 October 2026 --- Balanced emotion evaluation

The two saved emotion adapters were evaluated on the same deterministic
150-example test slice, with 25 examples for each of the six labels. Every
selected row fit the context limit, and neither adapter produced an invalid
label. This is a deliberately balanced diagnostic, not an estimate of
accuracy under the original test-set class distribution.

| Trained model | Response loss | Exact accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| SmolLM2-135M + LoRA | 0.7919 | 60/150 (40.0%) | 0.306 |
| SmolLM2-360M + LoRA | 0.6872 | 78/150 (52.0%) | 0.469 |

| Label | 135M correct / 25 | 360M correct / 25 |
| --- | ---: | ---: |
| sadness | 19 | 19 |
| joy | 24 | 23 |
| love | 1 | 5 |
| anger | 13 | 20 |
| fear | 3 | 9 |
| surprise | 0 | 2 |

The 360M adapter was correct on 24 cases the 135M adapter missed; the 135M
adapter was correct on six cases the 360M adapter missed. The larger model
improved coverage of the weaker classes but still missed most `love`, `fear`,
and `surprise` examples. The lower balanced accuracies relative to the
earlier random 128-example evaluation reflect this equal class mix; they
do not indicate a regression on the original test distribution.

Next, compare a larger training subset on a fixed, balanced **validation**
slice. The test set has already been inspected, so repeated training decisions
should use validation examples. Increasing to 2,000 training examples is a
controlled first step; class-balanced training can be a separate experiment
if rare-label recall remains poor. Keep the 512-example adapters and the
validation selection unchanged for the comparison, then run a final test
evaluation after choosing the training setup.

------------------------------------------------------------------------

## 39. 6 October 2026 --- Increasing 360M emotion training to 2,000 examples

SmolLM2-360M + LoRA trained for one epoch on 2,000 emotion examples, with the
same task configuration, seed, and 64 monitored validation examples as the
512-example run. MPS training took 441.7 seconds (about 7 minutes 22 seconds),
with train loss 0.4659. The adapter is saved under
`models/tasks/emotion_classification/smollm2-360m-lora-2000/`.

Both 360M adapters were then compared on the same deterministic, balanced
150-example **validation** slice (25 examples per label):

| Training examples | Response loss | Exact accuracy | Macro-F1 | Invalid labels |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.7988 | 67/150 (44.7%) | 0.410 | 0 |
| 2,000 | 0.4920 | 91/150 (60.7%) | 0.598 | 0 |

| Label | 512 correct / 25 | 2,000 correct / 25 |
| --- | ---: | ---: |
| sadness | 15 | 20 |
| joy | 24 | 24 |
| love | 3 | 11 |
| anger | 12 | 12 |
| fear | 10 | 14 |
| surprise | 3 | 10 |

The 2,000-example adapter gained 28 cases that the 512-example adapter missed
and lost four cases it previously got right. Its largest recall gains were
`love` (3 to 11 correct) and `surprise` (3 to 10 correct). This supports
selecting the 2,000-example setup for a final test evaluation, while the
remaining errors leave room for further improvement. Eight of the 150
balanced cases overlap the 64 examples used for the training run's end-of-epoch
validation loss; none were training examples. The balanced class mix is a
diagnostic and does not represent the original class distribution.

Next, evaluate the selected 2,000-example adapter once on the held-out test
split. Compare it with the already saved 512-example balanced test result,
then measure accuracy on a fixed, ordinary test sample to assess performance
under the original label mix. Do not use those test results to choose another
training configuration.

------------------------------------------------------------------------

## 40. 6 October 2026 --- Balanced test of the 2,000-example emotion adapter

The selected SmolLM2-360M + LoRA adapter trained on 2,000 examples was
evaluated on the same 150 balanced **test** examples as the earlier
512-example adapter. The comparison script verified identical prompts and
references, with 25 examples for each label.

| Training examples | Response loss | Exact accuracy | Macro-F1 | Invalid labels |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.6872 | 78/150 (52.0%) | 0.469 | 0 |
| 2,000 | 0.4734 | 96/150 (64.0%) | 0.624 | 0 |

| Label | 512 correct / 25 | 2,000 correct / 25 |
| --- | ---: | ---: |
| sadness | 19 | 22 |
| joy | 23 | 23 |
| love | 5 | 9 |
| anger | 20 | 19 |
| fear | 9 | 14 |
| surprise | 2 | 9 |

The 2,000-example adapter corrected 20 cases missed by the 512-example
adapter and lost two cases it previously got right. `Surprise` recall rose
from 2/25 to 9/25, though 16 of 25 `surprise` examples were still missed.
The test result supports the validation-based choice of the 2,000-example
setup. Because this slice has equal class counts, its 64.0% accuracy is a
balanced diagnostic, not accuracy under the original test distribution.
The test set had already been inspected in earlier experiments, so avoid
using this result to choose further training settings.

Next, generate answers for the same ordinary 128-example test sample used
by the 512-example adapter. This measures strict-label accuracy under the
test sample's original class mix, while the saved balanced result remains
the rare-label diagnostic.

------------------------------------------------------------------------

## 41. 6 October 2026 --- Ordinary test sample for the 2,000-example adapter

The selected 2,000-example SmolLM2-360M + LoRA adapter was also evaluated
on the same 128 ordinary **test** cases as the earlier 512-example adapter.
The comparison verified identical prompts and references. This sample
retains the test split's label mix rather than forcing equal class counts.

| Training examples | Response loss | Exact accuracy | Macro-F1 | Invalid labels |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.3823 | 93/128 (72.7%) | 0.539 | 1 |
| 2,000 | 0.2859 | 100/128 (78.1%) | 0.568 | 0 |

| Label | Support | 512 correct | 2,000 correct |
| --- | ---: | ---: | ---: |
| sadness | 45 | 28 | 34 |
| joy | 45 | 43 | 43 |
| love | 6 | 1 | 1 |
| anger | 17 | 13 | 14 |
| fear | 10 | 8 | 8 |
| surprise | 5 | 0 | 0 |

The 2,000-example adapter corrected ten cases missed by the 512-example
adapter and lost three cases it previously got right. The gain is mainly in
`sadness` on this sample. There are only five `surprise` examples here,
and both models missed all five; the separate balanced test found 9/25
`surprise` examples correct for the 2,000-example adapter. It also predicted
`surprise` for three other cases on the ordinary sample, so those predictions
were false positives. The balanced and ordinary samples answer different
questions and should be reported together.

The selected 2,000-example run now has both balanced and ordinary test
results. Further training changes should be chosen using validation data,
then checked on a new untouched test split where possible; repeatedly
adjusting to these inspected test cases would overstate generalization.

------------------------------------------------------------------------

## 42. 6 October 2026 --- Validation error review and balanced training option

The saved 2,000-example SmolLM2-360M validation generations show a repeated
confusion: of 25 `love` references, 11 were correct and 10 were predicted as
`joy`; of 25 `surprise` references, 10 were correct and 10 were predicted as
`joy`. The remaining `love` errors were three `sadness` and one `anger`;
the remaining `surprise` errors were four `fear` and one `love`. Some texts
are ambiguous, so these counts describe agreement with dataset labels.

The original 2,000-example shuffled training subset had 174 `love` and 80
`surprise` rows, compared with 591 `sadness` and 639 `joy` rows. The full
training split has enough unique examples to draw about 333 from each of the
six labels. I added `--balanced-train` to the reusable trainer for this
controlled experiment. It leaves the 64-example validation selection alone,
uses no duplicate training rows, and records label counts in the run summary.

A deterministic selection check produced 334 `sadness`, 334 `joy`, and 333
examples for each other label. A two-step CPU LoRA training check completed
and saved an adapter. The complete 2,000-example balanced run has not yet
been trained; this local tool environment reports MPS unavailable. The exact
MPS training and paired validation commands are in the
[pipeline guide](docs/reusable_pipeline.md#compare-equal-count-emotion-training).
Compare rare-label recall and macro-F1 on the balanced validation slice, then
check ordinary validation accuracy before selecting a setup. The aggregate
confusion table and rationale are in the
[validation error review](docs/emotion_validation_error_review.md).

------------------------------------------------------------------------

## 43. 6 October 2026 --- Equal-count emotion training on MPS

The full SmolLM2-360M + LoRA run completed with 2,000 unique training rows:
334 each for `sadness` and `joy`, and 333 each for `love`, `anger`, `fear`,
and `surprise`. The saved run summary reports 434.3 seconds of training on
MPS and no examples skipped for length.

Both 2,000-example adapters were evaluated on the same 150 balanced
**validation** examples. The comparison verified identical prompts and
references; neither model produced an invalid label.

| Training subset | Response loss | Exact accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Original shuffled subset | 0.4920 | 91/150 (60.7%) | 0.598 |
| Equal-count subset | 0.3330 | 106/150 (70.7%) | 0.707 |

| Label | Original correct / 25 | Equal-count correct / 25 |
| --- | ---: | ---: |
| sadness | 20 | 18 |
| joy | 24 | 20 |
| love | 11 | 19 |
| anger | 12 | 15 |
| fear | 14 | 15 |
| surprise | 10 | 19 |

The equal-count adapter corrected 25 cases the original adapter missed and
lost ten cases it previously got right. The intended rare-label gain is
visible: `love` rose by eight correct cases and `surprise` by nine. The
tradeoff is lower `joy` recall (24/25 to 20/25) and `sadness` recall
(20/25 to 18/25). `Joy` F1 still rose from 0.632 to 0.755 because fewer
other labels were incorrectly predicted as `joy`.

Do not select a model from this balanced slice alone. The next check is a
paired, ordinary 128-example validation sample for both 2,000-example
adapters. Compare overall accuracy, macro-F1, and per-label recall under
that original label mix before deciding whether equal-count training is
preferable for this task. The commands are in the
[pipeline guide](docs/reusable_pipeline.md#compare-equal-count-emotion-training).

------------------------------------------------------------------------

## Summary

The main application-level progression has been:

``` text
Load a pretrained model
        ↓
Fine-tune on plain text
        ↓
Separate dataset preparation
        ↓
Introduce structured training examples
        ↓
Create response-only labels
        ↓
Implement a custom collator
        ↓
Train on Apple MPS
        ↓
Save/load local models safely
        ↓
Build reproducible inference
        ↓
Scale training data
        ↓
Compare model behavior
        ↓
Train a PEFT LoRA adapter on MPS
        ↓
Test 4-bit MLX QLoRA on Apple Metal
        ↓
Train and evaluate 512-example MLX QLoRA
        ↓
Evaluate on a separate test slice and audit prompt-field cues
        ↓
Generalize the pipeline and train a SAMSum summarization adapter
```

The most important lesson so far is that fine-tuning is not only about
choosing a model and calling `Trainer.train()`.

The quality and behavior of the final model depend on the complete
application pipeline:

**dataset preparation → tokenization → labels → loss objective →
training configuration → model saving → inference → evaluation.**
