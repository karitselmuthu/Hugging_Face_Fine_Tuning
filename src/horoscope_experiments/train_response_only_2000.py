from pathlib import Path
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

TRAIN_DATA_PATH = Path("./data/processed/train")
TEST_DATA_PATH = Path("./data/processed/test")

OUTPUT_DIR = "./models/smollm-horoscope-response-only-2000"
FINAL_MODEL_PATH = f"{OUTPUT_DIR}/final"

MAX_LENGTH = 256

TRAIN_SAMPLES = 2000
EVAL_SAMPLES = 200

SEED = 42

HOROSCOPE_MARKER = "### Horoscope:"


# ============================================================
# 1. Load Prepared Dataset
# ============================================================

print("\nLoading prepared dataset...")

train_dataset = load_from_disk(
    str(TRAIN_DATA_PATH)
)

eval_dataset = load_from_disk(
    str(TEST_DATA_PATH)
)

print("Dataset loaded successfully!")

print(f"Available training examples: {len(train_dataset):,}")
print(f"Available evaluation examples: {len(eval_dataset):,}")


# ============================================================
# 2. Select Subsets
# ============================================================

print("\nSelecting training subsets...")

train_dataset = (
    train_dataset
    .shuffle(seed=SEED)
    .select(
        range(
            min(TRAIN_SAMPLES, len(train_dataset))
        )
    )
)

eval_dataset = (
    eval_dataset
    .shuffle(seed=SEED)
    .select(
        range(
            min(EVAL_SAMPLES, len(eval_dataset))
        )
    )
)

print(f"Selected training examples: {len(train_dataset):,}")
print(f"Selected evaluation examples: {len(eval_dataset):,}")


# ============================================================
# 3. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully!")
print(f"Vocabulary size: {len(tokenizer):,}")


# ============================================================
# 4. Tokenization + Response-Only Labels
# ============================================================

def tokenize_function(example):

    full_text = example["text"]

    prompt_part, response_part = full_text.split(
        HOROSCOPE_MARKER,
        maxsplit=1,
    )

    prompt = (
        prompt_part
        + HOROSCOPE_MARKER
        + "\n"
    )

    response = response_part.strip()

    # Tokenize prompt
    prompt_tokens = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    # Tokenize response
    response_tokens = tokenizer(
        response + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    # Combine
    input_ids = (
        prompt_tokens
        + response_tokens
    )

    # Ignore prompt when calculating loss
    labels = (
        [-100] * len(prompt_tokens)
        + response_tokens
    )

    # Truncate
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 5. Tokenize Dataset
# ============================================================

print("\nTokenizing training dataset...")

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names,
)

print("Tokenizing evaluation dataset...")

tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    remove_columns=eval_dataset.column_names,
)

print("Tokenization completed!")


# ============================================================
# 6. Dataset Statistics
# ============================================================

lengths = [
    len(example["input_ids"])
    for example in tokenized_train_dataset
]

print("\nToken statistics:")

print(f"Shortest: {min(lengths)}")
print(f"Longest:  {max(lengths)}")

print(
    f"Average:  "
    f"{sum(lengths) / len(lengths):.2f}"
)

truncated_examples = sum(
    1
    for length in lengths
    if length >= MAX_LENGTH
)

print(
    f"Examples reaching MAX_LENGTH: "
    f"{truncated_examples:,}"
)
# ============================================================
# 7. Verify Response-Only Labels
# ============================================================

example = tokenized_train_dataset[0]

input_ids = example["input_ids"]
labels = example["labels"]

prompt_token_count = sum(
    1
    for label in labels
    if label == -100
)

response_token_count = sum(
    1
    for label in labels
    if label != -100
)

print("\n" + "=" * 70)
print("RESPONSE-ONLY LABEL CHECK")
print("=" * 70)

print(f"Total tokens:    {len(input_ids)}")
print(f"Prompt tokens:   {prompt_token_count}")
print(f"Response tokens: {response_token_count}")


prompt_ids = [
    token_id
    for token_id, label in zip(input_ids, labels)
    if label == -100
]

response_ids = [
    token_id
    for token_id, label in zip(input_ids, labels)
    if label != -100
]


print("\n--- PROMPT (ignored by loss) ---\n")

print(
    tokenizer.decode(
        prompt_ids,
        skip_special_tokens=True,
    )
)


print("\n--- RESPONSE (used for loss) ---\n")

print(
    tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )
)
# ============================================================
# 8. Custom Data Collator
# ============================================================

class ResponseOnlyDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):

        labels = [
            feature["labels"]
            for feature in features
        ]

        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]

        padded_labels = []

        for label in labels:

            padding_length = (
                max_length - len(label)
            )

            padded_label = (
                label
                + [-100] * padding_length
            )

            padded_labels.append(
                padded_label
            )

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch


data_collator = ResponseOnlyDataCollator(
    tokenizer
)
# ============================================================
# 9. Load Model
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)

print("Model loaded successfully!")

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Model parameters: {total_parameters:,}")


# ============================================================
# 10. Training Configuration
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    num_train_epochs=1,

    learning_rate=2e-5,

    logging_steps=100,

    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=1,

    dataloader_pin_memory=False,

    report_to="none",

    seed=SEED,
)


# ============================================================
# 11. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,

    data_collator=data_collator,

    processing_class=tokenizer,
)


# ============================================================
# 12. Training Summary
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 - RESPONSE-ONLY TRAINING (2000 EXAMPLES)")
print("=" * 70)

print(f"Model:            {MODEL_NAME}")
print(f"Device:           {training_args.device}")
print(f"Training samples: {len(tokenized_train_dataset):,}")
print(f"Eval samples:     {len(tokenized_eval_dataset):,}")
print(f"Epochs:           {training_args.num_train_epochs}")
print(f"Batch size:       {training_args.per_device_train_batch_size}")
print(f"Learning rate:    {training_args.learning_rate}")

print("=" * 70)


# ============================================================
# 13. Train
# ============================================================

print("\nStarting Phase 4 fine-tuning...\n")

trainer.train()


# ============================================================
# 14. Save Model
# ============================================================

print("\nSaving final model...")

trainer.save_model(
    FINAL_MODEL_PATH
)

tokenizer.save_pretrained(
    FINAL_MODEL_PATH
)

print("\nTraining completed successfully!")

print(
    f"Model saved to: {FINAL_MODEL_PATH}"
)