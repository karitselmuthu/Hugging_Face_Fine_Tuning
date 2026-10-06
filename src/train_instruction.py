from pathlib import Path

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"

TRAIN_DATA_PATH = Path("./data/processed/train")
TEST_DATA_PATH = Path("./data/processed/test")

OUTPUT_DIR = "./models/smollm-horoscope-instruction-512"
FINAL_MODEL_PATH = f"{OUTPUT_DIR}/final"

MAX_LENGTH = 256

TRAIN_SAMPLES = 512
EVAL_SAMPLES = 64

SEED = 42


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

print("\nColumns:")
print(train_dataset.column_names)


# ============================================================
# 2. Inspect Prepared Example
# ============================================================

print("\n" + "=" * 70)
print("PREPARED TRAINING EXAMPLE")
print("=" * 70)

print(train_dataset[0]["text"])


# ============================================================
# 3. Select Training Subsets
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

print(f"Selected training examples: {len(train_dataset)}")
print(f"Selected evaluation examples: {len(eval_dataset)}")


# ============================================================
# 4. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully!")
print(f"Vocabulary size: {len(tokenizer):,}")


# ============================================================
# 5. Tokenization Function
# ============================================================

def tokenize_function(examples):

    texts = [
        text + tokenizer.eos_token
        for text in examples["text"]
    ]

    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ============================================================
# 6. Tokenize Dataset
# ============================================================

print("\nTokenizing training dataset...")

tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("Tokenizing evaluation dataset...")

tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
)

print("Tokenization completed!")


# ============================================================
# 7. Token Statistics
# ============================================================

train_lengths = [
    len(example["input_ids"])
    for example in tokenized_train_dataset
]

print("\nTraining token statistics:")

print(f"Shortest: {min(train_lengths)}")
print(f"Longest:  {max(train_lengths)}")

print(
    f"Average:  "
    f"{sum(train_lengths) / len(train_lengths):.2f}"
)


# ============================================================
# 8. Verify Tokenization
# ============================================================

example = tokenized_train_dataset[0]

decoded_text = tokenizer.decode(
    example["input_ids"],
    skip_special_tokens=True,
)

print("\n" + "=" * 70)
print("DECODED TRAINING EXAMPLE")
print("=" * 70)

print(decoded_text)


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
# 10. Data Collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 11. Training Configuration
# ============================================================

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

    seed=SEED,
)


# ============================================================
# 12. Create Trainer
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
# 13. Training Summary
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"Model:            {MODEL_NAME}")
print(f"Device:           {training_args.device}")

print(
    f"Training samples: {len(tokenized_train_dataset)}"
)

print(
    f"Eval samples:     {len(tokenized_eval_dataset)}"
)

print(f"Epochs:           {training_args.num_train_epochs}")

print(
    f"Batch size:       "
    f"{training_args.per_device_train_batch_size}"
)

print(f"Learning rate:    {training_args.learning_rate}")

print("=" * 70)


# ============================================================
# 14. Train
# ============================================================

print("\nStarting instruction fine-tuning...\n")

trainer.train()


# ============================================================
# 15. Save Final Model
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