from datasets import load_dataset
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
DATASET_NAME = "karthiksagarn/astro_horoscope"

OUTPUT_DIR = "./models/smollm-horoscope-512"
FINAL_MODEL_PATH = f"{OUTPUT_DIR}/final"

MAX_LENGTH = 256


# ============================================================
# 1. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully!")
print(f"Vocabulary size: {len(tokenizer):,}")


# ============================================================
# 2. Load Dataset
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
)

print("Dataset loaded successfully!")
print(f"Total examples: {len(dataset):,}")

print("\nColumns:")
print(dataset.column_names)


# ============================================================
# 3. Train/Test Split
# ============================================================

print("\nCreating train/test split...")

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42,
)

print(f"Training examples:   {len(dataset['train']):,}")
print(f"Evaluation examples: {len(dataset['test']):,}")


# ============================================================
# 4. Tokenization Function
# ============================================================

def tokenize_function(examples):
    texts = [
        text + tokenizer.eos_token
        for text in examples["horoscope"]
    ]

    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ============================================================
# 5. Tokenize Dataset
# ============================================================

print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print("Tokenization completed!")


# ============================================================
# 6. Inspect Tokenized Example
# ============================================================

example = tokenized_dataset["train"][0]

print("\nTokenized example:")
print(example)


# ============================================================
# 7. Decode Tokens Back to Text
# ============================================================

decoded_text = tokenizer.decode(
    example["input_ids"],
    skip_special_tokens=True,
)

print("\nDecoded text:")
print(decoded_text)


# ============================================================
# 8. Token Statistics
# ============================================================

print("\nToken statistics:")

lengths = [
    len(example["input_ids"])
    for example in tokenized_dataset["train"]
]

print(f"Shortest example: {min(lengths)} tokens")
print(f"Longest example:  {max(lengths)} tokens")
print(f"Average length:   {sum(lengths) / len(lengths):.2f} tokens")


print("\nDataset preparation completed successfully!")

# ============================================================
# # 9. Create Training/Evaluation Subsets
# ============================================================

print("\nPreparing small training dataset...")

small_train_dataset = (
    tokenized_dataset["train"]
    .shuffle(seed=42)
    .select(range(min(512, len(tokenized_dataset["train"]))))
)

small_eval_dataset = (
    tokenized_dataset["test"]
    .shuffle(seed=42)
    .select(range(min(64, len(tokenized_dataset["test"]))))
)

print(f"Training examples: {len(small_train_dataset)}")
print(f"Evaluation examples: {len(small_eval_dataset)}")


# ============================================================
# 10. Load Model
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
# 11. Data Collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 12. Training Configuration
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
)


# ============================================================
# 13. Create Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)


# ============================================================
# 14. Start Training
# ============================================================

print("\nTrainer device:")
print(training_args.device)

print("\nStarting smoke-test training...")
print(f"This will train on only {len(small_train_dataset)} examples.\n")

trainer.train()

# ============================================================
# 15. Save Model
# ============================================================


print("\nSaving model...")

trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)

print("\nTraining completed successfully!")
print(f"Model saved to: {FINAL_MODEL_PATH}")
