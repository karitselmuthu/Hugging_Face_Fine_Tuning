from pathlib import Path

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "karthiksagarn/astro_horoscope"

OUTPUT_DIR = Path("./data/processed")

TRAIN_OUTPUT = OUTPUT_DIR / "train"
TEST_OUTPUT = OUTPUT_DIR / "test"

TEST_SIZE = 0.1
SEED = 42


# ============================================================
# 1. Format Training Example
# ============================================================

def format_example(example):
    """
    Convert a horoscope dataset row into an instruction-style
    training example.

    Original:
        sign
        category
        date
        horoscope

    Result:
        Instruction + Sign + Category + Date + Horoscope
    """

    sign = example["sign"].strip().title()
    category = example["category"].strip().title()
    date = example["date"].strip()
    horoscope = example["horoscope"].strip()

    text = f"""### Instruction:
Generate a horoscope using the following information.

### Sign:
{sign}

### Category:
{category}

### Date:
{date}

### Horoscope:
{horoscope}"""

    return {
        "text": text
    }


# ============================================================
# 2. Load Original Dataset
# ============================================================

print("\nLoading original dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
)

print("Dataset loaded successfully!")
print(f"Total examples: {len(dataset):,}")

print("\nOriginal columns:")
print(dataset.column_names)


# ============================================================
# 3. Inspect Original Example
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL EXAMPLE")
print("=" * 70)

print(dataset[0])


# ============================================================
# 4. Format Dataset
# ============================================================

print("\nFormatting dataset...")

formatted_dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
)

print("Dataset formatting completed!")

print("\nFormatted columns:")
print(formatted_dataset.column_names)


# ============================================================
# 5. Inspect Formatted Examples
# ============================================================

print("\n" + "=" * 70)
print("FORMATTED EXAMPLES")
print("=" * 70)

for index in range(min(5, len(formatted_dataset))):

    print(f"\n--- Example {index + 1} ---\n")

    print(
        formatted_dataset[index]["text"]
    )


# ============================================================
# 6. Train/Test Split
# ============================================================

print("\nCreating train/test split...")

dataset_split = formatted_dataset.train_test_split(
    test_size=TEST_SIZE,
    seed=SEED,
)

print(
    f"Training examples: "
    f"{len(dataset_split['train']):,}"
)

print(
    f"Evaluation examples: "
    f"{len(dataset_split['test']):,}"
)


# ============================================================
# 7. Create Output Directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 8. Save Prepared Dataset
# ============================================================

print("\nSaving prepared dataset...")

dataset_split["train"].save_to_disk(
    str(TRAIN_OUTPUT)
)

dataset_split["test"].save_to_disk(
    str(TEST_OUTPUT)
)

print("\nDataset saved successfully!")

print(f"Train: {TRAIN_OUTPUT}")
print(f"Test:  {TEST_OUTPUT}")


# ============================================================
# 9. Final Summary
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(
    f"""
Original examples : {len(dataset):,}
Training examples : {len(dataset_split['train']):,}
Test examples     : {len(dataset_split['test']):,}

Output directory:
{OUTPUT_DIR}
"""
)