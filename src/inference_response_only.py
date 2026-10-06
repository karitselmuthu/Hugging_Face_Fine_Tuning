from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./models/smollm-horoscope-response-only-2000/final"

DEVICE = "mps"
SEED = 42


# ============================================================
# Verify Model Path
# ============================================================

model_path = Path(MODEL_PATH)

print(f"\nLoading model from: {model_path.resolve()}")

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path.resolve()}"
    )

# ============================================================
# 1. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully!")


# ============================================================
# 2. Load Model
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype="auto",
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# ============================================================
# 3. Build Prompt
# ============================================================

sign = "Aries"
category = "General"
date = "2026/09/30"

prompt = f"""### Instruction:
Generate a horoscope using the following information.

### Sign:
{sign}

### Category:
{category}

### Date:
{date}

### Horoscope:
"""


print("\n" + "=" * 70)
print("PROMPT")
print("=" * 70)

print(prompt)


# ============================================================
# 4. Tokenize Prompt
# ============================================================

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(DEVICE)


# ============================================================
# 5. Generate Horoscope
# ============================================================

torch.manual_seed(SEED)

print("\nGenerating horoscope...")

with torch.no_grad():

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


# ============================================================
# 6. Extract Only Generated Tokens
# ============================================================

prompt_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][prompt_length:]


# ============================================================
# 7. Decode Horoscope
# ============================================================

horoscope = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)


print("\n" + "=" * 70)
# print("PHASE 3 - GENERATED HOROSCOPE")
print("PHASE 4 - GENERATED HOROSCOPE (2000 EXAMPLES)")
print("=" * 70)

print(horoscope.strip())