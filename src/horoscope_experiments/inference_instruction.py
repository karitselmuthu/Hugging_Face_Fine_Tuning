import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "./models/smollm-horoscope-instruction-512/final"

DEVICE = "mps"


# ============================================================
# 1. Load Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

print("Tokenizer loaded successfully!")


# ============================================================
# 2. Load Fine-Tuned Model
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
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
).to(DEVICE)


# ============================================================
# 5. Generate
# ============================================================

torch.manual_seed(42)

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
# 6. Decode Full Output
# ============================================================

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)


print("\n" + "=" * 70)
print("FULL MODEL OUTPUT")
print("=" * 70)

print(generated_text)


# ============================================================
# 7. Decode Only Generated Horoscope
# ============================================================

prompt_length = inputs["input_ids"].shape[1]

generated_tokens = outputs[0][prompt_length:]

horoscope = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)


print("\n" + "=" * 70)
print("GENERATED HOROSCOPE")
print("=" * 70)

print(horoscope.strip())