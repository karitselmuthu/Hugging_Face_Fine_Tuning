import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"

FINE_TUNED_MODEL = "./models/smollm-horoscope-512/final"

PROMPT = "Today is going to be a wonderful day for Aries because"


def load_model(model_name):
    print(f"\nLoading: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )

    model.to("mps")
    model.eval()

    return tokenizer, model


def generate(tokenizer, model, prompt):
    torch.manual_seed(42)
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# --------------------------------------------------
# Base Model
# --------------------------------------------------

base_tokenizer, base_model = load_model(BASE_MODEL)

base_output = generate(
    base_tokenizer,
    base_model,
    PROMPT
)

print("\n================ BASE MODEL ================")
print(base_output)


# Free memory before loading second model
del base_model
del base_tokenizer

if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# --------------------------------------------------
# Fine-tuned Model
# --------------------------------------------------

fine_tokenizer, fine_model = load_model(FINE_TUNED_MODEL)

fine_output = generate(
    fine_tokenizer,
    fine_model,
    PROMPT
)

print("\n============= FINE-TUNED MODEL =============")
print(fine_output)