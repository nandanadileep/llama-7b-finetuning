import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_PATH = "adapters/final_adapter"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH
    )

    model.eval()
    return tokenizer, model

def generate(prompt, tokenizer, model, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer, model = load_model()

    prompt = """### Instruction:
Explain the difference between supervised and unsupervised learning

### Response:
"""

    output = generate(prompt, tokenizer, model)
    print(output)
