import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATA_PATH = "data/processed/train.jsonl"
OUTPUT_DIR = "outputs"

MAX_SEQ_LENGTH = 384
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
EPOCHS = 3


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = MAX_SEQ_LENGTH


dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,

    fp16=False,      # ❌ turn OFF AMP
    bf16=False,      # ❌ ensure bf16 is OFF
    max_grad_norm=0.0,

    optim="paged_adamw_8bit",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)



def formatting_func(example):
    return example["text"]


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
