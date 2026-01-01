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


MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DATA_PATH = "data/processed/train.jsonl"
OUTPUT_DIR = "outputs"

MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 3


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token


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


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
