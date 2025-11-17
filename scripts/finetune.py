import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# 1) 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3) Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.config.use_cache = False  # important when using gradient checkpointing

# 4) LoRA config – lightweight for 8 GB
lora_config = LoraConfig(
    r=8,                  # lower rank for less memory
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5) Example dataset – replace with your own
dataset = load_dataset("tatsu-lab/alpaca", split="train[:2000]")  # start small

def format_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return {"text": prompt + output}

dataset = dataset.map(format_example)

# 6) Training arguments tuned for 8 GB
training_args = TrainingArguments(
    output_dir="./mistral7b-qlora-a40-8g",
    per_device_train_batch_size=1,      # keep at 1 on 8GB
    gradient_accumulation_steps=8,      # effective batch size 8
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,                          # A40 supports bf16; use fp16=False
    fp16=False,
    optim="paged_adamw_32bit",          # bitsandbytes optimizer
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=1024,                # shorter context to reduce memory
    packing=False,                      # you can try True later if data is short
)

trainer.train()
trainer.save_model("./mistral7b-qlora-a40-8g")
tokenizer.save_pretrained("./mistral7b-qlora-a40-8g")
