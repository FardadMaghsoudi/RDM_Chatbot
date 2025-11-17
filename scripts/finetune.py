import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
data_path = "../dmp_questions.jsonl"
output_dir_adapter = "../mistral-qlora-dmp-adapter"
output_dir_final = "../mistral-qlora-dmp-final"
prompt_template = """You are Dizzi — a friendly and knowledgeable assistant for Research Data Management (RDM) at TU Delft. You are trained on TU Delft's official RDM guidelines, and you may also be provided with additional context below. Use markdown formatting for clarity, and keep responses concise yet informative.

        Your task is to:
        1. Start with general TU Delft RDM principles applicable to all researchers.
        2. Then ask the user which faculty, department, or role they belong to (e.g., PhD student in Aerospace, Data Steward in Applied Sciences).
        3. Once the user responds, tailor your advice using the provided context and your training to match their specific domain or role.
        4. Where available, provide real and verifiable links to TU Delft or trusted sources for further reference.

        If a question is outside your knowledge or the provided context, say so clearly and do not make assumptions.

        Use the following context for accurate answers:
        Context:
        {context}

        User Question:
        {query}

        Answer:
        {answer}"""

def formatting_func(batch: dict) -> str:
    return prompt_template.format(
        context=batch["context"],
        query=batch["query"],
        answer=batch["answer"]
    )

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
tokenizer.padding_side = "right"

# 2) Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

import warnings

# NOTE: in a future transformers update, you will need to pass use_reentrant=False as an argument to model.gradient_checkpointing_enable(). Until then, you can safely keep this annoying warning out.
warnings.filterwarnings(
    "ignore",
    message=".*use_reentrant parameter should be passed explicitly.*",
)

# 3) Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.config.use_cache = False  # important when using gradient checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

# 4) LoRA config – lightweight for 8 GB
lora_config = LoraConfig(
    r=8,                  # lower rank for less memory
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 5) DMP questions dataset
dataset = load_dataset("json", data_files={"train": data_path}, split="train")  # start small

# 6) Training arguments tuned for 8 GB
training_args = SFTConfig(
    output_dir=output_dir_adapter,
    per_device_train_batch_size=1,      # keep at 1 on 8GB
    gradient_accumulation_steps=1,      # effective batch size 1
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    bf16=True,                          # A40 supports bf16; use fp16=False
    fp16=False,
    optim="paged_adamw_32bit",          # bitsandbytes optimizer
    max_grad_norm=0.3,
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(output_dir_adapter)
tokenizer.save_pretrained(output_dir_adapter)

