import os
import torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, 
    Mistral3ForConditionalGeneration,
)
import warnings
from peft import LoraConfig, get_peft_model, TaskType
import gc
from tqdm import tqdm
from dotenv import load_dotenv
import wandb
import evaluate
from torch.utils.data import DataLoader
from mistral_model import build_prompt
from config import QNA_PATH
from pathlib import Path

# --- 1. Setup ---
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Experiment Pipeline")
    parser.add_argument("--model_name", type=str, default="mistralai/Ministral-3-3B-Instruct-2512-BF16", help="HuggingFace model ID")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--project_name", type=str, default="Dizzy", help="Wandb project name")
    return parser.parse_args()

args = parse_args()
load_dotenv()
token = os.getenv("HF_TOKEN")
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

run_name = f"{args.model_name.split('/')[-1]}-full-r{args.lora_r}-test{args.test_size}"
output_dir_ft = f"./results/{run_name}"
offload_folder = "offload_weights"
os.makedirs(offload_folder, exist_ok=True)

wandb.init(
    project=args.project_name,
    name=run_name,      
    config={
        "model_name": args.model_name,
        "dataset": "DMP-Policy-Questions",
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
)

# --- 2. Load Tokenizer & Model ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
if "Ministral-3" in args.model_name:
    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token,
    )

lora_config = LoraConfig(
    r=args.lora_r, 
    lora_alpha=args.lora_r * 2,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

def tokenize_and_mask(batch):
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
    MAX_TOTAL_LENGTH = 2048
    
    for i in range(len(batch['query'])):
        query = batch['query'][i]
        context = batch['context'][i]
        answer = str(batch['answer'][i]) if batch['answer'][i] else " [No Answer Provided]"
        
        prompt = build_prompt(query, context)
        answer_str = answer + "</s>"

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)        
        answer_ids = tokenizer.encode(answer_str, add_special_tokens=False)
        
        total_len = len(prompt_ids) + len(answer_ids)

        if total_len > MAX_TOTAL_LENGTH:
            overflow = total_len - MAX_TOTAL_LENGTH
            prompt_ids = prompt_ids[:len(prompt_ids) - overflow]

        input_ids = prompt_ids + answer_ids

        # Safety Truncate
        if len(input_ids) > MAX_TOTAL_LENGTH:
            input_ids = input_ids[:MAX_TOTAL_LENGTH]

        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        padding_len = MAX_TOTAL_LENGTH - len(input_ids)
        
        if padding_len > 0:
            attention_mask = [1] * len(input_ids) + [0] * padding_len
            input_ids += [tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len
        else:
            attention_mask = [1] * MAX_TOTAL_LENGTH

        model_inputs["input_ids"].append(input_ids[:MAX_TOTAL_LENGTH])
        model_inputs["attention_mask"].append(attention_mask[:MAX_TOTAL_LENGTH])
        model_inputs["labels"].append(labels[:MAX_TOTAL_LENGTH])
    
    return model_inputs

print("Processing dataset with manual masking...")
jsonl_files = sorted(str(path) for path in Path(QNA_PATH).glob("*.jsonl"))

dataset = load_dataset(
    "json",
    data_files={"train": jsonl_files},
    split="train",
)

split_dataset = dataset.train_test_split(test_size=args.test_size, seed=42)

# Run the masking function
tokenized_dataset = split_dataset.map(tokenize_and_mask, batched=True, remove_columns=dataset.column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Metric Computation for Evaluation ---
def preprocess_logits_for_metrics(logits, labels):
    """
    Reduces the logits tensor from (batch_size, seq_len, vocab_size)
    to (batch_size, seq_len) by taking the argmax.
    This is strictly required to prevent VRAM OOM on a 16GB GPU.
    """
    if isinstance(logits, tuple):
        # Depending on the model, logits might be returned as a tuple
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    """
    Computes token-level accuracy, accounting for the causal LM shift.
    """
    preds, labels = eval_preds

    # Causal LM shift: prediction at position i corresponds to label at position i+1
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Create a boolean mask to ignore all -100 padding/prompt tokens
    mask = labels != -100

    # Check where predictions match the true labels
    correct_predictions = (preds == labels)

    # Sum the correct predictions only where the mask is True, then divide by total valid tokens
    total_correct = correct_predictions[mask].sum()
    total_valid_tokens = mask.sum()

    accuracy = total_correct / total_valid_tokens

    return {"token_accuracy": float(accuracy)}

# --- 5. Training ---
training_args = TrainingArguments(
    output_dir=output_dir_ft,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=8, 
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=args.epochs,
    logging_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    bf16=True,  
    fp16=False,
    optim="adamw_torch",
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    data_collator=collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print("Starting training...")
trainer.train()

import shutil

# This removes all intermediate checkpoint folders to save space
for item in os.listdir(output_dir_ft):
    item_path = os.path.join(output_dir_ft, item)
    if os.path.isdir(item_path) and item.startswith("checkpoint-"):
        print(f"Deleting checkpoint: {item_path}")
        shutil.rmtree(item_path)

print("Running generative evaluation on full test set...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

tokenizer.padding_side = "left"
model.eval()

def collate_fn_generate(batch):
    prompts = [build_prompt(item['query'], item['context']) for item in batch]
    references = [item['answer'] for item in batch]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    return inputs, references

# Use batch size of 1 to prevent Out Of Memory (OOM) errors during generation
test_dataloader = DataLoader(split_dataset["test"], batch_size=1, collate_fn=collate_fn_generate)

generated_texts = []
reference_texts = []

for batch_inputs, batch_refs in tqdm(test_dataloader, desc="Generating Answers"):
    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True # Faster generation
        )

    input_lengths = batch_inputs["input_ids"].shape[1]
    for i, output in enumerate(outputs):
        gen_text = tokenizer.decode(output[input_lengths:], skip_special_tokens=True)
        generated_texts.append(gen_text.strip())
        reference_texts.append(batch_refs[i].strip())

# Compute all metrics
rouge_results = rouge.compute(predictions=generated_texts, references=reference_texts)
bertscore_results = bertscore.compute(predictions=generated_texts, references=reference_texts, lang="en")
bleu_results = bleu.compute(predictions=generated_texts, references=[[r] for r in reference_texts])
meteor_results = meteor.compute(predictions=generated_texts, references=reference_texts)

final_metrics = {
    "eval_gen/rouge1": rouge_results["rouge1"],
    "eval_gen/rougeL": rouge_results["rougeL"],
    "eval_gen/bertscore_f1": np.mean(bertscore_results["f1"]),
    "eval_gen/bleu": bleu_results["bleu"],
    "eval_gen/meteor": meteor_results["meteor"]
}

wandb.run.summary.update(final_metrics)

print("Final Generation Metrics:", final_metrics)

trainer.save_model(output_dir_ft)
wandb.finish()

print("Experiment Complete.")

