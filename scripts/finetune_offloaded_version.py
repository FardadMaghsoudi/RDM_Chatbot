import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling 
)
import warnings
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
import os
import gc
from dotenv import load_dotenv

# --- 1. Setup ---
load_dotenv()
token = os.getenv("HF_TOKEN")
gc.collect()
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
data_path = "dmp_questions.jsonl"
output_dir_ft = "mistral-qlora-dmp"
offload_folder = "offload_weights"
os.makedirs(offload_folder, exist_ok=True)

# --- 2. Load Tokenizer & Model ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    llm_int8_enable_fp32_cpu_offload=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",            
    offload_folder=offload_folder, 
    use_cache=False,
    token=token
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

# --- 3. THE FIX: Manual Masking Function ---
# Instead of searching for strings, we tokenize the prompt separately 
# to know EXACTLY how long it is.
def tokenize_and_mask(batch):
    # 1. The Full System Prompt (Restored)
    # We are restoring the full instructions since we raised the limit to 1024.
    prompt_template = """<s> Your name is Dizzy. You are a friendly knowledge assistant chatbot designed by Madalina Fron and Fardad Maghsoudi to support TU Delft data managers, data stewards, professors, researchers, and students. You answer questions related to data management, data engineering, data governance, data policy, data security, and research data management, in alignment with TU Delft rules, policies, and regulations.

You are trained on TU Delft’s official Research Data Management (RDM) guidelines. Use Markdown formatting for clarity, and provide responses that are concise, accurate, and informative.

When answering questions, prioritize information sources in the following order:
1. TU Delft official resources (PDF files and webpages)
2. Relevant and authoritative EU documents
3. Your general training and background knowledge

Tailor your responses by adapting advice to the user’s faculty, role, or discipline. Where applicable, provide real, verifiable links to official TU Delft pages.

    Context:
    {context}

    User Question:
    {query}

    Answer:
    """
    
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    # NEW LIMIT: 1024 tokens (The absolute max for 8GB VRAM)
    MAX_TOTAL_LENGTH = 2048
    
    # We calculate how much space is left for context dynamically
    # System Prompt (~250 tokens) + User Query (~50) + Answer (~300) = ~600 used.
    # This leaves ~400 tokens for Context.
    SAFE_CONTEXT_LIMIT = 400 
    
    for i in range(len(batch['query'])):
        # --- 1. Smart Context Truncation ---
        # We truncate the context first to ensure the prompt backbone fits.
        context_ids = tokenizer.encode(batch['context'][i], add_special_tokens=False)
        if len(context_ids) > SAFE_CONTEXT_LIMIT:
            context_ids = context_ids[:SAFE_CONTEXT_LIMIT]
        truncated_context = tokenizer.decode(context_ids)
        
        # --- 2. Format Prompt ---
        prompt_text = prompt_template.format(
            context=truncated_context,
            query=batch['query'][i]
        )
        
        # Ensure answer is a string and handle potential empty values
        answer_text = str(batch['answer'][i])
        if not answer_text or answer_text.strip() == "":
            answer_text = " [No Answer Provided]"
            
        full_text = prompt_text + answer_text + "</s>"
        
        # --- 3. Tokenize ---
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        
        # --- 4. DIAGNOSTIC (First item only) ---
        if i == 0:
            print("\n--- DIAGNOSTIC (Target: 1024 Tokens) ---")
            print(f"Prompt Length: {len(prompt_ids)} tokens")
            print(f"Full Length:   {len(full_ids)} tokens")
            print(f"Space for Answer: {MAX_TOTAL_LENGTH - len(prompt_ids)} tokens")
            if len(full_ids) > MAX_TOTAL_LENGTH:
                print(f"WARNING: Still truncating! Over by {len(full_ids) - MAX_TOTAL_LENGTH} tokens.")
            else:
                print("SUCCESS: Full text fits within 1024 limit.")
        
        # --- 5. Apply New Hard Limit (1024) ---
        if len(full_ids) > MAX_TOTAL_LENGTH:
            full_ids = full_ids[:MAX_TOTAL_LENGTH]
        
        # --- 6. Create Labels ---
        labels = full_ids.copy()
        
        # Mask the prompt part so we only train on the Answer
        # We take the min() to ensure we don't go out of bounds if truncation happened
        mask_len = min(len(prompt_ids), len(full_ids))
        for j in range(mask_len):
            labels[j] = -100
            
        # --- 7. Safety: Ensure we have SOMETHING to train on ---
        # If the prompt was so huge it pushed the answer off the cliff, force train the end.
        if (sum(1 for x in labels if x != -100) == 0):
            if i == 0: print("CRITICAL: Answer was pushed out. Forcing last 20 tokens.")
            # Unmask the last 20 tokens so the model doesn't crash with 0 loss
            start_safe = max(0, len(labels) - 20)
            for k in range(start_safe, len(labels)):
                labels[k] = full_ids[k]

        model_inputs["input_ids"].append(full_ids)
        model_inputs["attention_mask"].append([1] * len(full_ids))
        model_inputs["labels"].append(labels)
        
    return model_inputs

print("Processing dataset with manual masking...")
dataset = load_dataset("json", data_files={"train": data_path}, split="train")

# Run the masking function
tokenized_dataset = dataset.map(tokenize_and_mask, batched=True, remove_columns=dataset.column_names)

# We now need to pad the batch dynamically, but we already have the labels!
# So we use the STANDARD DataCollator (no custom searching logic needed)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("\n--- CHECKING MASKS (First 3 Examples) ---")
for i in range(min(15, len(tokenized_dataset))):
    labels = tokenized_dataset[i]["labels"]
    trainable_count = sum(1 for l in labels if l != -100)
    print(f"Example {i}: {trainable_count} trainable tokens.")
    
    if trainable_count == 0:
        print(f"WARNING: Example {i} has no tokens to learn! Check this data entry.")
print("------------------------------------------\n")

# --- 5. Training ---
training_args = TrainingArguments(
    output_dir=output_dir_ft,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    bf16=True,  
    fp16=False,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    report_to="none", 
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=collator, # Standard collator pads the lists we made above
)

print("Starting training...")
trainer.train()

trainer.save_model(output_dir_ft)
print("Training Complete.")
