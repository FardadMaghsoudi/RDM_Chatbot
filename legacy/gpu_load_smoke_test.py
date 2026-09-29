import torch
import os
import gc
from transformers import Mistral3ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/mistral-7b-instruct-v0.3"

print("CUDA device:", torch.cuda.get_device_name(0))
print("Total VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)

gc.collect()
torch.cuda.empty_cache()

offload_folder = "offload_weights"
os.makedirs(offload_folder, exist_ok=True)

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nLoading quantized model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    offload_folder=offload_folder,
    use_cache=False,
)

# Important for training scenarios
model.config.use_cache = False
model.config.tie_word_embeddings = False  # <-- silences your warning

print("\nModel loaded successfully!")

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"\nGPU memory allocated: {allocated:.2f} GB")
print(f"GPU memory reserved : {reserved:.2f} GB")

# Quick forward test
print("\nRunning small forward pass...")
inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

print("Forward pass successful.")
