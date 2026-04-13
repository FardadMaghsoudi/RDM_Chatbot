from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, Mistral3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model
import re
import os
from typing import Tuple
from dotenv import load_dotenv
from config import WEB_URLS
 
# Load environment variables from .env file
load_dotenv()

_forbidden_patterns_env = os.getenv("FORBIDDEN_INPUT_PATTERNS")
FORBIDDEN_INPUT_PATTERNS = [p.strip() for p in _forbidden_patterns_env.split("||")]

_disclosure_patterns_env = os.getenv("DISCLOSURE_OUTPUT_PATTERNS")
DISCLOSURE_OUTPUT_PATTERNS = [p.strip() for p in _disclosure_patterns_env.split("||")]

SAFE_RESPONSE = os.getenv("SAFE_RESPONSE")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT").replace("\\n", "\n")

URL_REF = "\n".join(
    f"- [{label}]({url})" for label, url in WEB_URLS.items()
)

# ---- CONFIG ----
BASE_MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
ADAPTER_DIR = "results/Ministral-3-3B-Instruct-2512-BF16-full-r16-test0.1"  # folder with adapter_model.safetensors, etc.

def _build_mistral_model(
    base_model_name: str = BASE_MODEL,
    adapter_dir: str = ADAPTER_DIR,
):
    """
    Load base Mistral in 4-bit and attach the LoRA adapter for Dizzi.
    Returns (model, tokenizer).
    """
    # Tokenizer – use the one from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization for 8 GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )
    model = model.merge_and_unload()
    model = torch.compile(model)
    
#    model.print_trainable_parameters()

    model.eval()
    return model, tokenizer


@lru_cache(maxsize=1)
def get_mistral_model(
    base_model_name: str = BASE_MODEL,
    adapter_dir: str = ADAPTER_DIR,
):
    """
    Cached once per process; returns (model, tokenizer).
    """
    return _build_mistral_model(base_model_name, adapter_dir)


def validate_input(query: str) -> Tuple[bool, str]:
    """
    Validate user input for suspicious or malicious patterns.

    Args:
        query: User's input query

    Returns:
        Tuple[is_safe, response]:
            - is_safe (bool): True if query is safe, False if suspicious
            - response (str): Safe response if query is suspicious, empty string if safe
    """
    # Check for forbidden patterns
    for pattern in FORBIDDEN_INPUT_PATTERNS:
        if re.search(pattern, query):
            return False, SAFE_RESPONSE

    # Query is safe
    return True, ""


def validate_output(response: str) -> Tuple[bool, str]:
    """
    Validate model output to prevent prompt disclosure.

    Args:
        response: Generated response from the model

    Returns:
        Tuple[is_safe, sanitized_response]:
            - is_safe (bool): True if response is safe, False if disclosure detected
            - sanitized_response (str): Safe response or original if safe
    """
    # Check for disclosure patterns
    for pattern in DISCLOSURE_OUTPUT_PATTERNS:
        if re.search(pattern, response):
            return False, SAFE_RESPONSE

    # Response is safe
    return True, response

def build_prompt(query, context, url_ref=""):
    system = SYSTEM_PROMPT.format(url_ref=url_ref)
    return f"<s>[INST] {system}\n\nContext:\n{context}\n\nQuestion:\n{query} [/INST] "

def generate_answer(query, vector_store, model_and_tokenizer):
    is_safe, safe_response = validate_input(query)
    if not is_safe:
        print(f"[SECURITY] Suspicious request detected: {query[:100]}...")
        return safe_response

    model, tokenizer = model_and_tokenizer
    docs = vector_store.similarity_search(query, k=5)
    chunks = [d if isinstance(d, str) else d.page_content for d in docs]
    context = "\n---\n".join(chunks)

    final_prompt = build_prompt(query, context, URL_REF)
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    raw_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    is_safe, final_answer = validate_output(raw_answer.strip())
    if not is_safe:
        print(f"[SECURITY] Prompt disclosure detected in output. Blocking response.")
        return final_answer

    return final_answer
