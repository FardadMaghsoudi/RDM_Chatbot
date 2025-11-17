from functools import lru_cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---- CONFIG ----
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "../mistral-qlora-dmp-adapter/checkpoint-22"  # folder with adapter_model.safetensors, etc.


def _build_mistral_model(
    base_model_name: str = BASE_MODEL,
    adapter_dir: str = ADAPTER_DIR,
):
    """
    Load base Mistral in 4-bit and attach the LoRA adapter for Dizzi.
    Returns (model, tokenizer).
    """
    # Tokenizer – use the one saved with the adapter (or fall back to base model)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit quantization for 8 GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    # Base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Attach LoRA adapter (Dizzi fine-tune)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

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


def build_pipe(model_and_tokenizer):
    """
    Build a callable that matches your old interface:
    pipe(prompt) -> [{"generated_text": full_text}]
    """
    model, tokenizer = model_and_tokenizer

    def _pipe(prompt: str):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": full_text}]

    return _pipe

def generate_answer(query, vector_store, mistral_pipe):
    docs = vector_store.similarity_search(query, k=2)
    chunks = [(getattr(d, "page_content", d) or "").strip() for d in docs]
    context = "\n\n".join(chunks)
    prompt = f"""
        You are Dizzi — a friendly and knowledgeable assistant for Research Data Management (RDM) at TU Delft. You are trained on TU Delft's official RDM guidelines, and you may also be provided with additional context below. Use markdown formatting for clarity, and keep responses concise yet informative.

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
        """
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip() 
