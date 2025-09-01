import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
from huggingface_hub import snapshot_download

def download_model(model_name, hf_token, local_model_path):
    allow_patterns = [
        "*.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "*.model",
        "*.txt",
    ]
    ignore_patterns = [
        "*.bin", "*.pt", "*.h5", "*.onnx", "*.msgpack", "*.safetensors.index.json"
    ]

    print(f"Downloading model {model_name} to {local_model_path}")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
        token=hf_token,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns
    )
    print(f"Model {model_name} downloaded to {local_model_path}")

def load_mistral_model(model_name, hf_token, local_model_path):
    #check if model is already downloaded
    if not os.path.exists(local_model_path):
        download_model(model_name, hf_token, local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(local_model_path, token=hf_token, device_map="auto")

    mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return mistral_pipe

def generate_answer(query, vector_store, mistral_pipe):
    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join(docs)
    prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip() 