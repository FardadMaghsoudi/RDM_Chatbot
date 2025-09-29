import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPTQConfig
from huggingface_hub import snapshot_download
import torch
from llama_cpp import Llama

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

def load_mistral_model(model_name, quant_model_name, hf_token):
    mistral_pipe = Llama.from_pretrained( 
        repo_id=model_name,
        filename=quant_model_name,
        n_ctx=8192,
        n_batch=512,
        n_threads=None,
        n_gpu_layers=999,
        main_gpu=0,
        tensor_split=None,
    )

    def _pipe(prompt: str):
        out = mistral_pipe.create_completion(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    repeat_penalty=1.1,
                )
        text = out["choices"][0]["text"]
        return [{"generated_text": prompt + text}]

    return _pipe

def generate_answer(query, vector_store, mistral_pipe):
    docs = vector_store.similarity_search(query, k=4)
    # chunks = [(getattr(d, "page_content", d) or "").strip() for d in docs]
    context = "\n\n".join(docs)
    prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip() 
