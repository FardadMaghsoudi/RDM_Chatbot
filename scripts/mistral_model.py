from functools import lru_cache
from llama_cpp import Llama


def _build_mistral_model(model_name, quant_model_name, hf_token):
    return Llama.from_pretrained( 
        repo_id=model_name,
        filename=quant_model_name,
        n_ctx=4096,
        n_batch=256,
        n_threads=None,
        n_gpu_layers=28,
        main_gpu=0,
        tensor_split=None,
        hf_token=hf_token,
    )

@lru_cache(maxsize=1)
def get_mistral_model(model_name, quant_model_name, hf_token=None):
    # Cached once per process; subsequent calls reuse the same model instance
    return _build_mistral_model(model_name, quant_model_name, hf_token)

def build_pipe(model: Llama):
    def _pipe(prompt: str):
        out = model.create_completion(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.2,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )
        text = out["choices"][0]["text"]
        return [{"generated_text": prompt + text}]
    return _pipe

def generate_answer(query, vector_store, mistral_pipe):
    docs = vector_store.similarity_search(query, k=2)
    chunks = [(getattr(d, "page_content", d) or "").strip() for d in docs]
    context = "\n\n".join(chunks)
    prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip() 
