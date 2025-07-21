from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline 
import torch

def load_mistral_model(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token, 
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32, 
        device_map="auto" if DEVICE.type == "cuda" else None
    )
    mistral_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return mistral_pipe

def generate_answer(query, vector_store, mistral_pipe):
    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join(docs)
    prompt = f"""Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    result = mistral_pipe(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip() 