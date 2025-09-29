from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from mistral_model import get_mistral_model, build_pipe, generate_answer
import config
from data_preprocessing import preprocess_data

# Preprocess data and create vector store
combined_chunks, vector_store = preprocess_data()

# Load Mistral model
print("Loading Mistral model...")
mistral_model = get_mistral_model(config.MODEL_NAME, config.QUANT_MODEL_NAME, config.HF_TOKEN)
mistral_pipe = build_pipe(mistral_model)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    answer = generate_answer(query.question, vector_store, mistral_pipe)
    return {"response": answer}

if __name__ == "__main__":
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = generate_answer(question, vector_store, mistral_pipe)
        print("\nAnswer:", answer)
        print("-" * 50) 
