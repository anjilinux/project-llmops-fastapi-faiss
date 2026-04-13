from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="LLMOps RAG API - Stable")

# -------------------
# REQUEST MODEL
# -------------------
class Query(BaseModel):
    question: str


# -------------------
# ROUTES
# -------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "API stable (HF router mode)"
    }


# ✅ SINGLE CHAT ENDPOINT (FIXED)
@app.post("/chat")
def chat(query: Query):

    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        return {"error": "HF_API_KEY not set"}

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/google/flan-t5-base",
        headers={
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": query.question
        }
    )

    result = response.json()

    return {
        "question": query.question,
        "answer": result[0]["generated_text"] if isinstance(result, list) else str(result)
    }




