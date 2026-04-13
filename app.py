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


# ✅ ONLY ONE CHAT ENDPOINT
@app.post("/chat")
def chat(query: Query):

    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        return {"error": "HF_API_KEY not set"}

    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/google/flan-t5-base",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"inputs": query.question},
            timeout=30
        )

        result = response.json()

        # ✅ SAFE parsing (no crash)
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            answer = result[0]["generated_text"]
        else:
            answer = str(result)

        return {
            "question": query.question,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}


# 🔍 OPTIONAL DEBUG
@app.get("/version")
def version():
    return {"version": "clean-fixed-router-code"}