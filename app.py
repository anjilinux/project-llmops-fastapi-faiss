from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="LLMOps RAG API - FINAL")

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "status": "running",
        "version": "FINAL_WORKING"
    }


@app.post("/chat")
def chat(query: Query):

    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        return {"error": "HF_API_KEY not set"}

    try:
        response = requests.post(
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": query.question,
                "parameters": {
                    "max_new_tokens": 100
                }
            },
            timeout=30
        )

        # 👇 VERY IMPORTANT (avoid crash)
        if response.status_code != 200:
            return {
                "error": "HF API failed",
                "status_code": response.status_code,
                "response": response.text
            }

        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            answer = result[0]["generated_text"]
        else:
            answer = str(result)

        return {
            "question": query.question,
            "answer": answer
        }

    except Exception as e:
        return {
            "error": str(e)
        }