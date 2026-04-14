from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="LLMOps RAG API - Stable")

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "API stable (HF direct mode)"
    }


@app.get("/version")
def version():
    return {"version": "FINAL_WORKING_V3"}


@app.post("/chat")
def chat(query: Query):

    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        return {"error": "HF_API_KEY not set"}

    try:
        response = requests.post(
            # ✅ DIRECT HF ENDPOINT
            "https://api-inference.huggingface.co/models/google/flan-t5-base",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}"
            },
            json={
                "inputs": query.question
            },
            timeout=30
        )

        if response.status_code != 200:
            return {
                "error": "HF API failed",
                "status_code": response.status_code,
                "response": response.text
            }

        try:
            result = response.json()
        except Exception:
            return {
                "error": "Invalid JSON",
                "raw": response.text
            }

        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get("generated_text", str(result))
        else:
            answer = str(result)

        return {
            "question": query.question,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}