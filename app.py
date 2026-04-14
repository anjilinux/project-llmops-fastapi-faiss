from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="LLMOps RAG API - FINAL")

# -------------------
# REQUEST MODEL
# -------------------
class Query(BaseModel):
    question: str


# -------------------
# HOME
# -------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "version": "FINAL_STABLE_OK"
    }


# -------------------
# CHAT (100% SAFE)
# -------------------
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

        # ✅ HANDLE NON-200 RESPONSE
        if response.status_code != 200:
            return {
                "error": "HF API failed",
                "status_code": response.status_code,
                "response": response.text
            }

        # ✅ SAFE JSON PARSE
        try:
            result = response.json()
        except Exception:
            return {
                "error": "Invalid JSON from HF",
                "raw_response": response.text
            }

        # ✅ SAFE OUTPUT
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
            "error": "Internal error",
            "details": str(e)
        }
    

    