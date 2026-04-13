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
# HOME
# -------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "API stable (HF router FINAL)"
    }


# -------------------
# VERSION (DEBUG)
# -------------------
@app.get("/version")
def version():
    return {"version": "FINAL_WORKING_V2"}


# -------------------
# CHAT ENDPOINT (FINAL FIX)
# -------------------
@app.post("/chat")
def chat(query: Query):

    HF_API_KEY = os.getenv("HF_API_KEY")

    if not HF_API_KEY:
        return {"error": "HF_API_KEY not set"}

    try:
        response = requests.post(
            # ✅ FINAL WORKING URL
            "https://router.huggingface.co/hf-inference/models/google/flan-t5-base?provider=hf-inference",
            headers={
                "Authorization": f"Bearer {HF_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": query.question
            },
            timeout=30
        )

        # ✅ HANDLE HTTP ERRORS
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

        # ✅ SAFE RESPONSE EXTRACTION
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
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