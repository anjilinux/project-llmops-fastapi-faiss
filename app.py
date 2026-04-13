
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
# SAFE GLOBALS (optional FAISS later)
# -------------------
retriever = None
llm = None


def load_models():
    global retriever, llm

    if retriever is None:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )

            retriever = db.as_retriever(search_kwargs={"k": 2})

        except Exception as e:
            print("FAISS load failed:", e)
            retriever = "FAILED"

    if llm is None:
        try:
            from transformers import pipeline

            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-base"
            )

        except Exception as e:
            print("LLM load failed:", e)
            llm = "FAILED"


# -------------------
# ROUTES
# -------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "API stable (HF API mode)"
    }


# ✅ SINGLE CHAT ENDPOINT (HuggingFace API)
HF_API_KEY = os.getenv("HF_API_KEY")

@app.post("/chat")
def chat(query: Query):

    if not HF_API_KEY:
        return {
            "error": "HF_API_KEY not set in environment variables"
        }

    response = requests.post(
        "https://api-inference.huggingface.co/models/google/flan-t5-base",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": query.question}
    )

    result = response.json()

    return {
        "question": query.question,
        "answer": result[0]["generated_text"] if isinstance(result, list) else str(result)
    }


# 🔍 DEBUG (optional)
@app.get("/debug")
def debug():
    load_models()

    return {
        "retriever": str(retriever),
        "llm": str(llm)
    }