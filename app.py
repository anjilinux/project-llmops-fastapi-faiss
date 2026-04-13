from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLMOps RAG API - Test Mode")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "FastAPI is deployed successfully on Render"
    }

@app.post("/chat")
def chat(query: Query):
    return {
        "question": query.question,
        "answer": "Render deployment is working (FAISS + LLM temporarily disabled for test)"
    }