from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLMOps RAG API - Stable")

class Query(BaseModel):
    question: str

# -------------------
# SAFE GLOBALS
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
        "message": "API stable (no crash)"
    }

@app.post("/chat")
def chat(query: Query):

    # ⚠️ IMPORTANT: DO NOT LOAD MODELS HERE
    # load_models()

    return {
        "question": query.question,
        "answer": "✅ API working (models disabled to prevent crash)"
    }


# 🔍 DEBUG ENDPOINT (use this instead of chat for testing models)
@app.get("/debug")
def debug():
    load_models()

    return {
        "retriever": str(retriever),
        "llm": str(llm)
    }





