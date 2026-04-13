from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLMOps RAG API")

# -------------------
# REQUEST MODEL
# -------------------
class Query(BaseModel):
    question: str

# -------------------
# GLOBALS (lazy load)
# -------------------
retriever = None
llm = None

# -------------------
# LOAD MODELS SAFELY
# -------------------
def load_models():
    global retriever, llm

    # -------- FAISS --------
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
            print("✅ FAISS loaded")

        except Exception as e:
            print("❌ FAISS load failed:", e)
            retriever = None

    # -------- LLM --------
    if llm is None:
        try:
            from transformers import pipeline

            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small"   # ✅ lighter model
            )

            print("✅ LLM loaded")

        except Exception as e:
            print("❌ LLM load failed:", e)
            llm = None


# -------------------
# ROUTES
# -------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "API live (safe mode)"
    }


# 🔥 DEBUG CHAT (NO HEAVY PROCESSING)
@app.post("/chat")
def chat(query: Query):

    load_models()

    return {
        "retriever": str(retriever),
        "llm": str(llm)
    }

