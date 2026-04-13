from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLMOps RAG API - Stable")

class Query(BaseModel):
    question: str

# -------------------
# ONLY LOAD LIGHT LLM
# -------------------
llm = None

def load_llm():
    global llm

    if llm is None:
        try:
            from transformers import pipeline

            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small"   # ✅ light model
            )

            print("✅ LLM loaded")

        except Exception as e:
            print("❌ LLM failed:", e)
            llm = None


# -------------------
# ROUTES
# -------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "LLM only mode (FAISS disabled)"
    }


@app.post("/chat")
def chat(query: Query):

    load_llm()

    if llm is None:
        return {
            "answer": "❌ LLM failed to load"
        }

    result = llm(query.question, max_length=100)

    return {
        "question": query.question,
        "answer": result[0]["generated_text"]
    }