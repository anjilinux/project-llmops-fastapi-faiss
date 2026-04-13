from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLMOps RAG API")

class Query(BaseModel):
    question: str

# -------------------
# SAFE INIT (lazy load)
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
            retriever = None

    if llm is None:
        try:
            from transformers import pipeline

            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-base"
            )

        except Exception as e:
            print("LLM load failed:", e)
            llm = None


# -------------------
# ROUTES
# -------------------

@app.get("/")
def home():
    return {"status": "running", "message": "API live (safe mode)"}


@app.post("/chat")
def chat(query: Query):

    load_models()

    # fallback if models not loaded
    if retriever is None or llm is None:
        return {
            "question": query.question,
            "answer": "⚠️ Model not loaded (FAISS/LLM failed on server)"
        }

    docs = retriever.invoke(query.question)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using only context:

Context:
{context}

Question:
{query.question}

Answer:
"""

    result = llm(prompt, max_length=120, do_sample=False)

    return {
        "question": query.question,
        "answer": result[0]["generated_text"]
    }