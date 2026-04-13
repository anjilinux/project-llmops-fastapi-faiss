from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

app = FastAPI(title="LLMOps RAG API")

# -------------------
# EMBEDDINGS
# -------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------
# LOAD FAISS
# -------------------
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 2})

# -------------------
# LLM
# -------------------
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -------------------
# REQUEST MODEL
# -------------------
class Query(BaseModel):
    question: str

# -------------------
# ROUTES
# -------------------
@app.get("/")
def home():
    return {"status": "running"}

@app.post("/chat")
def chat(query: Query):

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