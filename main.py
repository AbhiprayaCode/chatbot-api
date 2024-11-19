from fastapi import FastAPI
from pydantic import BaseModel
from src.helper import load_pdf_file, load_csv_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import os

app = FastAPI()

# Model request untuk input
class QueryRequest(BaseModel):
    question: str

# Load environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Inisialisasi Pinecone dan embeddings
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

# Load Pinecone vector store
docsearch = PineconeVectorStore(embedding=embeddings, index_name=index_name)

@app.post("/query")
async def get_answer(query: QueryRequest):
    user_question = query.question
    # Dapatkan respons dari Pinecone/AI
    result = docsearch.similarity_search(user_question)
    return {"response": result}
