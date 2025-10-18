# Minimal FastAPI ChromaDB server for Node.js integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.config import Settings

app = FastAPI()

# Initialize ChromaDB client and collection
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("knowledge-base")

class UploadRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]

class QueryRequest(BaseModel):
    embedding: List[float]
    top_k: int = 3

@app.post("/collections/knowledge-base/documents")
def add_documents(req: UploadRequest):
    ids = [f"doc_{i}" for i in range(len(req.documents))]
    try:
        collection.add(
            documents=req.documents,
            embeddings=req.embeddings,
            ids=ids
        )
        return {"status": "ok", "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/knowledge-base/query")
def query_collection(req: QueryRequest):
    try:
        results = collection.query(
            query_embeddings=[req.embedding],
            n_results=req.top_k
        )
        docs = results.get("documents", [[]])[0]
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ChromaDB FastAPI server running"}
