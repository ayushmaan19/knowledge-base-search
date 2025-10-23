# Minimal FastAPI ChromaDB server for Node.js integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

app = FastAPI()

# Initialize ChromaDB client and collection
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("knowledge-base")

class UploadRequest(BaseModel):
    documents: List[str]
    embeddings: List[List[float]]
    metadatas: Optional[List[Dict[str, Any]]] = None

class QueryRequest(BaseModel):
    embedding: List[float]
    top_k: int = 3
    where: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None

@app.post("/collections/knowledge-base/documents")
def add_documents(req: UploadRequest):
    ids = [f"doc_{i}" for i in range(len(req.documents))]
    try:
        if req.metadatas:
            collection.add(
                documents=req.documents,
                embeddings=req.embeddings,
                metadatas=req.metadatas,
                ids=ids,
            )
        else:
            collection.add(
                documents=req.documents,
                embeddings=req.embeddings,
                ids=ids,
            )
        return {"status": "ok", "count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/knowledge-base/query")
def query_collection(req: QueryRequest):
    try:
        results = collection.query(
            query_embeddings=[req.embedding],
            n_results=req.top_k,
        )

        # results typically has nested lists per query
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        # Apply simple where filtering (supports {"file": {"$eq": "name"}} or {"file": "name"})
        if req.where:
            file_value = None
            file_clause = req.where.get("file")
            if isinstance(file_clause, dict) and "$eq" in file_clause:
                file_value = file_clause["$eq"]
            elif isinstance(file_clause, str):
                file_value = file_clause
            if file_value is not None:
                filtered_docs = []
                filtered_metas = []
                filtered_dists = []
                filtered_ids = []
                for d, m, dist, _id in zip(docs, metadatas, distances, ids):
                    try:
                        if isinstance(m, dict) and m.get("file") == file_value:
                            filtered_docs.append(d)
                            filtered_metas.append(m)
                            filtered_dists.append(dist)
                            filtered_ids.append(_id)
                    except Exception:
                        continue
                docs = filtered_docs
                metadatas = filtered_metas
                distances = filtered_dists
                ids = filtered_ids

        # Build response including requested include fields or default set
        resp = {}
        resp["documents"] = docs
        resp["metadatas"] = metadatas
        resp["distances"] = distances
        resp["ids"] = ids
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ChromaDB FastAPI server running"}
