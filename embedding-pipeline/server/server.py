import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from server.config import (
    HOST,
    PORT,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    EMBEDDING_DIMENSIONS,
    CHROMA_PATH,
    EMBEDDING_API_KEY,
    ALLOWED_COLLECTIONS,
)
from server.model import BedrockTitanEmbeddingManager
from server.database import ChromaDBManager

# Runtime Statistics tracking
stats_counter = {
    "total_requests": 0,
    "total_chunks_processed": 0,
    "total_processing_time_seconds": 0.0,
}

security = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if EMBEDDING_API_KEY:
        if not credentials or credentials.credentials != EMBEDDING_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing Authorization Bearer token",
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVER] Starting FastAPI CPU Embedding Server (AWS Bedrock Titan V2)...")
    BedrockTitanEmbeddingManager.get_instance()
    ChromaDBManager.get_instance()
    yield
    print("[SERVER] Shutting down FastAPI CPU Embedding Server...")

app = FastAPI(
    title="AWS Bedrock Titan V2 Embedding Server",
    description="FastAPI service for bulk text embedding with Amazon Titan Text Embeddings V2 via AWS Bedrock, persisting into ChromaDB on EBS",
    version="2.0.0",
    lifespan=lifespan,
)

class ItemSchema(BaseModel):
    id: str = Field(..., description="Unique record chunk ID")
    text: str = Field(..., description="Text content to embed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata dictionary")

class EmbedRequest(BaseModel):
    collection_name: Optional[str] = Field("kcc_docs", description="Target collection name: kcc_docs or other_docs")
    items: List[ItemSchema] = Field(..., description="List of items to embed and store")

class ItemResult(BaseModel):
    id: str
    status: str

class EmbedResponse(BaseModel):
    status: str
    processed_count: int
    collection_name: str
    processing_time_sec: float
    items: List[ItemResult]

@app.get("/health")
def health_check():
    db_mgr = ChromaDBManager.get_instance()
    counts = {col: db_mgr.get_collection_count(col) for col in ALLOWED_COLLECTIONS}
    return {
        "status": "ok",
        "provider": "AWS Bedrock",
        "model": BEDROCK_MODEL_ID,
        "region": AWS_REGION,
        "dimensions": EMBEDDING_DIMENSIONS,
        "chroma_path": CHROMA_PATH,
        "collections": counts,
    }

@app.get("/stats")
def get_stats():
    avg_req_time = (
        stats_counter["total_processing_time_seconds"] / stats_counter["total_requests"]
        if stats_counter["total_requests"] > 0
        else 0.0
    )
    return {
        "total_requests": stats_counter["total_requests"],
        "total_chunks_processed": stats_counter["total_chunks_processed"],
        "total_processing_time_seconds": round(stats_counter["total_processing_time_seconds"], 3),
        "avg_request_time_seconds": round(avg_req_time, 3),
    }

@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(verify_token)])
def embed_batch(request: EmbedRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="Items list cannot be empty")

    start_t = time.time()
    col_name = request.collection_name if request.collection_name in ALLOWED_COLLECTIONS else "other_docs"

    ids = [item.id for item in request.items]
    texts = [item.text for item in request.items]
    metadatas = [item.metadata for item in request.items]

    try:
        model_mgr = BedrockTitanEmbeddingManager.get_instance()
        db_mgr = ChromaDBManager.get_instance()

        # 1. Encode text via AWS Bedrock Titan Text Embeddings V2
        embeddings = model_mgr.encode_texts(texts)

        # 2. Store directly into ChromaDB on EBS
        db_mgr.upsert_batch(
            collection_name=col_name,
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        elapsed = time.time() - start_t

        # Update stats
        stats_counter["total_requests"] += 1
        stats_counter["total_chunks_processed"] += len(ids)
        stats_counter["total_processing_time_seconds"] += elapsed

        item_results = [ItemResult(id=item_id, status="stored") for item_id in ids]

        return EmbedResponse(
            status="success",
            processed_count=len(ids),
            collection_name=col_name,
            processing_time_sec=round(elapsed, 3),
            items=item_results,
        )

    except Exception as e:
        print(f"[ERROR] Exception during /embed processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
