import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.documents import Document

from app.config import CHROMA_PATH, FIREWORKS_MODEL, HOST, PORT, RAG_API_KEY, VOYAGE_MODEL_ID
from app.rag.chain import RAGChain
from app.rag.vectorstore import VectorStoreManager
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, Source

state: dict = {}
security = HTTPBearer(auto_error=False)


def verify_token(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if RAG_API_KEY and (not creds or creds.credentials != RAG_API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing Bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    vs = VectorStoreManager()
    await run_in_threadpool(vs.warmup)
    state["vs"] = vs
    state["rag"] = RAGChain(vs)
    yield
    state.clear()


app = FastAPI(title="KisanMitra RAG API", version="0.1.0", lifespan=lifespan)


def to_sources(hits: List[Tuple[Document, float]]) -> List[Source]:
    return [
        Source(
            index=i,
            collection=d.metadata.get("collection", ""),
            score=round(s, 4),
            text=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if k != "collection"},
        )
        for i, (d, s) in enumerate(hits, start=1)
    ]


async def _retrieve(req: SearchRequest):
    rag: RAGChain = state["rag"]
    filters = req.filters.as_dict() if req.filters else None
    t0 = time.perf_counter()
    # Chroma + Voyage calls are blocking -> threadpool
    hits = await run_in_threadpool(rag.retrieve, req.query, req.collections, req.k, filters)
    return hits, (time.perf_counter() - t0) * 1000


@app.get("/health")
def health():
    vs: VectorStoreManager = state["vs"]
    return {
        "status": "ok",
        "chroma_path": CHROMA_PATH,
        "embedding_model": VOYAGE_MODEL_ID,
        "llm": FIREWORKS_MODEL,
        "collections": {n: vs.count(n) for n in vs.stores},
    }


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_token)])
async def search(req: SearchRequest):
    hits, ms = await _retrieve(req)
    return SearchResponse(query=req.query, results=to_sources(hits), latency_ms=round(ms, 1))


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest):
    hits, r_ms = await _retrieve(req)
    t0 = time.perf_counter()
    answer = await state["rag"].answer(req.query, hits)
    return ChatResponse(
        answer=answer,
        sources=to_sources(hits),
        retrieval_ms=round(r_ms, 1),
        generation_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/chat/stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest):
    """Server-Sent Events: one `sources` event, then `token` events, then `done`."""
    hits, _ = await _retrieve(req)

    async def gen():
        payload = [s.model_dump() for s in to_sources(hits)]
        yield f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        async for tok in state["rag"].astream(req.query, hits):
            yield f"event: token\ndata: {json.dumps(tok, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # Single worker: Chroma PersistentClient is not multi-process safe.
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
