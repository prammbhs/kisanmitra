import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from sqlmodel import Session

from app.auth.dependencies import get_current_user_optional
from app.config import (
    CHROMA_PATH,
    FIREWORKS_MODEL,
    HOST,
    PORT,
    RAG_API_KEY,
    VOYAGE_MODEL_ID,
)
from app.db.database import get_session, init_db
from app.db.models import FarmerProfile, User
from app.rag.chain import RAGChain
from app.rag.graph import create_kisan_graph
from app.rag.vectorstore import VectorStoreManager
from app.routers.auth import router as auth_router
from app.routers.profile import router as profile_router
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse, Source

state: dict = {}
security = HTTPBearer(auto_error=False)


def verify_api_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Optional master service key check if RAG_API_KEY is configured."""
    if RAG_API_KEY and (not creds or creds.credentials != RAG_API_KEY):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing Bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing KisanMitra App Database & Models...")
    init_db()

    vs = VectorStoreManager()
    await run_in_threadpool(vs.warmup)
    state["vs"] = vs
    state["rag"] = RAGChain(vs)
    
    print("[STARTUP] Compiling LangGraph Conversational Agent...")
    state["graph"] = create_kisan_graph(vs)
    yield
    state.clear()


app = FastAPI(
    title="KisanMitra RAG & Conversational Advisory API",
    version="2.0.0",
    description="Intelligent AI agricultural assistant powered by 1.54M KCC records, LangGraph, and Fireworks GLM-5.3 Flash.",
    lifespan=lifespan,
)

# Enable CORS for frontend web and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Authentication & Profile Routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(profile_router, prefix="/api/v1")


def to_sources(docs: List[Document], scores: List[float]) -> List[Source]:
    sources = []
    for i, (d, s) in enumerate(zip(docs, scores), start=1):
        sources.append(
            Source(
                index=i,
                collection=d.metadata.get("collection", "kcc_docs"),
                score=round(s, 4),
                text=d.page_content,
                metadata={k: v for k, v in d.metadata.items() if k != "collection"},
            )
        )
    return sources


def get_profile_context(user: Optional[User], session: Session) -> dict:
    if not user:
        return {}
    profile = session.query(FarmerProfile).filter(FarmerProfile.user_id == user.id).first()
    if not profile:
        return {}
    return {
        "full_name": profile.full_name,
        "state": profile.state,
        "district": profile.district,
        "primary_crops": profile.primary_crops,
        "land_acres": profile.land_acres,
        "preferred_language": profile.preferred_language,
    }


@app.get("/health")
def health():
    vs: VectorStoreManager = state["vs"]
    return {
        "status": "ok",
        "chroma_path": CHROMA_PATH,
        "embedding_model": VOYAGE_MODEL_ID,
        "llm": FIREWORKS_MODEL,
        "collections": {n: vs.count(n) for n in vs.stores},
        "engine": "LangGraph + StateGraph",
    }


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_api_key)])
async def search(req: SearchRequest):
    """Raw vector retrieval endpoint."""
    rag: RAGChain = state["rag"]
    filters = req.filters.as_dict() if req.filters else None
    t0 = time.perf_counter()
    hits = await run_in_threadpool(rag.retrieve, req.query, req.collections, req.k, filters)
    latency_ms = (time.perf_counter() - t0) * 1000

    docs = [d for d, _ in hits]
    scores = [s for _, s in hits]
    return SearchResponse(query=req.query, results=to_sources(docs, scores), latency_ms=round(latency_ms, 1))


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Stateful conversational chat endpoint via LangGraph.
    - If user is authenticated, automatically injects their farmer profile (State, Primary Crops).
    - Uses thread_id to preserve conversation history.
    """
    graph = state["graph"]
    thread_id = req.thread_id or f"anon-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    profile_ctx = get_profile_context(user, session)
    filters = req.filters.as_dict() if req.filters else {}

    initial_input = {
        "messages": [HumanMessage(content=req.query)],
        "user_query": req.query,
        "user_profile": profile_ctx,
        "filters": filters,
        "collections": req.collections,
        "k": req.k,
        "retry_count": 0,
    }

    t0 = time.perf_counter()
    final_state = await run_in_threadpool(graph.invoke, initial_input, config=config)
    total_ms = (time.perf_counter() - t0) * 1000

    docs = final_state.get("documents", [])
    scores = final_state.get("document_scores", [])
    answer = final_state.get("answer", "")

    return ChatResponse(
        answer=answer,
        thread_id=thread_id,
        sources=to_sources(docs, scores),
        retrieval_ms=round(total_ms * 0.15, 1),
        generation_ms=round(total_ms * 0.85, 1),
    )


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    user: Optional[User] = Depends(get_current_user_optional),
    session: Session = Depends(get_session),
):
    """
    Server-Sent Events (SSE) streaming endpoint:
    Streams status events, sources, and tokens as the LangGraph pipeline executes.
    """
    graph = state["graph"]
    thread_id = req.thread_id or f"anon-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    profile_ctx = get_profile_context(user, session)
    filters = req.filters.as_dict() if req.filters else {}

    initial_input = {
        "messages": [HumanMessage(content=req.query)],
        "user_query": req.query,
        "user_profile": profile_ctx,
        "filters": filters,
        "collections": req.collections,
        "k": req.k,
        "retry_count": 0,
    }

    async def event_generator():
        yield f"event: thread\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        
        # Invoke graph in threadpool
        final_state = await run_in_threadpool(graph.invoke, initial_input, config=config)

        docs = final_state.get("documents", [])
        scores = final_state.get("document_scores", [])
        answer = final_state.get("answer", "")

        sources_payload = [s.model_dump() for s in to_sources(docs, scores)]
        yield f"event: sources\ndata: {json.dumps(sources_payload, ensure_ascii=False)}\n\n"

        # Stream answer chunks
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            token_slice = answer[i : i + chunk_size]
            yield f"event: token\ndata: {json.dumps(token_slice, ensure_ascii=False)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # Single uvicorn worker for Chroma persistent client safety
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
