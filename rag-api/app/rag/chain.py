"""Simple (linear) RAG: retrieve -> prompt -> Fireworks LLM.

Kept as plain functions so it can later become nodes in a LangGraph StateGraph.
"""
from typing import AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks

from app.config import (
    FIREWORKS_API_KEY,
    FIREWORKS_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MIN_RELEVANCE,
)
from app.rag.vectorstore import VectorStoreManager, build_where

SYSTEM_PROMPT = """You are KisanMitra, an expert agricultural advisor for Indian farmers.

Your goal is to provide practical, reliable, and actionable guidance to farmers:
1. **Primary Grounding**: Prioritize and integrate all specific facts, chemicals, doses, seed varieties, portal links, and local KVK contacts provided in the context below. Cite them inline as [1], [2] corresponding to the context numbers.
2. **General Cultivation Guidance**: If the farmer asks a broad cultivation or management question (e.g., how to grow a crop, package of practices) and the retrieved records only contain partial or contact information, provide the standard agronomic package of practices (optimal sowing window, seed rate, field preparation, fertilizer schedule, and critical irrigation stages).
3. **Regional Tailoring**: If specific regional data (like Bihar-specific dates or varieties) is in the context, highlight it; otherwise, provide general recommended practices and guide the farmer to their local KVK/agriculture officer or call the Kisan Call Centre at 1800-180-1551 (toll-free) for local agro-climatic adjustments.
4. **Language**: Always reply in the same language as the farmer's question (Hindi, English, Gujarati, etc.).
5. **Practicality**: Be clear, concise, and structured with bullet points or numbered steps.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def format_context(hits: List[Tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, _score) in enumerate(hits, start=1):
        m = doc.metadata
        tags = ", ".join(
            f"{k}={m[k]}" for k in ("StateName", "DistrictName", "Crop", "Season", "source") if m.get(k)
        )
        parts.append(f"[{i}] ({tags})\n{doc.page_content}")
    return "\n\n".join(parts) if parts else "(no relevant context found)"


class RAGChain:
    def __init__(self, vs: VectorStoreManager) -> None:
        if not FIREWORKS_API_KEY:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        self.vs = vs
        self.llm = ChatFireworks(
            model=FIREWORKS_MODEL,
            api_key=FIREWORKS_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        self.chain = PROMPT | self.llm | StrOutputParser()

    def retrieve(
        self, question: str, collections: List[str], k: int, filters: Optional[Dict[str, str]]
    ) -> List[Tuple[Document, float]]:
        hits = self.vs.search(question, collections, k, build_where(filters))
        return [h for h in hits if h[1] >= MIN_RELEVANCE]

    async def answer(self, question: str, hits: List[Tuple[Document, float]]) -> str:
        return await self.chain.ainvoke({"question": question, "context": format_context(hits)})

    async def astream(self, question: str, hits: List[Tuple[Document, float]]) -> AsyncIterator[str]:
        async for token in self.chain.astream({"question": question, "context": format_context(hits)}):
            yield token
