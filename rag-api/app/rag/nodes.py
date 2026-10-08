"""LangGraph Nodes for Agricultural Analysis, Retrieval, Grading, and Generation."""
import json
from typing import Any, Dict, List, Literal, Tuple
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks

from app.config import (
    FIREWORKS_API_KEY,
    FIREWORKS_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MIN_RELEVANCE,
)
from app.rag.chain import SYSTEM_PROMPT, format_context
from app.rag.state import AgentState
from app.rag.vectorstore import VectorStoreManager, build_where


def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatFireworks:
    return ChatFireworks(
        model=FIREWORKS_MODEL,
        api_key=FIREWORKS_API_KEY,
        temperature=temperature,
        max_tokens=LLM_MAX_TOKENS,
    )


# --- Node 1: Analyze & Contextualize Query ---

REWRITE_PROMPT = """You are an agricultural query analyst for an Indian farmer advisory assistant.
Analyze the farmer's current message in the context of previous conversation history and their profile.

Farmer Profile:
- State: {state}
- District: {district}
- Primary Crops: {crops}
- Preferred Language: {language}

Tasks:
1. De-reference pronouns and ambiguous references into a clear, standalone agricultural search query.
   - Example 1: If history discussed 'wheat leaf yellowing' and user asks 'इसका क्या इलाज है?', rewrite to 'गेहूं में पत्तियों के पीलेपन का रासायनिक उपचार'.
   - Example 2: If user asks 'सफेद मक्खी की दवा' and their profile has crop 'Cotton', rewrite to 'कपास में सफेद मक्खी की रोकथाम और कीटनाशक'.
2. Extract relevant metadata filter if explicitly mentioned or clearly implied:
   - StateName (e.g. 'HARYANA', 'BIHAR', 'GUJARAT')
   - Crop (e.g. 'Wheat', 'Cotton (Kapas)', 'Paddy (Dhan)', 'Mustard')

Respond ONLY in valid JSON matching this structure:
{{
  "standalone_query": "<rewritten complete query>",
  "detected_crop": "<crop name or null>",
  "detected_state": "<state name or null>"
}}"""


def analyze_and_rewrite(state: AgentState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)
    profile = state.get("user_profile") or {}
    
    prompt = REWRITE_PROMPT.format(
        state=profile.get("state") or "Unknown",
        district=profile.get("district") or "Unknown",
        crops=", ".join(profile.get("primary_crops") or []) or "None specified",
        language=profile.get("preferred_language") or "hi",
    )

    # Convert past messages into text transcript
    history_lines = []
    for msg in state.get("messages", [])[:-1]:  # exclude latest message
        role = "Farmer" if isinstance(msg, HumanMessage) else "Advisor"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines[-6:]) if history_lines else "(No prior conversation)"

    prompt_messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Conversation History:\n{history_text}\n\nCurrent Farmer Message: {state['user_query']}")
    ]

    try:
        response = llm.invoke(prompt_messages)
        content = response.content.strip()
        # Clean potential markdown fences
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        parsed = json.loads(content.strip())
        
        standalone_q = parsed.get("standalone_query") or state["user_query"]
        crop = parsed.get("detected_crop")
        state_name = parsed.get("detected_state")

        # Build initial filters from query or profile
        filters = dict(state.get("filters") or {})
        if crop:
            filters["Crop"] = crop
        if state_name:
            filters["StateName"] = state_name

        print(f"[LANGGRAPH] Rewritten Query: '{standalone_q}' | Filters: {filters}")
        return {
            "standalone_query": standalone_q,
            "filters": filters,
            "retry_count": state.get("retry_count", 0),
        }
    except Exception as e:
        print(f"[LANGGRAPH] Error in analyze_and_rewrite: {e}. Falling back to raw query.")
        return {
            "standalone_query": state["user_query"],
            "filters": state.get("filters"),
            "retry_count": state.get("retry_count", 0),
        }


# --- Node 2: Retrieve from ChromaDB ---

def retrieve_kcc(state: AgentState, vs: VectorStoreManager) -> Dict[str, Any]:
    query = state.get("standalone_query") or state["user_query"]
    collections = state.get("collections") or ["kcc_docs"]
    k = state.get("k", 4)
    filters = state.get("filters")

    hits = vs.search(query=query, collections=collections, k=k, where=build_where(filters))
    
    docs = [doc for doc, _ in hits]
    scores = [score for _, score in hits]
    print(f"[LANGGRAPH] Retrieved {len(docs)} documents. Top Score: {scores[0] if scores else 0.0:.3f}")
    return {
        "documents": docs,
        "document_scores": scores,
    }


# --- Node 3: Grade Documents / Route Decision ---

def decide_next_step(state: AgentState) -> Literal["generate_answer", "relax_and_retry", "fallback_advisory"]:
    scores = state.get("document_scores", [])
    retry_count = state.get("retry_count", 0)

    # Check if we have at least one hit above MIN_RELEVANCE
    valid_hits = [s for s in scores if s >= MIN_RELEVANCE]
    
    if valid_hits:
        return "generate_answer"
    
    if retry_count < 1 and state.get("filters"):
        print("[LANGGRAPH] No high-confidence documents found with filters. Relaxing filters and retrying...")
        return "relax_and_retry"
    
    print("[LANGGRAPH] No relevant documents found after relaxation. Routing to fallback advisory.")
    return "fallback_advisory"


# --- Node 4: Relax Filters & Retry Search ---

def relax_and_retry(state: AgentState) -> Dict[str, Any]:
    """Drops strict filters to allow a broader semantic search across all states."""
    print("[LANGGRAPH] Dropping metadata filters for broader search.")
    return {
        "filters": None,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# --- Node 5: Generate Answer ---

def generate_answer(state: AgentState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)
    docs = state.get("documents", [])
    scores = state.get("document_scores", [])
    query = state.get("standalone_query") or state["user_query"]

    paired_hits = list(zip(docs, scores))
    context_str = format_context(paired_hits)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    chain = prompt | llm
    res = chain.invoke({"question": query, "context": context_str})
    answer_text = res.content

    return {
        "answer": answer_text,
        "messages": [AIMessage(content=answer_text)]
    }


# --- Node 6: Fallback Advisory ---

def fallback_advisory(state: AgentState) -> Dict[str, Any]:
    lang = (state.get("user_profile") or {}).get("preferred_language", "hi")
    query = state["user_query"]

    if lang == "en":
        fallback_msg = (
            f"I could not find exact advisory records matching your specific query: '{query}'.\n\n"
            f"**Recommendation:** Please contact the Kisan Call Centre at **1800-180-1551** (Toll-Free). "
            f"Agriculture experts are available from 6:00 AM to 10:00 PM to assist you with local guidance."
        )
    else:
        fallback_msg = (
            f"आपके प्रश्न '{query}' के लिए डेटाबेस में कोई सीधा ऐतिहासिक रिकॉर्ड नहीं मिल पाया है।\n\n"
            f"**सुझाव:** कृपया अधिक सटीक और स्थानीय सलाह के लिए राष्ट्रीय किसान कॉल सेंटर (Kisan Call Centre) के "
            f"टोल-फ्री नंबर **1800-180-1551** पर संपर्क करें। कृषि विशेषज्ञ सुबह 6:00 बजे से रात 10:00 बजे तक निःशुल्क उपलब्ध हैं।"
        )

    return {
        "answer": fallback_msg,
        "messages": [AIMessage(content=fallback_msg)]
    }
