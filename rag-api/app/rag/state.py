"""LangGraph State definition for KisanMitra Conversational Agent."""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history with automatic message appending
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Input query for current turn
    user_query: str
    
    # Profile context (State, District, Primary Crops, Language)
    user_profile: Dict[str, Any]
    
    # De-contextualized standalone query after analyzing conversation history
    standalone_query: str
    
    # Extracted or profile-defaulted metadata filters
    filters: Optional[Dict[str, str]]
    
    # Collections to search
    collections: List[str]
    
    # Top-K to retrieve
    k: int
    
    # Retrieved chunks and scores
    documents: List[Document]
    document_scores: List[float]
    
    # Retry tracker for conditional relaxation loop
    retry_count: int
    
    # Final synthesized answer
    answer: str
