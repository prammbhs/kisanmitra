"""LangGraph StateGraph builder with SQLite Checkpointing on EBS."""
import os
import sqlite3
from typing import Any, Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import CHECKPOINT_DB_PATH
from app.rag.nodes import (
    analyze_and_rewrite,
    decide_next_step,
    fallback_advisory,
    generate_answer,
    relax_and_retry,
    retrieve_kcc,
)
from app.rag.state import AgentState
from app.rag.vectorstore import VectorStoreManager


def create_kisan_graph(vector_store: VectorStoreManager):
    """Constructs and compiles the KisanMitra Conversational RAG graph."""
    # Ensure directory on EBS exists for checkpointer
    checkpoint_dir = os.path.dirname(CHECKPOINT_DB_PATH)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # SQLite connection for persistent conversation checkpoints
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Initialize Graph
    workflow = StateGraph(AgentState)

    # Wrap vectorstore retrieve node
    def retrieve_node(state: AgentState):
        return retrieve_kcc(state, vector_store)

    # Add Nodes
    workflow.add_node("analyze_and_rewrite", analyze_and_rewrite)
    workflow.add_node("retrieve_kcc", retrieve_node)
    workflow.add_node("relax_and_retry", relax_and_retry)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("fallback_advisory", fallback_advisory)

    # Define Graph Edges
    workflow.add_edge(START, "analyze_and_rewrite")
    workflow.add_edge("analyze_and_rewrite", "retrieve_kcc")

    # Conditional Routing after Retrieval
    workflow.add_conditional_edges(
        "retrieve_kcc",
        decide_next_step,
        {
            "generate_answer": "generate_answer",
            "relax_and_retry": "relax_and_retry",
            "fallback_advisory": "fallback_advisory",
        },
    )

    # Loop back relaxed search to retrieve node
    workflow.add_edge("relax_and_retry", "retrieve_kcc")

    # Final nodes complete the graph
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("fallback_advisory", END)

    # Compile with EBS checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    print(f"[LANGGRAPH] Compiled KisanMitra StateGraph (Checkpointer: '{CHECKPOINT_DB_PATH}')")
    return app
