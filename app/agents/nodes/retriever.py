import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge
)
from app.services.retrieval.ranking_service import (
    rerank_documents
)


def retrieve_node(state: AgentState):

    query = state["current_query"]

    category = state.get("category")

    with logfire.span("🔍 Knowledge Retrieval"):

        logfire.info(
            f"Searching Qdrant for: {query}"
        )

        logfire.info(
            f"Category filter: {category or 'NONE'}"
        )

        raw_results = search_enterprise_knowledge(
            query,
            limit=15,
            category=category,
        )

        logfire.info(
            f"Retrieved {len(raw_results)} "
            f"candidates from Vector DB"
        )

        # ---------------------------------------------------------
        # Reranking
        # ---------------------------------------------------------

        with logfire.span(
            "⚖️ Semantic Reranking"
        ):

            reranked_docs = rerank_documents(
                query,
                raw_results,
                top_n=5,
            )

        logfire.info(
            f"Reranking complete. "
            f"Kept top {len(reranked_docs)} chunks."
        )

        formatted_docs = [

            (
                f"CONTENT: {doc['content']}\n"
                f"SOURCE: {doc.get('source', 'Unknown')}\n"
                f"CATEGORY: {doc.get('category', 'Unknown')}"
            )

            for doc in reranked_docs
        ]

    return {

        "documents": formatted_docs,

        "status": (
            "Found technical context."
            if formatted_docs
            else "No relevant technical context found."
        ),

        "plan": state["plan"] + [
            "Context Retrieved"
        ],
    }