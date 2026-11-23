import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embeddings import embed_query


# -----------------------------------------------------------------------------
# Qdrant Client
# -----------------------------------------------------------------------------

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# -----------------------------------------------------------------------------
# Enterprise Knowledge Search
# -----------------------------------------------------------------------------

def search_enterprise_knowledge(
    query: str,
    limit: int = 15,
    category: str | None = None,
):
    """
    Semantic search. The category is a HINT, not a gate:
    we search with the category filter AND without it, merge the results
    (deduplicated), and let the reranker choose the best chunks.
    A wrong category guess from the planner can no longer hide the answer.
    """

    try:
        query_vector = embed_query(query)

        def _run(query_filter):
            return client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ).points

        merged = {}
        filtered_count = 0

        if category:
            category_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category),
                    )
                ]
            )
            logfire.info(f"🔎 Category hint → {category} (soft filter)")
            for p in _run(category_filter):
                merged[p.id] = p
            filtered_count = len(merged)
        else:
            logfire.info("🔎 No category hint.")

        for p in _run(None):
            merged.setdefault(p.id, p)

        points = sorted(merged.values(), key=lambda p: p.score, reverse=True)

        results = []
        for res in points:
            payload = res.payload or {}
            results.append(
                {
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "source_type": payload.get("source_type", "Unknown"),
                    "category": payload.get("category", "general"),
                    "score": res.score,
                }
            )

        logfire.info(
            f"✅ Qdrant returned {len(results)} candidates "
            f"({filtered_count} from category filter, rest from unfiltered search)."
        )

        return results

    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []