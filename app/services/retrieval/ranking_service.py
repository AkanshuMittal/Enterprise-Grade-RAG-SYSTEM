import time
 
import requests
import logfire
 
from app.config import settings
 
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
MAX_RETRIES = 4
 
 
def rerank_documents(
    query: str,
    documents: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Rerank `documents` (each a dict with a 'content' key, as returned by
    qdrant_service.search_enterprise_knowledge) against `query` using Jina's
    hosted reranker. Returns the top_n documents, in relevance order, with
    their original fields preserved plus a new 'rerank_score' field.
    No silent fallback — a reranker failure should surface, not silently
    return the unranked top-N by vector score.
    """
    if not documents:
        return []
 
    if not settings.JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is not configured.")
 
    passages = [doc["content"] for doc in documents]
    last_error: Exception | None = None
 
    with logfire.span(
        "Jina Reranking",
        candidates=len(passages),
        top_n=top_n,
    ):
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    JINA_RERANK_URL,
                    headers={
                        "Authorization": f"Bearer {settings.JINA_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": JINA_RERANKER_MODEL,
                        "query": query,
                        "documents": passages,
                        "top_n": top_n,
                        "return_documents": False,  # we already have the text locally
                    },
                    timeout=60,
                )
 
                if response.status_code == 429 and attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Jina rerank rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})."
                    )
                    time.sleep(wait)
                    continue
 
                response.raise_for_status()
                results = response.json()["results"]
 
                reranked = []
                for item in results:
                    original_doc = documents[item["index"]]
                    reranked.append(
                        {
                            **original_doc,
                            "rerank_score": item["relevance_score"],
                        }
                    )
 
                logfire.info(
                    f"Jina reranked {len(passages)} candidates → "
                    f"top {len(reranked)} returned."
                )
 
                return reranked
 
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logfire.warning(f"Jina rerank request failed ({e}) — retrying in {wait}s.")
                    time.sleep(wait)
                else:
                    logfire.error(f"Jina reranking failed after {MAX_RETRIES} attempts: {e}")
                    raise
 
    raise RuntimeError(f"Jina reranking failed: {last_error}")
 