import logfire
import requests

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings


BATCH_SIZE = 64

_EMBEDDING_DIM = 1024

_JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"
_JINA_MODEL = "jina-embeddings-v3"


_active_model = None
_model_type: str | None = None  # "jina"


# ── Model initialisation ───────────────────────────────────────────────────────


def _probe_jina_api() -> bool:
    """Verify the Jina Embeddings API is reachable with the configured key."""

    if not settings.JINA_API_KEY:
        logfire.error("JINA_API_KEY is not set.")
        return False

    try:
        response = requests.post(
            _JINA_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_MODEL,
                "task": "retrieval.query",
                "normalized": True,
                "input": ["probe"],
            },
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        if not payload.get("data"):
            raise RuntimeError("Jina API returned empty data")

        logfire.info(
            "Jina Embeddings API ready "
            "(jina-embeddings-v3, 1024-dim)."
        )

        return True

    except Exception as e:
        logfire.warning(
            f"Jina Embeddings API probe failed: {e}."
        )
        return False


def _init():
    """Initialise embedding provider once per process."""

    global _active_model, _model_type

    if _active_model is not None or _model_type is not None:
        return

    if not _probe_jina_api():
        raise RuntimeError(
            "Jina Embeddings API is unavailable. "
            "Check JINA_API_KEY and connectivity."
        )

    # Jina API is stateless; no local model to keep.
    _active_model = None
    _model_type = "jina"


# ── Public helpers ─────────────────────────────────────────────────────────────


def get_embedding_dim() -> int:
    """Return the vector dimension for the active model."""

    _init()

    return _EMBEDDING_DIM


# ── Jina error handling ────────────────────────────────────────────────────────


def _is_retryable_jina_error(exception: Exception) -> bool:
    """
    Decide whether a Jina API error is temporary and worth retrying.

    Retry:
        429 -> Rate limit
        500 -> Internal server error
        502 -> Bad gateway
        503 -> Service unavailable
        504 -> Gateway timeout

    Do NOT retry:
        400 -> Bad request
        401 -> Invalid API key
        403 -> Forbidden
        404 -> Not found
    """

    if isinstance(exception, requests.HTTPError):

        response = exception.response

        if response is not None:
            return response.status_code in (
                429,
                500,
                502,
                503,
                504,
            )

    return False


# ── Jina API embedding ─────────────────────────────────────────────────────────



def _embed_jina_batch(
    texts: list[str],
    task: str,
) -> list[list[float]]:
    """Call the Jina Embeddings API for a single batch."""

    @retry(
        retry=retry_if_exception(_is_retryable_jina_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(
            multiplier=2,
            min=10,
            max=60,
        ),
        reraise=True,
        before_sleep=before_sleep_log(logfire, "warning"),
    )
    def _post_request():
        response = requests.post(
            _JINA_EMBEDDING_URL,
            headers={
                "Authorization": f"Bearer {settings.JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _JINA_MODEL,
                "task": task,
                "normalized": True,
                "input": texts,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response

    response = _post_request()

    response.raise_for_status()

    payload = response.json()

    results = payload.get("data", [])

    if not results:
        raise RuntimeError(
            "Jina API returned empty embedding data."
        )

    # Make sure embeddings stay in the same order as input.
    results_sorted = sorted(
        results,
        key=lambda x: x.get("index", 0),
    )

    return [
        item["embedding"]
        for item in results_sorted
    ]


def _embed(
    texts: list[str],
    task: str,
) -> list[list[float]]:
    """
    Embed texts using Jina API.

    Documents are processed using retrieval.passage.
    Queries are processed using retrieval.query.
    """

    _init()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i : i + BATCH_SIZE]

        with logfire.span(
            "Embed batch via Jina API",
            start=i,
            size=len(batch),
        ):
            embeddings = _embed_jina_batch(
                batch,
                task,
            )

            all_embeddings.extend(embeddings)

    return all_embeddings


# ── Public API ─────────────────────────────────────────────────────────────────


def embed_query(query: str) -> list[float]:
    """Embed a single user query."""

    return _embed(
        [query],
        task="retrieval.query",
    )[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks."""

    return _embed(
        texts,
        task="retrieval.passage",
    )