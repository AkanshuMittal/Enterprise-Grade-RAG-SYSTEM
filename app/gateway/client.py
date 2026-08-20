from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_SLUG,
)

def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI using the saved gateway configuration.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model="llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_SLUG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "development"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"