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
        model="openai/gpt-oss-120b",
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
    getter = getattr(response, "get_headers", None)
    if callable(getter):
        headers = getter() or {}
        for key, value in headers.items():
            if key.lower() == "x-portkey-cache-status" and value:
                return str(value).upper()
    return "MISS"