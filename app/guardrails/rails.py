import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, DIALOG_INDICATORS, YAML_CONTENT, BLOCK_INDICATORS

import os

os.environ.setdefault("FASTEMBED_CACHE_PATH", os.path.join(os.getcwd(), ".cache", "fastembed"))
    
_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-20b for fast intent classification at the gate —
    the heavier openai/gpt-oss-120b is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b).")
    
    



from app.guardrails.prompt_guard import check_jailbreak

JAILBREAK_REFUSAL = "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?"


def guard(message: str) -> tuple[str, str | None]:
    """
    Run a user message through the guardrail gates.

    Layer 1: Prompt Guard classifier — catches jailbreak/prompt-injection
             attempts regardless of exact phrasing (not dependent on
             example-list similarity).
    Layer 2: NeMo Guardrails — catches off-topic queries and handles
             scripted dialog flows (greeting/farewell/capabilities).

    Returns:
        ("block",  response) — refusal; skip the RAG pipeline.
        ("dialog", response) — scripted dialog answer; skip the RAG pipeline.
        ("pass",   None)     — message is clean; proceed to LangGraph.
    """
    # Layer 1: dedicated jailbreak classifier
    is_jailbreak, score = check_jailbreak(message)
    if is_jailbreak:
        logfire.info(f"🛡️ Prompt Guard blocked | score={score:.4f} | query='{message[:80]}'")
        return "block", JAILBREAK_REFUSAL

    # Layer 2: NeMo Guardrails (off-topic + dialog flows)
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return "pass", None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception as e:
            # e.g. Groq 429. Prompt Guard (layer 1) and the planner scope check
            # (layer 3) still protect us, so we pass to the planner instead of crashing.
            logfire.warning(
                f"⚠️ NeMo gate failed ({type(e).__name__}): {str(e)[:150]} - passing to planner."
            )
            return "pass", None
        content = result.get("content", "") if isinstance(result, dict) else str(result)
          
        if any(indicator in content for indicator in BLOCK_INDICATORS):
            logfire.info(f"🛡️ Guardrails blocked | query='{message[:80]}'")
            return "block", content

        if any(indicator in content for indicator in DIALOG_INDICATORS):
            logfire.info(f"💬 Guardrails dialog flow | query='{message[:80]}'")
            return "dialog", content

        logfire.info("✅ Guardrails passed.")
        return "pass", None