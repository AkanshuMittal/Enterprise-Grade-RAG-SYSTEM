from groq import Groq
from app.config import settings

_client = Groq(api_key=settings.GROQ_API_KEY)

JAILBREAK_THRESHOLD = 0.5


def check_jailbreak(message: str) -> tuple[bool, float]:
    """
    Runs the message through Meta's Prompt Guard 2 classifier.
    Returns (is_jailbreak, score) where score is a 0-1 probability
    that the message is a prompt-injection / jailbreak attempt.
    """
    response = _client.chat.completions.create(
        model="meta-llama/llama-prompt-guard-2-86m",
        messages=[{"role": "user", "content": message}],
    )
    score = float(response.choices[0].message.content)
    return score >= JAILBREAK_THRESHOLD, score