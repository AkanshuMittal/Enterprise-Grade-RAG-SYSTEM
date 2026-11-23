import os
import secrets
from pathlib import Path

import logfire
from dotenv import load_dotenv


def configure_logfire() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=False)

    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("Logfire token not found; skipping telemetry.")
        return

    try:
        logfire.configure(token=token, service_name="enterprise-agentic-rag-api")
    except Exception as exc:
        print(f"Logfire authorization failed: {exc}")
        logfire.configure(send_to_logfire=False)


configure_logfire()

# Now safe to import app modules - logfire is already active
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.agents.graph import rag_agent
from app.guardrails.rails import initialize_rails, guard
from app.config import settings

from typing import Optional

RAG_API_KEY = os.getenv("RAG_API_KEY", "")

def client_ip(request: Request) -> str:
    """Client IP for rate limiting. Behind Cloud Run it is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_api_key(x_api_key: str = Header(default="")) -> None:
    # Fail closed: if RAG_API_KEY is not configured, every protected call is rejected.
    if not RAG_API_KEY or not secrets.compare_digest(
        x_api_key.encode(), RAG_API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


limiter = Limiter(key_func=client_ip)

# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


logfire.instrument_fastapi(app)

@app.on_event("startup")
def startup_event():
    if not RAG_API_KEY:
        logfire.error("RAG_API_KEY is not set: protected requests will be rejected.")
    initialize_rails()


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=1000)
    thread_id: str = Field(
        default="default_user", max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/graph", dependencies=[Depends(require_api_key)])
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def query(request: Request, body: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = body.q
    thread_id = body.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic/jailbreaks, handles dialog shortcuts
        action, rail_response = guard(q)

        if action == "block":
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired (Blocked)", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        if action == "dialog":
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Dialog Flow", "Retrieval: Skipped"],
                "status": "Response generated.",
                "sources": []
            }

        
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }