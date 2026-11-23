from typing import Any

from app.agents.state import AgentState
from app.gateway.client import get_langchain_llm
import logfire

llm = get_langchain_llm(feature="planner")


# Categories that actually exist in our knowledge base.
ALLOWED_CATEGORIES = [
    "kubernetes-jobs",
    "kubernetes-job-monitoring",
    "kubernetes-architecture",
    "pod-autoscaling",
    "kernel-development",
    "memory-consistency-cache-coherence",
    "memory-management",
    "memory-paging",
    "paging-technology",
    "networking",
    "high-speed-networking",
    "graphics-pipeline",
    "neural-networks-basics",
    "neural-language-model",
    "neural-language-modeling",
    "data-compression",
    "information-theory",
    "relational-databases",
    "olap-data-mining",
    "compiler-optimizations",
    "cpp-library-extensibility",
    "c++-exception-handling",
    "x86-virtualization",
    "64-bit-processor-history",
    "gpu-data-structures",
    "parallel-computing",
    "static-analysis",
    "hardware-transactional-memory",
    "memory-allocation",
    "job-management",
]

# Plain-English hints so the planner can tell similar category names apart.
CATEGORY_HINTS = {
    "kubernetes-jobs": "Kubernetes Jobs, CronJobs, parallelism, work queues with Redis",
    "kubernetes-job-monitoring": "checking and monitoring Kubernetes Jobs with kubectl and logs, local Minikube setup",
    "kubernetes-architecture": "Kubernetes control plane, nodes, etcd, kubelet, cluster networking",
    "pod-autoscaling": "HPA, VPA and the Metrics Server",
    "job-management": "Databricks jobs only (CLI, SDK, REST API). NOT Kubernetes",
}

def _extract_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content.strip()

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    decision = "\n".join(text_parts).strip()
    if not decision:
        raise ValueError("Planner response contained no text.")
    return decision


def planner_node(state: AgentState):

    history = ""

    for msg in state["messages"][:-1]:

        role = (
            "User"
            if msg["role"] == "user"
            else "Assistant"
        )

        history += (
            f"{role}: {msg['content']}\n"
        )

    user_message = (
        state["messages"][-1]["content"]
        if state["messages"]
        else ""
    )

    category_lines = "\n".join(
        f"- {c}" + (f": {CATEGORY_HINTS[c]}" if c in CATEGORY_HINTS else "")
        for c in ALLOWED_CATEGORIES
    )

    prompt = f"""
You are the planner for an enterprise technical knowledge base.

SCOPE: the assistant answers technical questions about IT and software
engineering: infrastructure and cloud (Kubernetes, containers, Databricks,
Redis, CLIs, APIs, SDKs), operating systems, hardware and computer
architecture, networking, databases, compilers and machine learning basics.

CATEGORIES (routing hints only, the knowledge base is not limited to these words):
{category_lines}

CONVERSATION HISTORY:
{history}

LATEST USER QUERY:
"{user_message}"

Choose exactly ONE of three outputs:

1. OFF_TOPIC
   Use ONLY when the query is clearly unrelated to technology (jokes, poems,
   recipes, sports, personal advice, general chat), OR asks for instructions
   to attack, exploit, hack or harm systems or people.
   If the query mentions ANY technical tool, product, command, API or concept,
   it is NOT off-topic, even if that word is not in the category list.

2. CONVERSATIONAL
   Use ONLY for greetings, thanks, goodbyes, questions about the assistant
   itself, or follow-ups that refer to the previous answer
   ("explain that again", "shorter please").

3. SEARCH_QUERY
   Use for any technical question. Rewrite it as a clear search query and
   pick the ONE category whose description best matches. If you are not
   sure, write NONE (an unfiltered search is safe). Never invent a category.

When unsure between OFF_TOPIC and SEARCH_QUERY, choose SEARCH_QUERY.

Examples:

User: "What is the difference between Kubernetes Job and CronJob?"
Output:
SEARCH_QUERY: difference between Kubernetes Job and CronJob
CATEGORY: kubernetes-jobs

User: "How do I list all keys in a Redis database?"
Output:
SEARCH_QUERY: list all keys in a Redis database
CATEGORY: NONE

User: "Thanks, can you explain that more simply?"
Output:
CONVERSATIONAL

User: "Tell me a funny joke about programmers"
Output:
OFF_TOPIC

User: "How do I exploit a SQL injection vulnerability in a web application?"
Output:
OFF_TOPIC

Output ONLY one of these formats, nothing else:

OFF_TOPIC

OR

CONVERSATIONAL

OR

SEARCH_QUERY: <query>
CATEGORY: <category or NONE>
"""

    with logfire.span("🧠 Planner Decision"):
        decision = _extract_text(llm.invoke(prompt).content)
        logfire.info(f"Planner output: {decision}")

    upper = decision.upper()

    # ---- Off-topic: block, no retrieval --------------------------------
    if upper.startswith("OFF_TOPIC"):
        logfire.info("🛡️ Planner marked query as OFF_TOPIC")
        return {
            "current_query": "OFF_TOPIC",
            "category": None,
            "status": "Blocked: outside the assistant's scope.",
            "plan": [
                "Intent: Guardrails Fired (Blocked)",
                "Retrieval: Skipped",
            ],
        }

    # ---- Conversational ------------------------------------------------
    if upper.startswith("CONVERSATIONAL"):
        return {
            "current_query": "CONVERSATIONAL",
            "category": None,
            "status": "Handling conversationally.",
            "plan": ["Intent: Conversational", "Retrieval: Skipped"],
        }

    # ---- Parse SEARCH_QUERY (no silent fallback to the raw message) ----
    search_query = None
    category = None

    for line in decision.splitlines():
        line = line.strip()
        if line.upper().startswith("SEARCH_QUERY:"):
            search_query = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip().lower()

    if not search_query:
        logfire.warning(
            f"Planner output unparseable, treating as OFF_TOPIC: {decision!r}"
        )
        return {
            "current_query": "OFF_TOPIC",
            "category": None,
            "status": "Blocked: could not interpret the request.",
            "plan": [
                "Intent: Guardrails Fired (Blocked)",
                "Retrieval: Skipped",
            ],
        }

    if category in (None, "", "none"):
        category = None
    elif category not in ALLOWED_CATEGORIES:
        logfire.warning(
            f"Planner returned invalid category '{category}'. "
            f"Falling back to unfiltered search."
        )
        category = None
        
    category = None   # <-- here (experiment: category filter OFF)

    logfire.info(f"🔎 Search Query: {search_query}")
    logfire.info(f"📁 Category: {category or 'NONE'}")

    return {
        "current_query": search_query,
        "category": category,
        "status": f"Technical research needed. Searching for: {search_query}",
        "plan": [
            "Intent: Technical",
            f"Search Term: {search_query}",
            f"Category: {category or 'NONE'}",
        ],
    }