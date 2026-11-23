from app.agents.state import AgentState

OFF_TOPIC_MESSAGE = (
    "I can only help with technical topics in my knowledge base, such as "
    "Kubernetes, Databricks, networking, and computer architecture. "
    "What would you like to know?"
)


def off_topic_node(state: AgentState):
    return {
        "final_answer": OFF_TOPIC_MESSAGE,
        "status": "Blocked by guardrails.",
        # keeps conversation memory consistent: every user turn gets a reply
        "messages": [{"role": "assistant", "content": OFF_TOPIC_MESSAGE}],
    }