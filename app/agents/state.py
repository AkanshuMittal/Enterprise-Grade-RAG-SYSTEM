from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):

    messages: Annotated[
        List[dict],
        operator.add
    ]

    current_query: str

    category: str | None

    documents: List[str]

    plan: List[str]

    status: str

    final_answer: str