import os
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.prompt import Prompt
from typing_extensions import TypedDict

from tools import add_experience, search_experiences, edit_experience

load_dotenv()

TOOLS = [add_experience, search_experiences, edit_experience]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        timeout=float(os.getenv("AI_TIMEOUT_SECONDS", "45")),
        max_retries=1,
    ).bind_tools(TOOLS)


def human_input_node(state: AgentState) -> dict:
    question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
    return {"messages": [HumanMessage(content=question)]}


LLM_MIN_TOKENS = int(os.getenv("AI_MIN_TOKENS", "2"))
LLM_MAX_RETRIES = int(os.getenv("AI_MIN_TOKENS_RETRIES", "10"))
LLM_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", "1.0"))
console = Console()


def response_is_empty(response: AIMessage) -> bool:
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    if has_tool_calls:
        return False
    content = response.content
    if isinstance(content, str):
        return len(content.split()) < LLM_MIN_TOKENS
    return False


def llm_node(state: AgentState) -> dict:
    try:
        model = build_model()
        for attempt in range(LLM_MAX_RETRIES):
            if attempt > 0:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                console.print(
                    f"[red]LLM returned empty response, retrying (attempt {attempt}/{LLM_MAX_RETRIES - 1}, delay {delay:.1f}s)...[/red]")
                time.sleep(delay)
            response = model.invoke(state["messages"])
            if not response_is_empty(response):
                return {"messages": [response]}
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


tool_node = ToolNode(TOOLS)


def after_llm(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content.startswith("LLM error:"):
        return "human_input_node"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return "human_input_node"


def after_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return "llm_node"
    return "human_input_node"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("human_input_node", human_input_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "human_input_node")
    graph.add_edge("human_input_node", "llm_node")

    graph.add_conditional_edges(
        "llm_node",
        after_llm,
        {"tool_node": "tool_node", "human_input_node": "human_input_node"},
    )

    graph.add_conditional_edges(
        "tool_node",
        after_tool,
        {"human_input_node": "human_input_node", "llm_node": "llm_node"},
    )

    return graph.compile()


career_graph = build_graph()
