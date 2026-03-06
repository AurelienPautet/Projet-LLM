import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from rich.prompt import Prompt
from typing_extensions import TypedDict

from tools import add_experience

load_dotenv()

TOOLS = [add_experience]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    ).bind_tools(TOOLS)


def human_input_node(state: AgentState) -> dict:
    question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
    return {"messages": [HumanMessage(content=question)]}


def llm_node(state: AgentState) -> dict:
    model = _build_model()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


def after_llm(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return "human_input_node"


def after_tool(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return END
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
        {"human_input_node": "human_input_node", END: END},
    )

    return graph.compile()


career_graph = build_graph()
