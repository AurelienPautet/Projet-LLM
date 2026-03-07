import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from rich.prompt import Prompt

from tools import addExperience, searchExperiences, editExperience
from base_graph import BaseState
from llm_utils import buildModel, invokeModelWithRetries

load_dotenv()

TOOLS = [addExperience, searchExperiences, editExperience]


def humanInputNode(state: BaseState) -> Command:
    question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
    return Command(
        goto="llmNode",
        update={
            "messages": [HumanMessage(content=question)],
            "status": "Thinking..."
        }
    )


def llmNode(state: BaseState) -> dict:
    try:
        model = buildModel(TOOLS)
        response = invokeModelWithRetries(model, state["messages"])
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


def afterLlmNode(state: BaseState) -> Command:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content.startswith("LLM error:"):
        return Command(goto="humanInputNode", update={"status": ""})

    if getattr(last, "tool_calls", None):
        return Command(goto="toolNode", update={"status": "Running tools..."})

    return Command(goto="humanInputNode", update={"status": ""})


def afterToolNode(state: BaseState) -> Command:
    last = state["messages"][-1]

    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return Command(goto="llmNode", update={"status": "Analysing tool output..."})

    return Command(goto="llmNode", update={"status": "Handling error..."})


def buildGraph() -> StateGraph:
    graph = StateGraph(BaseState)

    graph.add_node("humanInputNode", humanInputNode)
    graph.add_node("llmNode", llmNode)
    graph.add_node("toolNode", ToolNode(TOOLS))

    graph.add_node("afterLlmNode", afterLlmNode)
    graph.add_node("afterToolNode", afterToolNode)

    graph.add_edge(START, "humanInputNode")
    graph.add_edge("llmNode", "afterLlmNode")
    graph.add_edge("toolNode", "afterToolNode")

    return graph.compile()


career_graph = buildGraph()
