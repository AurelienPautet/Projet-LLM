import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from rich.prompt import Prompt

from tool.tools import addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile
from graph.baseGraph import BaseState
from llmUtils import buildModel, invokeAgentWithRetries

load_dotenv()

TOOLS = [addExperience, searchExperiences,
         editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile]


def humanInputNode(state: BaseState) -> Command:
    question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
    return Command(
        goto="agentNodeExperience_manager",
        update={
            "messages": [HumanMessage(content=question)],
            "status": "Thinking..."
        }
    )


def agentNodeExperience_manager(state: BaseState) -> dict:
    try:
        model = buildModel(TOOLS)
        systemPrompt = "You are an expert at collecting and organizing professional experiences. Your goal is to help the user add or modify professional experiences in their database. You can add experiences manually by gathering details from the user (title, description, start and end dates, company, location, technologies used), or load them from a CV file using the loadCvFromFile tool if the user provides a file path. Ask clarifying questions if needed."
        response = invokeAgentWithRetries(
            model, systemPrompt, state["messages"])
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


def afteragentNodeExperience_manager(state: BaseState) -> Command:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content.startswith("LLM error:"):
        return Command(goto="agentNodeExperience_manager", update={"status": ""})

    if getattr(last, "tool_calls", None):
        return Command(goto="toolNode", update={"status": "Running tools..."})

    return Command(goto="humanInputNode", update={"status": ""})


def afterToolNode(state: BaseState) -> Command:
    last = state["messages"][-1]

    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return Command(goto="agentNodeExperience_manager", update={"status": "Analysing tool output..."})

    return Command(goto="agentNodeExperience_manager", update={"status": "Handling error..."})


def buildGraph() -> StateGraph:
    graph = StateGraph(BaseState)

    graph.add_node("humanInputNode", humanInputNode)
    graph.add_node("agentNodeExperience_manager", agentNodeExperience_manager)
    graph.add_node("toolNode", ToolNode(TOOLS))

    graph.add_node("afteragentNodeExperience_manager",
                   afteragentNodeExperience_manager)
    graph.add_node("afterToolNode", afterToolNode)

    graph.add_edge(START, "humanInputNode")
    graph.add_edge("agentNodeExperience_manager",
                   "afteragentNodeExperience_manager")
    graph.add_edge("toolNode", "afterToolNode")

    return graph.compile()


career_graph = buildGraph()
