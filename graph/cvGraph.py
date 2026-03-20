import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from rich.prompt import Prompt
from sqlmodel import Field
from pydantic import BaseModel

from tool.tools import addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile
from graph.baseGraph import BaseState
from llmUtils import buildModel, invokeAgentWithRetries

load_dotenv()

TOOLS = [addExperience, searchExperiences,
         editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile]


class CVState(BaseState):
    cv: str = ""


def humanInputNode(state: CVState) -> Command:
    question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
    return Command(
        goto="agentNodeCV_writer",
        update={
            "messages": [HumanMessage(content=question)],
            "status": "Thinking..."
        }
    )


class CV_writer_Output(BaseModel):
    cv: Optional[str] = Field(
        description="the generated CV in markdown format")
    message: str = Field(
        description="a message to the user, explaining what was done and what are the next steps and if you dont have enough information to create a cv ask for more details")


def agentNodeCV_writer(state: CVState) -> dict:
    try:
        model = buildModel(TOOLS)
        systemPrompt = "You are an expert CV writer. Your goal is to help the user create a CV that highlights their professional experiences and skills in the best possible way. You can ask the user for details about their experiences (title, description, start and end dates, company, location, technologies used) and use the addExperience tool to save them in the database. You can also load experiences from a CV file using the loadCvFromFile tool if the user provides a file path. Once you have enough information, create a well-structured CV that effectively showcases the user's background and expertise."
        response = invokeAgentWithRetries(
            model, systemPrompt, state["messages"], schema=CV_writer_Output)
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


class ATS_critique_Output(BaseModel):
    ats: int = Field(
        description="the ATS compatibility score of the CV, between 0 and 100")
    feedback: str = Field(
        description="detailed feedback on how to improve the CV for better ATS compatibility, including keyword optimization, formatting, and ensuring all relevant information is included in a way that ATS can easily parse")
    message: str = Field(
        description="a short messsage to the user giving the ats score and a summary of the feedback")


def agentNodeATS_critique(state: CVState) -> dict:
    try:
        model = buildModel(TOOLS)
        systemPrompt = "You are an expert ATS (Applicant Tracking System) critic. Your goal is to analyze the user's CV and provide feedback on how to improve it for better compatibility with ATS systems. Focus on keyword optimization, formatting, and ensuring all relevant information is included in a way that ATS can easily parse."
        response = invokeAgentWithRetries(
            model, systemPrompt, state["messages"], schema=ATS_critique_Output)
        return {"messages": [response], "ats_score": response.ats}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


def afterCV_writer(state: CVState) -> Command:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content.startswith("LLM error:"):
        return Command(goto="agentNodeCV_writer", update={"status": "Retrying..."})

    if getattr(last, "tool_calls", None):
        return Command(goto="toolNode", update={"status": "Running tools..."})

    if (last.cv and len(last.cv) > 0):
        return Command(goto="agentNodeATS_critique", update={"status": "Reviewing CV for ATS compatibility..."})
    return Command(goto="humanInputNode", update={"status": ""})


def afterATS_critique(state: CVState) -> Command:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content.startswith("LLM error:"):
        return Command(goto="agentNodeATS_critique", update={"status": "Retrying..."})

    if (last.ats < 80):
        return Command(goto="agentNodeCV_writer", update={"status": "Improving CV based on feedback..."})
    return Command(goto="humanInputNode", update={"status": ""})


def afterToolNode(state: CVState) -> Command:
    last = state["messages"][-1]

    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return Command(goto="agentNodeCV_writer", update={"status": "Analysing tool output..."})

    return Command(goto="agentNodeCV_writer", update={"status": "Handling error..."})


def buildGraph() -> StateGraph:
    graph = StateGraph(CVState)

    graph.add_node("humanInputNode", humanInputNode)
    graph.add_node("agentNodeCV_writer", agentNodeCV_writer)
    graph.add_node("agentNodeATS_critique", agentNodeATS_critique)
    graph.add_node("toolNode", ToolNode(TOOLS))

    graph.add_node("afterCV_writer", afterCV_writer)
    graph.add_node("afterATS_critique", afterATS_critique)
    graph.add_node("afterToolNode", afterToolNode)

    graph.add_edge(START, "humanInputNode")
    graph.add_edge("agentNodeCV_writer", "afterCV_writer")
    graph.add_edge("agentNodeATS_critique", "afterATS_critique")
    graph.add_edge("toolNode", "afterToolNode")

    return graph.compile()


cv_graph = buildGraph()
