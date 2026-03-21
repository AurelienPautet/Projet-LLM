import json
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

from tool.tools import addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile
from graph.baseGraph import BaseState
from llmUtils import buildChatModel

load_dotenv()

TOOLS = [addExperience, searchExperiences,
         editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile]


class CVState(BaseState):
    writerOutput: Optional[dict] = None
    reviewerOutput: Optional[dict] = None
    structured_response: Optional[dict] = None


class CvWriterResponse(BaseModel):
    message: str = Field(
        description="A summary of the CV writing process, that will be displayed to the user. If the CV generation failed, this should contain an explanation.")
    cv: Optional[str] = Field(
        description="The generated CV in text format. This can be None if the agent failed to generate a CV, in which case the message field should contain an explanation.")


class AtsReviewerResponse(BaseModel):
    message: str = Field(
        description="A summary of the ATS review results, that will be displayed to the user. Write the ats score in the message.")
    ats: int = Field(
        ge=0, le=100, description="The ATS compatibility score (0-100).")
    feedback: str = Field(
        description="Actionable feedback to improve ATS compatibility. This can be empty if the review failed, in which case the message field should contain an explanation.")


CV_WRITER_PROMPT = "You are an expert CV writer. Build or improve the user's CV using the available experience database. Use tools when retrieval, persistence, counting, editing, or deletion is needed. Never claim those actions succeeded unless a tool result confirms it. If a tool returns an error, do not retry in a loop. Explain the error and ask one concise next step. Ask concise clarification questions if information is missing. Return structured output only."
ATS_REVIEWER_PROMPT = "You are an ATS reviewer. Evaluate the CV draft for ATS compatibility and provide actionable feedback. Use tools when needed to verify experiences or retrieve missing details. Never claim tool-backed actions succeeded without tool confirmation. If a tool returns an error, do not retry in a loop. Explain the error and provide actionable next steps. Return structured output only."
AGENT_RECURSION_LIMIT = int(os.getenv("AI_AGENT_RECURSION_LIMIT", "8"))


def toDict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {"message": str(value)}


def extractStructuredOutput(result: dict) -> dict:
    structured = result.get("structured_response")
    if structured is None:
        return {}
    return toDict(structured)


def invokeStructuredAgent(agent, inputPayload: dict) -> dict:
    return agent.invoke(inputPayload, config={"recursion_limit": AGENT_RECURSION_LIMIT})


def formatLlmError(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if "429" in text or "rate-limit" in lowered or "rate limited" in lowered or "temporarily rate-limited" in lowered:
        return "The current model provider is rate-limited right now. Please retry in a few seconds."
    return f"LLM error: {text}"


def buildCvWriterAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=CV_WRITER_PROMPT,
        response_format=CvWriterResponse,
    )


def buildAtsReviewerAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=ATS_REVIEWER_PROMPT,
        response_format=AtsReviewerResponse,
    )


cvWriterAgent = buildCvWriterAgent()
atsReviewerAgent = buildAtsReviewerAgent()


def agentNodeCV_writer(state: CVState) -> dict:
    try:
        inputMessages = state["messages"]
        result = invokeStructuredAgent(
            cvWriterAgent, {"messages": inputMessages})
        allMessages = result.get("messages", [])
        newMessages = allMessages[len(inputMessages):]
        writerOutput = extractStructuredOutput(result)
        if not writerOutput:
            writerOutput = {
                "message": "I could not produce a structured CV response. Please retry.",
                "cv": None,
            }
        nextStatus = "Reviewing CV for ATS compatibility..." if writerOutput.get(
            "cv") else ""
        return {
            "messages": newMessages,
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": nextStatus,
        }
    except GraphRecursionError:
        writerOutput = {
            "message": "I could not finish CV generation because tool calls exceeded the safety limit. Please check database/tool availability and try again.",
            "cv": None,
        }
        return {
            "messages": [AIMessage(content=writerOutput["message"])],
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": "",
        }
    except Exception as exc:
        writerOutput = {
            "message": formatLlmError(exc),
            "cv": None,
        }
        return {
            "messages": [AIMessage(content=writerOutput["message"])],
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": "",
        }


def agentNodeATS_reviewer(state: CVState) -> dict:
    try:
        writerOutput = state.get("writerOutput") or {}
        if not writerOutput.get("cv"):
            reviewerOutput = {
                "message": "ATS review skipped because no CV draft is available.",
                "ats": 0,
                "feedback": "Generate a CV draft first, then run ATS review.",
            }
            return {
                "messages": [AIMessage(content=reviewerOutput["message"])],
                "reviewerOutput": reviewerOutput,
                "structured_response": reviewerOutput,
                "status": "",
            }
        reviewerContext = SystemMessage(
            content=f"CV writer output JSON:\n{json.dumps(writerOutput, ensure_ascii=True)}"
        )
        inputMessages = list(state["messages"]) + [reviewerContext]
        result = invokeStructuredAgent(
            atsReviewerAgent, {"messages": inputMessages})
        allMessages = result.get("messages", [])
        newMessages = allMessages[len(inputMessages):]
        reviewerOutput = extractStructuredOutput(result)
        if not reviewerOutput:
            reviewerOutput = {
                "message": "I could not produce a structured ATS response. Please retry.",
                "ats": 0,
                "feedback": "",
            }
        return {
            "messages": newMessages,
            "reviewerOutput": reviewerOutput,
            "structured_response": reviewerOutput,
            "status": "",
        }
    except GraphRecursionError:
        reviewerOutput = {
            "message": "I could not finish ATS review because tool calls exceeded the safety limit.",
            "ats": 0,
            "feedback": "Check database/tool availability and retry.",
        }
        return {
            "messages": [AIMessage(content=reviewerOutput["message"])],
            "reviewerOutput": reviewerOutput,
            "structured_response": reviewerOutput,
            "status": "",
        }
    except Exception as exc:
        reviewerOutput = {
            "message": formatLlmError(exc),
            "ats": 0,
            "feedback": "Retry after checking model and tool configuration.",
        }
        return {
            "messages": [AIMessage(content=reviewerOutput["message"])],
            "reviewerOutput": reviewerOutput,
            "structured_response": reviewerOutput,
            "status": "",
        }


def buildGraph() -> StateGraph:
    def routeAfterCvWriter(state: CVState) -> str:
        writerOutput = state.get("writerOutput") or {}
        if writerOutput.get("cv"):
            return "atsReviewer"
        return "human"

    def routeAfterAtsReviewer(state: CVState) -> str:
        reviewerOutput = state.get("reviewerOutput") or {}
        try:
            atsScore = int(reviewerOutput.get("ats", 0) or 0)
        except Exception:
            atsScore = 0
        if atsScore < 90:
            return "cvWriter"
        return "human"

    graph = StateGraph(CVState)
    graph.add_node("agentNodeCV_writer", agentNodeCV_writer)
    graph.add_node("agentNodeATS_reviewer", agentNodeATS_reviewer)
    graph.add_edge(START, "agentNodeCV_writer")
    graph.add_conditional_edges(
        "agentNodeCV_writer",
        routeAfterCvWriter,
        {
            "atsReviewer": "agentNodeATS_reviewer",
            "human": END,
        },
    )
    graph.add_conditional_edges(
        "agentNodeATS_reviewer",
        routeAfterAtsReviewer,
        {
            "cvWriter": "agentNodeCV_writer",
            "human": END,
        },
    )
    return graph.compile()


cv_graph = buildGraph()
