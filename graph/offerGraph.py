import os
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from graph.baseGraph import BaseState
from tool.tools import fetchWebPageContent, saveOfferRecord, getActiveOfferRecord
from llmUtils import buildChatModel, invokeStructuredAgent, extractStructuredOutput, formatLlmError

load_dotenv()


class OfferState(BaseState):
    structured_response: Optional[dict] = None
    activeOfferId: Optional[int] = None
    offerText: Optional[str] = None
    offerSource: Optional[str] = None


class OfferIntakeResponse(BaseModel):
    message: str = Field(
        description="Short result message about offer intake.")
    hasOffer: bool = Field(
        description="True if an offer was found in user input.")
    offerType: str = Field(
        description="Detected offer type: none, text, or url.")
    offerText: Optional[str] = Field(
        default=None, description="Offer content as plain text.")
    offerSource: Optional[str] = Field(
        default=None, description="Offer source identifier or URL.")


OFFER_INTAKE_PROMPT = "You analyze the latest user request and detect whether it contains a job or internship offer. If there is a URL to an offer page, call fetchWebPageContent exactly once and use the fetched content as offerText. If there is plain offer text, extract it directly as offerText. If there is no offer, set hasOffer to false and offerType to none. Keep message concise. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."
AGENT_RECURSION_LIMIT = int(os.getenv("AI_AGENT_RECURSION_LIMIT", "80"))


def buildOfferIntakeAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[fetchWebPageContent],
        system_prompt=OFFER_INTAKE_PROMPT,
        response_format=OfferIntakeResponse,
    )


offerIntakeAgent = buildOfferIntakeAgent()


def extractLatestHumanText(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content or "").strip()
    return ""


def extractLatestUrlFromHumanMessages(messages: list) -> str:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        text = str(message.content or "")
        match = re.search(r"https?://\S+", text)
        if match:
            return match.group(0).rstrip(".,;)")
    return ""


def agentNodeOfferIntake(state: OfferState, config: dict) -> dict:
    try:
        allMessages = list(state.get("messages") or [])
        userText = extractLatestHumanText(allMessages)
        latestUrl = extractLatestUrlFromHumanMessages(allMessages)

        intakeText = userText
        if latestUrl and latestUrl not in intakeText:
            intakeText = f"{intakeText}\n\nPreviously shared offer URL: {latestUrl}".strip(
            )

        result = invokeStructuredAgent(
            offerIntakeAgent,
            {"messages": [HumanMessage(content=intakeText)]},
            config,
            recursionLimit=AGENT_RECURSION_LIMIT,
        )
        intakeOutput = extractStructuredOutput(result)

        if intakeOutput and bool(intakeOutput.get("hasOffer", False)):
            offerText = str(intakeOutput.get("offerText") or "").strip()
            offerSource = str(intakeOutput.get(
                "offerSource") or "").strip() or None
            if offerText:
                dbOffer = saveOfferRecord(
                    offerText=offerText, offerSource=offerSource)
                payload = {
                    "message": f"Offer saved with id={dbOffer.id} and status={dbOffer.status.value}.",
                    "offerId": dbOffer.id,
                    "status": dbOffer.status.value,
                }
                return {
                    "messages": [AIMessage(content=payload["message"])],
                    "structured_response": payload,
                    "activeOfferId": dbOffer.id,
                    "offerText": dbOffer.offerText,
                    "offerSource": dbOffer.offerSource,
                    "status": "",
                }

        activeOffer = getActiveOfferRecord()
        if activeOffer is None:
            payload = {
                "message": "No offer found. Share an offer URL or paste the offer text.",
                "offerId": None,
                "status": "none",
            }
            return {
                "messages": [AIMessage(content=payload["message"])],
                "structured_response": payload,
                "status": "",
            }

        payload = {
            "message": f"No new offer detected. Keeping active offer id={activeOffer.id} with status={activeOffer.status.value}.",
            "offerId": activeOffer.id,
            "status": activeOffer.status.value,
        }
        return {
            "messages": [AIMessage(content=payload["message"])],
            "structured_response": payload,
            "activeOfferId": activeOffer.id,
            "offerText": activeOffer.offerText,
            "offerSource": activeOffer.offerSource,
            "status": "",
        }
    except Exception as exc:
        payload = {"message": formatLlmError(
            exc), "offerId": None, "status": "error"}
        return {
            "messages": [AIMessage(content=payload["message"])],
            "structured_response": payload,
            "status": "",
        }


def buildGraph() -> StateGraph:
    graph = StateGraph(OfferState)
    graph.add_node("agentNodeOfferIntake", lambda state,
                   config: agentNodeOfferIntake(state, config))
    graph.add_edge(START, "agentNodeOfferIntake")
    graph.add_edge("agentNodeOfferIntake", END)
    return graph.compile()


offerGraph = buildGraph()
