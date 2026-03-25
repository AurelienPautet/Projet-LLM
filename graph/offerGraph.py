from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from tool.tools import fetchWebPageContent, saveOffer, getOfferById, searchOffers, getOfferBySource
from llmUtils import buildChatModel

load_dotenv()

getOfferByIdTool = tool(getOfferById)

TOOLS = [fetchWebPageContent, searchOffers, getOfferByIdTool, saveOffer, getOfferBySource]


class OfferResolutionResponse(BaseModel):
    message: str = Field(description="Short summary for the user.")
    offerId: Optional[int] = Field(default=None, description="Offer id that was found or created.")


SYSTEM_PROMPT = "You manage internship and job offers. Always search existing offers first using searchOffers before creating a new one. If user asks for an existing offer, retrieve its id with searchOffers then confirm details with getOfferById. If user provides a URL, call fetchWebPageContent once, then searchOffers with that content or URL. Only call saveOffer when no existing offer matches. Your final response must always include offerId when an offer is found or created. Never claim a save or lookup succeeded unless tool output confirms it. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only. Always respond with an existing offer, either found or newly created"


def buildGraph():
    model = buildChatModel()
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, response_format=OfferResolutionResponse)


offerGraph = buildGraph()
