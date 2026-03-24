from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Optional
from tool.tools import fetchWebPageContent, saveOffer, getActiveOffer, getOfferById, searchOffers
from llmUtils import buildChatModel

load_dotenv()

TOOLS = [fetchWebPageContent, searchOffers,
         getOfferById, getActiveOffer, saveOffer]


class OfferResolutionResponse(BaseModel):
    message: str = Field(description="Short summary for the user.")
    offerId: Optional[int] = Field(
        default=None, description="Offer id that was found or created.")
    resolution: str = Field(
        description="One of found, created, none, or error.")
    status: Optional[str] = Field(
        default=None, description="Offer status when known.")


SYSTEM_PROMPT = "You manage internship and job offers. Always search existing offers first using searchOffers before creating a new one. If user asks for an existing offer, retrieve its id with searchOffers then confirm details with getOfferById. If user provides a URL, call fetchWebPageContent once, then searchOffers with that content or URL. Only call saveOffer when no existing offer matches. Your final response must always include offerId when an offer is found or created. Use resolution=found when reusing an existing offer, resolution=created when saving a new one, resolution=none when no offer context exists, and resolution=error on failure. Never claim a save or lookup succeeded unless tool output confirms it. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."


def buildGraph():
    model = buildChatModel()
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT, response_format=OfferResolutionResponse)


offerGraph = buildGraph()
