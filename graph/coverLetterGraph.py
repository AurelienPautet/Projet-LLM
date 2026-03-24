import os
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

from tool.tools import searchExperiences, getAllExperiences, getExperienceCount, getPersonalInfo, getAllPersonalInfo, generatePdfFromLatex,  saveOfferRecord, getOfferByIdRecord, updateOfferCoverLetterOutputRecord
from graph.baseGraph import BaseState
from llmUtils import buildChatModel, extractStructuredOutput, invokeStructuredAgentWithEnforcedResponseTool, invokeStructuredAgent, formatLlmError

load_dotenv()


class CoverLetterState(BaseState):
    writerOutput: Optional[dict] = None
    structured_response: Optional[dict] = None
    internshipOfferText: Optional[str] = None
    internshipOfferSource: Optional[str] = None
    questionAskerHasRun: bool = False
    activeOfferId: Optional[int] = None


class CoverLetterWriterResponse(BaseModel):
    message: str = Field(
        description="A summary of the cover letter writing process, to display to the user."
    )
    coverLetter: Optional[str] = Field(
        description="The generated cover letter text. This can be None if generation failed."
    )


class QuestionAskerResponse(BaseModel):
    message: str = Field(
        description="Exactly three specific questions for the user, tailored to the offer and focused on motivation and key points to insist on."
    )


COVER_LETTER_WRITER_PROMPT = "You are an expert cover letter writer. Build or improve a personalized cover letter for the user using available experience and personal information data. Tailor the letter to the job offer when provided by the user. Use tools when retrieval, counting, or lookup is needed. Never claim a tool-backed action succeeded unless a tool result confirms it. If a tool returns an error, explain it and ask one concise next step. Keep the user-facing message concise and do not include the full cover letter in the message field. Put the complete letter in the coverLetter field. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."
PDF_GENERATOR_PROMPT = "You are a PDF generation agent. You receive cover letter text only. Build a valid standalone one-page LaTeX document using article class. Keep layout professional and concise. Escape LaTeX special characters in all text values: \\, &, %, $, #, _, {, }, ~, ^. Never call the PDF tool more than once. Call generatePdfFromLatex exactly once. For the `outputName` argument, use the exact file name provided by the user in the prompt. If tool returns an error, report it clearly in one short sentence. After the tool call, provide one concise final sentence with the generated PDF path when successful. Always return a message and never leave it empty. Return structured output only."
QUESTION_ASKER_PROMPT = "You are a strategic cover letter interviewer. Generate exactly three concise and specific questions that help understand why the user applies and what they want to insist on in the letter. Questions must be specific to the target offer. Use available tools to retrieve offer context, experiences, and personal information when needed. If no offer context exists, ask one of the three questions to request the offer and keep the other two focused on motivation and emphasis. Output only the questions in the message field, numbered 1 to 3, and nothing else. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."

AGENT_RECURSION_LIMIT = int(os.getenv("AI_AGENT_RECURSION_LIMIT", "80"))
PDF_AGENT_MAX_RETRIES = int(os.getenv("AI_PDF_MAX_RETRIES", "1"))
PDF_AGENT_RECURSION_LIMIT = int(os.getenv("AI_PDF_AGENT_RECURSION_LIMIT", "6"))


def buildCoverLetterWriterAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[searchExperiences, getAllExperiences,
               getExperienceCount, getPersonalInfo, getAllPersonalInfo],
        system_prompt=COVER_LETTER_WRITER_PROMPT,
        response_format=CoverLetterWriterResponse,
    )


def buildPdfGeneratorAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[generatePdfFromLatex],
        system_prompt=PDF_GENERATOR_PROMPT,
    )


def buildQuestionAskerAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[searchExperiences, getAllExperiences,
               getExperienceCount, getAllPersonalInfo, getPersonalInfo],
        system_prompt=QUESTION_ASKER_PROMPT,
        response_format=QuestionAskerResponse,
    )


coverLetterWriterAgent = buildCoverLetterWriterAgent()
pdfGeneratorAgent = buildPdfGeneratorAgent()
questionAskerAgent = buildQuestionAskerAgent()


def agentNodeQuestionAsker(state: CoverLetterState, config: dict) -> dict:
    try:
        if bool(state.get("questionAskerHasRun", False)):
            return {
                "status": "",
            }

        messages = list(state.get("messages") or [])
        contextMessages = list(messages)
        internshipOfferText = str(
            state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(
            state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            parts = [
                "A job offer context is available. Generate offer-specific questions.",
            ]
            if internshipOfferSource:
                parts.append(f"Source: {internshipOfferSource}")
            parts.append(f"Offer content:\n{internshipOfferText}")
            offer_str = "\n\n[CONTEXT: JOB OFFER]\n" + "\n\n".join(parts) + "\n[/CONTEXT]"
            
            if contextMessages and isinstance(contextMessages[-1], HumanMessage):
                contextMessages[-1] = HumanMessage(content=str(contextMessages[-1].content) + offer_str)
            else:
                contextMessages.append(HumanMessage(content=offer_str))

        result = invokeStructuredAgentWithEnforcedResponseTool(
            questionAskerAgent,
            contextMessages,
            config,
            "QuestionAskerResponse",
            recursionLimit=AGENT_RECURSION_LIMIT,
        )
        allMessages = result.get("messages", [])
        newMessages = allMessages[-1:] if allMessages else []
        questionOutput = extractStructuredOutput(result)
        if not questionOutput:
            questionOutput = {
                "message": "1. What is the main reason you are applying for this role?\n2. Which two experiences should be emphasized for this offer?\n3. What impact or value do you want to insist on in your cover letter?"
            }

        return {
            "messages": newMessages,
            "structured_response": questionOutput,
            "questionAskerHasRun": True,
            "status": "",
        }
    except GraphRecursionError:
        questionOutput = {
            "message": "I could not generate the motivation questions because tool calls exceeded the safety limit. Please retry."
        }
        return {
            "messages": [AIMessage(content=questionOutput["message"])],
            "structured_response": questionOutput,
            "questionAskerHasRun": True,
            "status": "",
        }
    except Exception as exc:
        questionOutput = {
            "message": formatLlmError(exc)
        }
        return {
            "messages": [AIMessage(content=questionOutput["message"])],
            "structured_response": questionOutput,
            "questionAskerHasRun": True,
            "status": "",
        }


def agentNodeCoverLetterWriter(state: CoverLetterState, config: dict) -> dict:
    try:
        inputMessages = list(state["messages"])
        internshipOfferText = str(
            state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(
            state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            offerContextParts = [
                "A job offer is provided below. Tailor the cover letter to this offer while staying truthful to known experiences."
            ]
            if internshipOfferSource:
                offerContextParts.append(f"Source: {internshipOfferSource}")
            offerContextParts.append(f"Offer content:\n{internshipOfferText}")
            offer_str = "\n\n[CONTEXT: JOB OFFER]\n" + "\n\n".join(offerContextParts) + "\n[/CONTEXT]"
            
            if inputMessages and isinstance(inputMessages[-1], HumanMessage):
                inputMessages[-1] = HumanMessage(content=str(inputMessages[-1].content) + offer_str)
            else:
                inputMessages.append(HumanMessage(content=offer_str))

        result = invokeStructuredAgentWithEnforcedResponseTool(
            coverLetterWriterAgent,
            inputMessages,
            config,
            "CoverLetterWriterResponse",
            recursionLimit=AGENT_RECURSION_LIMIT,
        )
        allMessages = result.get("messages", [])
        newMessages = allMessages[-1:] if allMessages else []
        writerOutput = extractStructuredOutput(result)
        if not writerOutput:
            writerOutput = {
                "message": "I could not produce a structured cover letter response. Please retry.",
                "coverLetter": None,
            }
        activeOfferId = state.get("activeOfferId")
        if activeOfferId and writerOutput.get("coverLetter"):
            try:
                updateOfferCoverLetterOutputRecord(
                    offerId=int(activeOfferId),
                    coverLetterOutput=str(
                        writerOutput.get("coverLetter") or ""),
                )
            except Exception:
                pass
        return {
            "messages": newMessages,
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": "",
        }
    except GraphRecursionError:
        writerOutput = {
            "message": "I could not finish cover letter generation because tool calls exceeded the safety limit. Please check database and model availability, then retry.",
            "coverLetter": None,
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
            "coverLetter": None,
        }
        return {
            "messages": [AIMessage(content=writerOutput["message"])],
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": "",
        }


def edgeNodeCoverLetterWriterToPdfGenerator(state: CoverLetterState) -> dict:
    return {"status": "Generating final cover letter PDF..."}


def agentNodePdfGenerator(state: CoverLetterState) -> dict:
    writerOutput = state.get("writerOutput") or {}
    coverLetterText = writerOutput.get("coverLetter")
    if not coverLetterText:
        return {
            "messages": [AIMessage(content="PDF generation skipped because no cover letter draft is available.")],
            "status": "",
        }

    maxAttempts = max(1, PDF_AGENT_MAX_RETRIES + 1)
    message = ""
    previousError = ""

    versionSuffix = "1"
    activeOfferId = state.get("activeOfferId")
    if activeOfferId:
        try:
            dbOffer = getOfferByIdRecord(int(activeOfferId))
            if dbOffer and hasattr(dbOffer, "coverLetterVersion"):
                versionSuffix = str(dbOffer.coverLetterVersion)
        except Exception:
            pass

    for attempt in range(maxAttempts):
        try:
            promptText = (
                f"Generate PDF from this cover letter text only, and use a concise `outputName` derived from the text (e.g. 'cover_letter_company_role_v{versionSuffix}') without the .pdf extension:\n\n{coverLetterText}"
            )
            if previousError:
                promptText = (
                    f"Previous attempt failed with: {previousError}\n"
                    "Retry now with safer LaTeX and ASCII-only text content.\n"
                    + promptText
                )
            result = invokeStructuredAgent(
                pdfGeneratorAgent,
                {"messages": [HumanMessage(content=promptText)]},
                {"recursion_limit": PDF_AGENT_RECURSION_LIMIT},
                recursionLimit=PDF_AGENT_RECURSION_LIMIT,
            )
            allMessages = result.get("messages", [])
            newMessages = allMessages[-1:] if allMessages else []
            message = ""
            if newMessages and isinstance(newMessages[0], AIMessage):
                message = str(newMessages[0].content or "").strip()
            if not message:
                message = "PDF generation failed."

            loweredMessage = message.lower()
            if "error" not in loweredMessage and "failed" not in loweredMessage:
                return {
                    "messages": [AIMessage(content=message)],
                    "status": "",
                }

            previousError = message
            if attempt == maxAttempts - 1:
                break
        except Exception as exc:
            previousError = formatLlmError(exc)
            if attempt == maxAttempts - 1:
                message = previousError
                break

    if not message:
        message = previousError or "PDF generation failed."

    return {
        "messages": [AIMessage(content=message)],
        "status": "",
    }


def buildGraph() -> StateGraph:
    def routeAfterQuestionAsker(state: CoverLetterState) -> str:
        if bool(state.get("questionAskerHasRun", False)) and len(state.get("messages", [])) > 1:
            return "coverLetterWriter"
        return "human"

    def routeAfterCoverLetterWriter(state: CoverLetterState) -> str:
        writerOutput = state.get("writerOutput") or {}
        if writerOutput.get("coverLetter"):
            return "pdfGenerator"
        return "human"

    graph = StateGraph(CoverLetterState)
    graph.add_node("agentNodeQuestionAsker", lambda state,
                   config: agentNodeQuestionAsker(state, config))
    graph.add_node("agentNodeCoverLetterWriter", lambda state,
                   config: agentNodeCoverLetterWriter(state, config))
    graph.add_node("edgeNodeCoverLetterWriterToPdfGenerator", lambda state,
                   config: edgeNodeCoverLetterWriterToPdfGenerator(state))
    graph.add_node("agentNodePdfGenerator", lambda state,
                   config: agentNodePdfGenerator(state))
    graph.add_edge(START, "agentNodeQuestionAsker")
    graph.add_conditional_edges(
        "agentNodeQuestionAsker",
        routeAfterQuestionAsker,
        {
            "human": END,
            "coverLetterWriter": "agentNodeCoverLetterWriter",
        },
    )
    graph.add_conditional_edges(
        "agentNodeCoverLetterWriter",
        routeAfterCoverLetterWriter,
        {
            "pdfGenerator": "edgeNodeCoverLetterWriterToPdfGenerator",
            "human": END,
        },
    )
    graph.add_edge("edgeNodeCoverLetterWriterToPdfGenerator",
                   "agentNodePdfGenerator")
    graph.add_edge("agentNodePdfGenerator", END)
    return graph.compile()


coverLetterGraph = buildGraph()
