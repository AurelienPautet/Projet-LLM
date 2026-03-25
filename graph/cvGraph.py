from langchain_core.runnables import RunnableConfig
import json
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from tool.tools import searchExperiences, getAllExperiences, getExperienceCount, generatePdfFromLatex, getPersonalInfo, getAllPersonalInfo, getOfferById, updateOfferCvOutput
from graph.baseGraph import BaseState
from llmUtils import buildChatModel, extractStructuredOutput, invokeStructuredAgent, invokeStructuredAgentWithEnforcedResponseTool, formatLlmError, handleNodeError

load_dotenv()

AGENT_RECURSION_LIMIT = 80
PDF_AGENT_RECURSION_LIMIT = 6
ATS_MAX_IMPROVEMENT_CYCLES = 4


class CVState(BaseState):
    writerOutput: Optional[dict] = None
    reviewerOutput: Optional[dict] = None
    structured_response: Optional[dict] = None
    atsIterationCount: int = 0
    lastAtsScore: Optional[int] = None
    previousAtsScore: Optional[int] = None
    internshipOfferText: Optional[str] = None
    internshipOfferSource: Optional[str] = None
    activeOfferId: Optional[int] = None


class CvWriterResponse(BaseModel):
    message: str = Field(
        description="A summary of the CV writing process, that will be displayed to the user. If the CV generation failed, this should contain an explanation.")
    cv: Optional[str] = Field(
        description="The generated CV in text format. This can be None if the agent failed to generate a CV, in which case the message field should contain an explanation.")


class AtsReviewerResponse(BaseModel):
    message: str = Field(
        description="A very short summary of the ATS review results, that will be displayed to the user. Write the ats score in the message.")
    ats: int = Field(
        ge=0, le=100, description="The ATS compatibility score (0-100).")
    feedback: str = Field(
        description="Actionable feedback to improve ATS compatibility. This can be empty if the review failed, in which case the message field should contain an explanation.")


CV_WRITER_PROMPT = "You are an expert CV writer. Build or improve the user's CV using the available experience and personal information database. Use tools when retrieval, persistence, counting, editing, or deletion is needed. Never claim those actions succeeded unless a tool result confirms it. If a tool returns an error, do not retry in a loop. Explain the error and ask one concise next step. Ask concise clarification questions if information is missing. Retrieve personal information with getAllPersonalInfo or getPersonalInfo when needed to build the CV header and contact section. When ATS feedback is provided in context, apply it directly by rewriting the CV draft with concrete improvements using only known facts, and avoid repeating the ATS report verbatim. Never include the full CV in your user-facing message: only return the summary in the message field, and put the CV text in the cv field. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."
ATS_REVIEWER_PROMPT = "You are an ATS reviewer. Evaluate the CV draft for ATS compatibility and provide actionable feedback. If a job offer is provided in context, evaluate alignment with this offer and include concrete ATS-oriented adjustments to match keywords and requirements. Write your response in the same language as the CV content. Use tools when needed to verify experiences or retrieve missing details. Never claim tool-backed actions succeeded without tool confirmation. If a tool returns an error, do not retry in a loop. Explain the error and provide actionable next steps. You must finish by calling the response tool for the structured schema exactly once as your final action. Return structured output only."
PDF_GENERATOR_PROMPT = "You are a PDF generation agent. You receive CV text only. Build a valid standalone one-page LaTeX CV using moderncv only: documentclass moderncv, moderncvstyle classic, moderncvcolor blue. Do not use optional icon packages or uncommon dependencies. Keep sections concise so output stays on one page. Use a safe structure only: preamble, begin document, makecvtitle, sections, end document. Use only \\section and \\cvitem for content rows. Do not use \\cventry, tabular, itemize, enumerate, custom macros, nested environments, or multiline command arguments. Escape LaTeX special characters in all text values: \\, &, %, $, #, _, {, }, ~, ^. Never include blank lines inside a command argument. Never call the PDF tool more than once. Call generatePdfFromLatex exactly once with the requested `outputName`. If tool returns an error, report it clearly in one short sentence. After the tool call, provide one concise final sentence with the generated PDF path when successful. Always return a message and never leave it empty. Return structured output only. Answer as quickly as possible, do not explain your reasoning, do not reflect, just output the LaTeX and call the tool immediately."


def buildCvWriterAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[searchExperiences, getAllExperiences,
               getExperienceCount, getPersonalInfo, getAllPersonalInfo],
        system_prompt=CV_WRITER_PROMPT,
        response_format=CvWriterResponse,
    )


def buildAtsReviewerAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[],
        system_prompt=ATS_REVIEWER_PROMPT,
        response_format=AtsReviewerResponse,
    )


def buildPdfGeneratorAgent():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[generatePdfFromLatex],
        system_prompt=PDF_GENERATOR_PROMPT,
    )


cvWriterAgent = buildCvWriterAgent()
atsReviewerAgent = buildAtsReviewerAgent()
pdfGeneratorAgent = buildPdfGeneratorAgent()


def agentNodeLoad_Offer(state: CVState, config: RunnableConfig) -> dict:
    offerId = state.get("activeOfferId")
    if not offerId:
        return {
            "internshipOfferText": None,
            "internshipOfferSource": None,
            "status": "",
        }
    try:
        dbOffer = getOfferById(int(offerId))
        if dbOffer is None:
            return {
                "messages": [AIMessage(content=f"Offer id={offerId} was not found. Continuing without offer context.")],
                "internshipOfferText": None,
                "internshipOfferSource": None,
                "activeOfferId": None,
                "status": "",
            }
        return {
            "internshipOfferText": dbOffer.offerText,
            "internshipOfferSource": dbOffer.offerSource,
            "activeOfferId": dbOffer.id,
            "status": "",
        }
    except Exception as exc:
        return {
            "messages": [AIMessage(content=formatLlmError(exc))],
            "internshipOfferText": None,
            "internshipOfferSource": None,
            "activeOfferId": None,
            "status": "",
        }


def buildOfferContext(offerText: str, offerSource: str, intro: str) -> str:
    parts = [intro]
    if offerSource:
        parts.append(f"Source: {offerSource}")
    parts.append(f"Offer content:\n{offerText}")
    return "\n\n[CONTEXT: JOB OFFER]\n" + "\n\n".join(parts) + "\n[/CONTEXT]"


def appendToLastMessage(messages: list, extra: str) -> list:
    if messages and isinstance(messages[-1], HumanMessage):
        messages[-1] = HumanMessage(content=str(messages[-1].content) + extra)
    else:
        messages.append(HumanMessage(content=extra))
    return messages


def agentNodeCV_Writer(state: CVState, config: RunnableConfig) -> dict:
    try:
        writerMessages = list(state["messages"][:1])
        internshipOfferText = str(state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            offerStr = buildOfferContext(internshipOfferText, internshipOfferSource, "A job offer is provided below. Tailor the CV to this offer while staying truthful to known experiences.")
            writerMessages = appendToLastMessage(writerMessages, offerStr)
        reviewerOutput = state.get("reviewerOutput") or {}
        try:
            reviewerAts = int(reviewerOutput.get("ats", 0) or 0)
        except Exception:
            reviewerAts = 0
        if reviewerOutput and reviewerAts < 80:
            revisionContext = (
                "\n\n[ATS REVIEW FEEDBACK]\n"
                "You are now revising the previous CV draft after ATS review. "
                "Improve the CV content itself first. Keep the summary short. "
                "Do not paste ATS feedback bullets in your message. "
                f"ATS reviewer output JSON:\n{json.dumps(reviewerOutput, ensure_ascii=True)}"
                "\n[/ATS REVIEW FEEDBACK]"
            )
            writerMessages = appendToLastMessage(writerMessages, revisionContext)
        result = invokeStructuredAgentWithEnforcedResponseTool(
            cvWriterAgent,
            writerMessages,
            config,
            "CvWriterResponse",
            recursionLimit=AGENT_RECURSION_LIMIT,
        )
        allMessages = result.get("messages", [])
        newMessages = allMessages[-1:] if allMessages else []
        writerOutput = extractStructuredOutput(result)
        if not writerOutput:
            writerOutput = {
                "message": "I could not produce a structured CV response. Please retry.",
                "cv": None,
            }
        activeOfferId = state.get("activeOfferId")
        if activeOfferId and writerOutput.get("cv"):
            try:
                updateOfferCvOutput(
                    offerId=int(activeOfferId),
                    cvOutput=str(writerOutput.get("cv") or ""),
                )
            except Exception:
                pass
        return {
            "messages": newMessages,
            "writerOutput": writerOutput,
            "structured_response": writerOutput,
            "status": "",
        }
    except Exception as exc:
        return handleNodeError(exc, "writerOutput", {"cv": None})


def agentNodeATS_Reviewer(state: CVState, config: RunnableConfig) -> dict:
    try:
        writerOutput = state.get("writerOutput") or {}
        previousAtsScore = state.get("lastAtsScore")
        previousIterationCount = int(state.get("atsIterationCount", 0) or 0)
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
                "atsIterationCount": previousIterationCount,
                "lastAtsScore": previousAtsScore,
                "previousAtsScore": previousAtsScore,
                "status": "",
            }
        reviewerContextStr = f"CV writer output JSON:\n{json.dumps(writerOutput, ensure_ascii=True)}"
        internshipOfferText = str(state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            reviewerContextStr += buildOfferContext(internshipOfferText, internshipOfferSource, "Internship offer context is provided for ATS alignment analysis.")
        
        formatted_messages = [{"role": "user", "content": reviewerContextStr}]
        result = invokeStructuredAgentWithEnforcedResponseTool(
            atsReviewerAgent,
            formatted_messages,
            config,
            "AtsReviewerResponse",
            recursionLimit=AGENT_RECURSION_LIMIT,
        )
        allMessages = result.get("messages", [])
        newMessages = allMessages[-1:] if allMessages else []
        reviewerOutput = extractStructuredOutput(result)
        if not reviewerOutput:
            reviewerOutput = {
                "message": "I could not produce a structured ATS response. Please retry.",
                "ats": 0,
                "feedback": "",
            }
        try:
            currentAtsScore = int(reviewerOutput.get("ats", 0) or 0)
        except Exception:
            currentAtsScore = 0
        return {
            "messages": newMessages,
            "reviewerOutput": reviewerOutput,
            "structured_response": reviewerOutput,
            "atsIterationCount": previousIterationCount + 1,
            "lastAtsScore": currentAtsScore,
            "previousAtsScore": previousAtsScore,
            "status": "",
        }
    except Exception as exc:
        return handleNodeError(exc, "reviewerOutput", {"ats": 0, "feedback": ""})


def agentNodePdf_Generator(state: CVState, config: RunnableConfig) -> dict:
    try:
        writerOutput = state.get("writerOutput") or {}
        cvText = writerOutput.get("cv")
        if not cvText:
            return {
                "messages": [AIMessage(content="PDF generation skipped because no CV draft is available.")],
                "status": "",
            }

        versionSuffix = "1"
        activeOfferId = state.get("activeOfferId")
        if activeOfferId:
            try:
                dbOffer = getOfferById(int(activeOfferId))
                if dbOffer and hasattr(dbOffer, "cvVersion"):
                    versionSuffix = str(dbOffer.cvVersion)
            except Exception:
                pass

        promptText = f"Generate PDF from this CV text only, and use a concise `outputName` derived from the target role (e.g. 'cv_company_role_v{versionSuffix}') without the .pdf extension:\n\n{cvText}"
        result = invokeStructuredAgent(
            pdfGeneratorAgent,
            {"messages": [HumanMessage(content=promptText)]},
            {"recursion_limit": PDF_AGENT_RECURSION_LIMIT},
            recursionLimit=PDF_AGENT_RECURSION_LIMIT,
        )
        msgs = result.get("messages", [])
        message = str(msgs[-1].content or "").strip() if msgs and isinstance(msgs[-1], AIMessage) else ""
        message = message or "PDF generation failed."
        return {
            "messages": [AIMessage(content=message)],
            "status": "",
        }
    except Exception as exc:
        return handleNodeError(exc)

def edgeNodeCvWriterToAtsReviewer(state: CVState) -> dict:
    return {"status": "Reviewing CV for ATS compatibility..."}


def edgeNodeAtsReviewerToCvWriter(state: CVState) -> dict:
    return {"status": "Improving CV based on ATS feedback..."}


def edgeNodeAtsReviewerToPdfGenerator(state: CVState) -> dict:
    return {"status": "Generating final CV PDF..."}


def buildGraph() -> StateGraph:
    def routeAfterCvWriter(state: CVState) -> str:
        writerOutput = state.get("writerOutput") or {}
        if writerOutput.get("cv"):
            return "atsReviewer"
        return "supervisor"

    def routeAfterAtsReviewer(state: CVState) -> str:
        reviewerOutput = state.get("reviewerOutput") or {}
        messageText = str(reviewerOutput.get("message", "")).lower()
        if "could not produce a structured ats response" in messageText:
            return "supervisor"
        previousAtsScore = state.get("previousAtsScore")
        iterationCount = int(state.get("atsIterationCount", 0) or 0)
        try:
            atsScore = int(reviewerOutput.get("ats", 0) or 0)
        except Exception:
            atsScore = 0
        if atsScore < 75:
            if iterationCount >= ATS_MAX_IMPROVEMENT_CYCLES:
                return "pdfGenerator"
            return "cvWriter"
        return "pdfGenerator"

    graph = StateGraph(CVState)
    graph.add_node("agentNodeLoad_Offer", agentNodeLoad_Offer)
    graph.add_node("agentNodeCV_Writer", agentNodeCV_Writer)
    graph.add_node("agentNodeATS_Reviewer", agentNodeATS_Reviewer)
    graph.add_node("edgeNodeCvWriterToAtsReviewer", edgeNodeCvWriterToAtsReviewer)
    graph.add_node("edgeNodeAtsReviewerToCvWriter", edgeNodeAtsReviewerToCvWriter)
    graph.add_node("edgeNodeAtsReviewerToPdfGenerator", edgeNodeAtsReviewerToPdfGenerator)
    graph.add_node("agentNodePdf_Generator", agentNodePdf_Generator)
    graph.add_edge(START, "agentNodeLoad_Offer")
    graph.add_edge("agentNodeLoad_Offer", "agentNodeCV_Writer")
    graph.add_conditional_edges(
        "agentNodeCV_Writer",
        routeAfterCvWriter,
        {
            "atsReviewer": "edgeNodeCvWriterToAtsReviewer",
            "supervisor": END,
        },
    )
    graph.add_edge("edgeNodeCvWriterToAtsReviewer", "agentNodeATS_Reviewer")
    graph.add_conditional_edges(
        "agentNodeATS_Reviewer",
        routeAfterAtsReviewer,
        {
            "cvWriter": "edgeNodeAtsReviewerToCvWriter",
            "pdfGenerator": "edgeNodeAtsReviewerToPdfGenerator",
            "supervisor": END,
        },
    )
    graph.add_edge("edgeNodeAtsReviewerToCvWriter", "agentNodeCV_Writer")
    graph.add_edge("edgeNodeAtsReviewerToPdfGenerator", "agentNodePdf_Generator")
    graph.add_edge("agentNodePdf_Generator", END)
    return graph.compile()


cv_graph = buildGraph()
