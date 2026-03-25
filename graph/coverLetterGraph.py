from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from tool.tools import searchExperiences, getAllExperiences, getExperienceCount, getPersonalInfo, getAllPersonalInfo, generatePdfFromLatex, getOfferById, updateOfferCoverLetterOutput
from graph.baseGraph import BaseState
from llmUtils import buildChatModel, extractStructuredOutput, invokeStructuredAgentWithEnforcedResponseTool, invokeStructuredAgent, formatLlmError, handleNodeError

load_dotenv()

AGENT_RECURSION_LIMIT = 80
PDF_AGENT_RECURSION_LIMIT = 6


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


def agentNodeLoad_Offer(state: CoverLetterState, config: RunnableConfig) -> dict:
    offerId = state.get("activeOfferId")
    if not offerId:
        return {
            "internshipOfferText": None,
            "internshipOfferSource": None,
            "status": "Analyzing request to ask motivation questions...",
        }
    try:
        dbOffer = getOfferById(int(offerId))
        if dbOffer is None:
            return {
                "messages": [AIMessage(content=f"Offer id={offerId} was not found. Continuing without offer context.")],
                "internshipOfferText": None,
                "internshipOfferSource": None,
                "activeOfferId": None,
                "status": "Analyzing request to ask motivation questions...",
            }
        return {
            "internshipOfferText": dbOffer.offerText,
            "internshipOfferSource": dbOffer.offerSource,
            "activeOfferId": dbOffer.id,
            "status": "Analyzing offer context to prepare questions...",
        }
    except Exception as exc:
        return {
            "messages": [AIMessage(content=formatLlmError(exc))],
            "internshipOfferText": None,
            "internshipOfferSource": None,
            "activeOfferId": None,
            "status": "Error loading offer, continuing to questions...",
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


def agentNodeQuestion_Asker(state: CoverLetterState, config: RunnableConfig) -> dict:
    try:
        if bool(state.get("questionAskerHasRun", False)):
            return {"status": "Generating cover letter draft..."}

        contextMessages = list(state.get("messages") or [])
        internshipOfferText = str(state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            offerStr = buildOfferContext(internshipOfferText, internshipOfferSource, "A job offer context is available. Generate offer-specific questions.")
            contextMessages = appendToLastMessage(contextMessages, offerStr)

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
    except Exception as exc:
        return {**handleNodeError(exc, "structured_response"), "questionAskerHasRun": True}


def agentNodeCover_Letter_Writer(state: CoverLetterState, config: RunnableConfig) -> dict:
    try:
        inputMessages = list(state["messages"][:1])
        internshipOfferText = str(state.get("internshipOfferText") or "").strip()
        internshipOfferSource = str(state.get("internshipOfferSource") or "").strip()
        if internshipOfferText:
            offerStr = buildOfferContext(internshipOfferText, internshipOfferSource, "A job offer is provided below. Tailor the cover letter to this offer while staying truthful to known experiences.")
            inputMessages = appendToLastMessage(inputMessages, offerStr)

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
                updateOfferCoverLetterOutput(
                    offerId=int(activeOfferId),
                    coverLetterOutput=str(writerOutput.get("coverLetter") or ""),
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
        return handleNodeError(exc, "writerOutput", {"coverLetter": None})


def agentNodePdf_Generator(state: CoverLetterState, config: RunnableConfig) -> dict:
    try:
        writerOutput = state.get("writerOutput") or {}
        coverLetterText = writerOutput.get("coverLetter")
        if not coverLetterText:
            return {
                "messages": [AIMessage(content="PDF generation skipped because no cover letter draft is available.")],
                "status": "",
            }

        versionSuffix = "1"
        activeOfferId = state.get("activeOfferId")
        if activeOfferId:
            try:
                dbOffer = getOfferById(int(activeOfferId))
                if dbOffer and hasattr(dbOffer, "coverLetterVersion"):
                    versionSuffix = str(dbOffer.coverLetterVersion)
            except Exception:
                pass

        promptText = f"Generate PDF from this cover letter text only, and use a concise `outputName` derived from the text (e.g. 'cover_letter_company_role_v{versionSuffix}') without the .pdf extension:\n\n{coverLetterText}"
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


def edgeNodeCoverLetterWriterToPdfGenerator(state: CoverLetterState) -> dict:
    return {"status": "Generating final cover letter PDF..."}



def buildGraph() -> StateGraph:
    def routeAfterQuestionAsker(state: CoverLetterState) -> str:
        if bool(state.get("questionAskerHasRun", False)) and len(state.get("messages", [])) > 1 and (state.get("status", "") == "Generating cover letter draft..."):
            return "coverLetterWriter"
        return "supervisor"

    def routeAfterCoverLetterWriter(state: CoverLetterState) -> str:
        writerOutput = state.get("writerOutput") or {}
        if writerOutput.get("coverLetter"):
            return "pdfGenerator"
        return "supervisor"

    graph = StateGraph(CoverLetterState)
    graph.add_node("agentNodeLoad_Offer", agentNodeLoad_Offer)
    graph.add_node("agentNodeQuestion_Asker", agentNodeQuestion_Asker)
    graph.add_node("agentNodeCover_Letter_Writer", agentNodeCover_Letter_Writer)
    graph.add_node("edgeNodeCoverLetterWriterToPdfGenerator", edgeNodeCoverLetterWriterToPdfGenerator)
    graph.add_node("agentNodePdf_Generator", agentNodePdf_Generator)
    graph.add_edge(START, "agentNodeLoad_Offer")
    graph.add_edge("agentNodeLoad_Offer", "agentNodeQuestion_Asker")
    graph.add_conditional_edges(
        "agentNodeQuestion_Asker",
        routeAfterQuestionAsker,
        {
            "supervisor": END,
            "coverLetterWriter": "agentNodeCover_Letter_Writer",
        },
    )
    graph.add_conditional_edges(
        "agentNodeCover_Letter_Writer",
        routeAfterCoverLetterWriter,
        {
            "pdfGenerator": "edgeNodeCoverLetterWriterToPdfGenerator",
            "supervisor": END,
        },
    )
    graph.add_edge("edgeNodeCoverLetterWriterToPdfGenerator", "agentNodePdf_Generator")
    graph.add_edge("agentNodePdf_Generator", END)
    return graph.compile()


coverLetterGraph = buildGraph()
