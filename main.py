import warnings
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from graph.experienceGraph import career_graph
from graph.cvGraph import cv_graph
from graph.coverLetterGraph import coverLetterGraph
from graph.offerGraph import offerGraph
from graph.baseGraph import runGraph
from llmUtils import buildChatModel, formatLlmError, extractStructuredOutput, invokeStructuredAgentWithEnforcedResponseTool
from db.db import createDbAndTables

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Pydantic serializer warnings:[\s\S]*PydanticSerializationUnexpectedValue[\s\S]*",
)

console = Console()


class SupervisorDecision(BaseModel):
    message: str = Field(description="Short message shown to the user.")
    route: Literal["offer", "experience", "cv", "coverLetter", "user", "quit", "clarify"] = Field(
        description="Selected route.")
    offerId: Optional[int] = Field(
        default=None,
        description="Offer id to use for offer, CV, or cover letter workflows when provided by user.")
    tailoredQuestion: Optional[str] = Field(
        default=None,
        description="Rewritten and clear request tailored specifically for the chosen specialist route, free from filler."
    )
    restartSpecialist: bool = Field(
        default=False,
        description="Set to true ONLY if the user explicitly asks to restart, clear, or redo the task from scratch (e.g., 'redo the letter', 'recommence', 'refait')."
    )


SUPERVISOR_PROMPT = """
You are a routing supervisor for a career assistant.

Route capabilities:
- offer: collect/store a job offer from user text or URL, keep offer tracking state.
- experience: load CV files into DB, add/edit/delete/search experiences and personal information.
- cv: generate or improve CV output using DB data, optionally tailored to an existing offer id.
- coverLetter: generate or improve a cover letter, optionally tailored to an existing offer id.
- user: do not call any specialist graph, just let the user continue speaking.
- clarify: ask a short clarification question when truly ambiguous.
- quit: explicit exit intent.

Decision rules:
1. If user asks to import/load/extract data from a CV file, pick experience.
2. If user pastes a offer URL or a raw offer text, pick offer first. Dont forget to give the full url to the offer graph so it can fetch and extract data from it. 
3. If user asks to save/store/fetch an offer, pick offer.
4. If user asks to generate/improve a CV, pick cv.
5. If user asks to generate/improve a cover letter, pick coverLetter.
6. If user asks to continue/refine/retry, or if the user is providing answers to questions previously asked by a specialist, MUST keep the previous specialist route.
7. Use user when the input is purely conversational and no specialist action is required.
8. Always provide a tailoredQuestion when routing to specialist agents. Extract only the crucial instructions, format them clearly, and leave out conversational filler.
9. If the user asks to redo, restart, or recreate the document entirely from scratch, set restartSpecialist to true so the old state is cleared before running.

IMPORTANT: You now have full memory of the conversation. DO NOT repeat the questions or output from the specialists. If the most recent event is a specialist asking the user a question, select the 'user' route and leave message blank (so the user can reply). BUT IF the most recent event is the user replying to those questions, you MUST route them to that previous specialist so it can continue!

offerId policy:
- Extract a numeric offerId from user text when present.
- If an existing offer candidate is provided in context, use it when user intent targets an existing offer.
- If user asks for CV/cover letter for the same/previous offer and activeOfferId exists, set offerId to activeOfferId.
- For offer route, set offerId when user refers to an existing offer.

Keep message concise and actionable. Return structured output only.
""".strip()


def buildSupervisor():
    model = buildChatModel()
    return create_agent(
        model=model,
        tools=[],
        system_prompt=SUPERVISOR_PROMPT,
        response_format=SupervisorDecision
    )


def routeWithSupervisor(supervisor, messages: list) -> SupervisorDecision:
    result = invokeStructuredAgentWithEnforcedResponseTool(
        supervisor,
        messages,
        config=None,
        schemaName="SupervisorDecision"
    )
    structured_dict = extractStructuredOutput(result)
    return SupervisorDecision(**structured_dict)


def runSelectedGraph(route: str, firstQuestion: str, offerId: Optional[int], states: dict):
    if route == "offer":
        if "offer" not in states:
            states["offer"] = {"messages": [], "status": ""}
        return runGraph(offerGraph, states["offer"],
                        agentName="Offer manager", firstQuestion=firstQuestion, allowUserInput=False)
    if route == "experience":
        if "experience" not in states:
            states["experience"] = {"messages": [], "status": ""}
        return runGraph(career_graph, states["experience"],
                        agentName="Experience manager", firstQuestion=firstQuestion, allowUserInput=False)
    if route == "cv":
        if "cv" not in states:
            states["cv"] = {"messages": [], "status": "", "activeOfferId": offerId}
        else:
            states["cv"]["activeOfferId"] = offerId
        return runGraph(cv_graph, states["cv"],
                        agentName="CV manager", firstQuestion=firstQuestion, allowUserInput=False)
    if route == "coverLetter":
        if "coverLetter" not in states:
            states["coverLetter"] = {"messages": [], "status": "", "activeOfferId": offerId}
        else:
            states["coverLetter"]["activeOfferId"] = offerId
        return runGraph(coverLetterGraph, states["coverLetter"],
                        agentName="Cover letter manager", firstQuestion=firstQuestion, allowUserInput=False)
    return []


def main():
    createDbAndTables()
    console.print(Rule(style="blue"))
    console.print(Panel("[bold blue]Career Copilot[/bold blue]",
                  border_style="blue", padding=(0, 4)))
    console.print(Rule(style="blue"))

    supervisor = buildSupervisor()
    previousRoute: Optional[str] = None
    chatHistory = []
    graphStates = {}

    while True:
        try:
            userText = Prompt.ask("\n[bold magenta]You[/bold magenta]")
        except (EOFError, KeyboardInterrupt):
            break
        if userText is None:
            break
        userText = userText.strip()
        if not userText:
            continue

        chatHistory.append(HumanMessage(content=userText))

        while True:
            supervisorMessages = list(chatHistory)
            if previousRoute:
                supervisorMessages.append(HumanMessage(
                    content=f"Hint: The previous specialist route was '{previousRoute}'."))

            try:
                with console.status("[bold cyan]Supervisor is deciding...[/bold cyan]", spinner="dots"):
                    decision = routeWithSupervisor(supervisor, supervisorMessages)
            except Exception as exc:
                console.print(f"[yellow]{formatLlmError(exc)}[/yellow]")
                decision = SupervisorDecision(
                    message="I could not decide the next route right now. Please rephrase your request.",
                    route="clarify",
                    offerId=None,
                )

            if decision.route == "quit":
                console.print("[bold blue]Supervisor:[/bold blue] Goodbye!")
                return

            if decision.route == "clarify":
                console.print(
                    f"[bold blue]Supervisor:[/bold blue] {decision.message}")
                chatHistory.append(AIMessage(content=decision.message))
                break

            if decision.route == "user":
                if decision.message.strip():
                    console.print(
                        f"[bold blue]Supervisor:[/bold blue] {decision.message}")
                    chatHistory.append(AIMessage(content=decision.message))
                break

            if decision.message.strip():
                console.print(
                    f"[bold blue]Supervisor:[/bold blue] {decision.message}")
                chatHistory.append(AIMessage(content=decision.message))

            previousRoute = decision.route
            actualQuestion = decision.tailoredQuestion if decision.tailoredQuestion else userText

            if decision.restartSpecialist and decision.route in graphStates:
                del graphStates[decision.route]

            graphHistory = runSelectedGraph(
                decision.route, actualQuestion, decision.offerId, graphStates)

            lastAgentMessage = ""
            if graphHistory:
                for msg in reversed(graphHistory):
                    if isinstance(msg, AIMessage) and msg.content and isinstance(msg.content, str):
                        lastAgentMessage = msg.content
                        break

            if decision.route == "coverLetter":
                coverLetterState = graphStates.get("coverLetter", {})
                if coverLetterState.get("questionAskerHasRun") and not coverLetterState.get("writerOutput"):
                    if lastAgentMessage:
                        chatHistory.append(AIMessage(content=lastAgentMessage))
                    break

            if lastAgentMessage:
                chatHistory.append(HumanMessage(
                    content=f"[SYSTEM TO SUPERVISOR]: The specialist '{decision.route}' just outputted the following to the user:\n{lastAgentMessage}\n\nIMPORTANT: If the specialist's output ends with questions for the user (numbered questions, '?'), you MUST select 'user' route immediately and leave message blank — the user must reply first. Only route back to '{decision.route}' if the user's CURRENT message is a direct answer to those questions."))


if __name__ == "__main__":
    main()
