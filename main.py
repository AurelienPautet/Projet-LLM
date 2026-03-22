from typing import Literal, Optional
import warnings

from langchain_core.messages import HumanMessage, SystemMessage
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
from llmUtils import buildChatModel, formatLlmError
from db.db import createDbAndTables
from tool.tools import getActiveOfferRecord

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
        description="Offer id to use for CV or cover letter workflows when provided by user.")


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
2. If user pastes a new offer URL or a new raw offer text, pick offer first.
3. If user asks to save/store/fetch an offer, pick offer.
4. If user asks to generate/improve a CV, pick cv.
5. If user asks to generate/improve a cover letter, pick coverLetter.
5. If user asks to continue/refine/retry and previous route is meaningful, keep previous route.
6. Use user when the input is conversational and no specialist action is required.

offerId policy:
- Extract a numeric offerId from user text when present.
- If user asks for CV/cover letter for the same/previous offer and activeOfferId exists, set offerId to activeOfferId.
- For offer route, keep offerId as null.

Keep message concise and actionable. Return structured output only.
""".strip()


def buildSupervisor():
    model = buildChatModel()
    return model.with_structured_output(SupervisorDecision)


def routeWithSupervisor(supervisor, userText: str, previousRoute: Optional[str]) -> SupervisorDecision:
    activeOffer = getActiveOfferRecord()
    activeOfferId = activeOffer.id if activeOffer else None
    activeOfferStatus = activeOffer.status.value if activeOffer else "none"
    activeOfferSource = activeOffer.offerSource if activeOffer else "none"
    contextText = (
        f"Previous route: {previousRoute or 'none'}\n"
        f"Active offer id: {activeOfferId}\n"
        f"Active offer status: {activeOfferStatus}\n"
        f"Active offer source: {activeOfferSource}\n"
        f"User request: {userText}"
    )
    return supervisor.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=contextText),
    ])


def runSelectedGraph(route: str, firstQuestion: str, offerId: Optional[int] = None):
    if route == "offer":
        runGraph(offerGraph, {"messages": [], "status": ""},
                 agentName="Offer manager", firstQuestion=firstQuestion, allowUserInput=False)
        return
    if route == "experience":
        runGraph(career_graph, {"messages": [], "status": ""},
                 agentName="Experience manager", firstQuestion=firstQuestion, allowUserInput=False)
        return
    if route == "cv":
        initialState = {"messages": [], "status": "", "activeOfferId": offerId}
        runGraph(cv_graph, initialState,
                 agentName="CV manager", firstQuestion=firstQuestion, allowUserInput=False)
        return
    if route == "coverLetter":
        initialState = {"messages": [], "status": "", "activeOfferId": offerId}
        runGraph(coverLetterGraph, initialState,
                 agentName="Cover letter manager", firstQuestion=firstQuestion, allowUserInput=False)


def main():
    createDbAndTables()
    console.print(Rule(style="blue"))
    console.print(Panel("[bold blue]Career Copilot[/bold blue]",
                  border_style="blue", padding=(0, 4)))
    console.print(Rule(style="blue"))

    supervisor = buildSupervisor()
    previousRoute: Optional[str] = None

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

        try:
            with console.status("[bold cyan]Supervisor is deciding...[/bold cyan]", spinner="dots"):
                decision = routeWithSupervisor(
                    supervisor, userText, previousRoute)
        except Exception as exc:
            console.print(f"[yellow]{formatLlmError(exc)}[/yellow]")
            decision = SupervisorDecision(
                message="I could not decide the next route right now. Please rephrase your request.",
                route="clarify",
                offerId=None,
            )

        if decision.route == "quit":
            console.print("[bold blue]Goodbye![/bold blue]")
            break

        if decision.route == "clarify":
            console.print(
                f"[bold blue]Supervisor:[/bold blue] {decision.message}")
            continue

        if decision.route == "user":
            console.print(
                f"[bold blue]Supervisor:[/bold blue] {decision.message}")
            continue

        console.print(f"[bold blue]Supervisor:[/bold blue] {decision.message}")
        previousRoute = decision.route
        runSelectedGraph(decision.route, userText, decision.offerId)


if __name__ == "__main__":
    main()
