from typing import Annotated, Any

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel
from IPython.display import Image, display


class BaseState(MessagesState):
    status: str


def runGraph(graph, initial_state: dict):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.rule import Rule
    import os

    console = Console()
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    hadError = False
    statusIndicator = None

    def setStatus(text: str):
        nonlocal statusIndicator
        if statusIndicator is not None:
            statusIndicator.stop()
        if text:
            statusIndicator = console.status(
                f"[dim]{text}[/dim]", spinner="dots")
            statusIndicator.start()

    def clearStatus():
        nonlocal statusIndicator
        if statusIndicator is not None:
            statusIndicator.stop()
            statusIndicator = None

    def toText(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str)
                else item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else ""
                for item in content
            )
        return ""

    if "status" in initial_state:
        setStatus(initial_state["status"])

    for updates in graph.stream(initial_state, stream_mode="updates"):
        if not updates:
            continue

        nodeName = next(iter(updates))
        nodeData = updates[nodeName]

        if "status" in nodeData:
            newStatus = nodeData["status"]
            if newStatus:
                setStatus(newStatus)
            else:
                clearStatus()

        messages = nodeData.get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        if "agentNode" in nodeName:
            text = toText(last.content)
            if text.startswith("LLM error:"):
                hadError = True
                clearStatus()
                console.print(Markdown(f"**Error:** {text}"), style="red")
            else:
                agentName = nodeName.replace("agentNode", "").replace("_", " ")
                if isinstance(last, BaseModel) and not isinstance(last, (AIMessage, ToolMessage, BaseMessage)):
                    text = getattr(last, "message", None)
                    if text:
                        clearStatus()
                        console.print(f"\n[bold blue]{agentName}:[/bold blue]")
                        console.print(Markdown(str(text)))
                if text:
                    clearStatus()
                    console.print(f"\n[bold blue]{agentName}:[/bold blue]")
                    console.print(Markdown(text))
                if getattr(last, "tool_calls", None):
                    clearStatus()
                    for tc in last.tool_calls:
                        args = ", ".join(
                            f"{k}={v!r}" for k, v in tc.get("args", {}).items())
                        console.print(
                            f"\n[bold green]Tool:[/bold green] {tc['name']}({args})")

        elif nodeName == "toolNode":
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    if "Error" in msg.content:
                        hadError = True
                        clearStatus()
                        console.print(
                            Markdown(f"**Tool error:** {msg.content}"), style="red")
                    if DEBUG:
                        clearStatus()
                        console.print(
                            f"[bold orange_red1]Tool output:[/bold orange_red1] {msg.content}")

    clearStatus()
    if hadError:
        console.print(Rule(style="yellow"))
        console.print("[bold yellow]  Finished with errors[/bold yellow]")
        console.print(Rule(style="yellow"))
    else:
        console.print(Rule(style="green"))
        console.print("[bold green]  Success![/bold green]")
        console.print(Rule(style="green"))


def drawGraph(graph):
    display(Image(graph.get_graph().draw_mermaid_png()))
