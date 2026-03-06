import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolMessage
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
import questionary

from graph import career_graph

console = Console()
load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def main():
    console.print(Rule(style="blue"))
    console.print(Panel("[bold blue]Career Copilot[/bold blue]",
                  border_style="blue", padding=(0, 4)))
    console.print(Rule(style="blue"))

    category = questionary.select(
        "What do you want to add?",
        choices=[
            "Professional experience",
            "Academic / personal project",
            "Education",
            "Back to main menu",
        ]
    ).ask()

    if category == "Back to main menu":
        return

    in_llm_turn = False
    status: Status | None = None

    def set_status(text: str):
        nonlocal status
        if status is not None:
            status.stop()
        status = console.status(f"[dim]{text}[/dim]", spinner="dots")
        status.start()

    def clear_status():
        nonlocal status
        if status is not None:
            status.stop()
            status = None

    for mode, payload in career_graph.stream(
        {"messages": []},
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates":
            updates = payload
            if not updates:
                continue
            node = next(iter(updates))
            node_state = updates[node]
            messages = node_state.get("messages", [])
            last_message = messages[-1] if messages else None

            if node == "human_input_node":
                clear_status()
                set_status("Waiting for next step...")
                continue

            if node == "llm_node":
                has_tool_calls = bool(
                    getattr(last_message, "tool_calls", None)
                )
                if has_tool_calls:
                    set_status("Waiting for next step...")
                else:
                    clear_status()
                continue

            if node == "tool_node":
                if isinstance(last_message, ToolMessage) and "Error" in last_message.content:
                    clear_status()
                continue

        if mode != "messages":
            continue

        chunk, metadata = payload
        node = metadata.get("langgraph_node", "")

        if node == "llm_node" and isinstance(chunk, AIMessageChunk):
            if chunk.content:
                clear_status()
                if not in_llm_turn:
                    in_llm_turn = True
                    console.print(
                        "\n[bold blue]Assistant:[/bold blue] ", end="")
                console.print(chunk.content, end="", markup=False)
            continue

        if in_llm_turn:
            in_llm_turn = False
            console.print()

        if node == "tool_node" and isinstance(chunk, ToolMessage):
            clear_status()
            content = escape(chunk.content)
            if "Error" not in chunk.content:
                console.print(f"\n[bold green]  {content}[/bold green]")
            else:
                console.print(f"\n[bold red]  Error: {content}[/bold red]")
            if DEBUG:
                console.log(f"[DEBUG] tool raw: {chunk.content}")

    clear_status()
    console.print(Rule(style="green"))
    console.print("[bold green]  Saved successfully![/bold green]")
    console.print(Rule(style="green"))


if __name__ == "__main__":
    main()
