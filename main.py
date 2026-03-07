import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status
import questionary

from graph import career_graph

console = Console()
load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


def to_text(content: object) -> str:
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

    had_error = False
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

    for updates in career_graph.stream(
        {"messages": []},
        stream_mode="updates",
    ):
        if not updates:
            continue

        node = next(iter(updates))
        messages = updates[node].get("messages", [])
        if not messages:
            continue
        last = messages[-1]

        if node == "human_input_node":
            set_status("Thinking...")
            continue

        if node == "llm_node":
            clear_status()

            if not isinstance(last, AIMessage):
                continue

            text = to_text(last.content)

            if text.startswith("LLM error:"):
                had_error = True
                console.print(Markdown(f"**Error:** {text}"), style="red")
                continue

            if text:
                console.print("\n[bold blue]Assistant:[/bold blue]")
                console.print(Markdown(text))

            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    args = ", ".join(f"{k}={v!r}" for k,
                                     v in tc.get("args", {}).items())
                    console.print(
                        f"\n[bold green]Tool:[/bold green] {tc['name']}({args})")
                set_status("Running tool...")
            continue

        if node == "tool_node":
            clear_status()

            any_success = False
            for msg in messages:
                if not isinstance(msg, ToolMessage):
                    continue
                if "Error" in msg.content:
                    had_error = True
                    console.print(
                        Markdown(f"**Tool error:** {msg.content}"), style="red")
                else:
                    any_success = True
                if DEBUG:
                    console.print(
                        f"[bold orange_red1]Tool output:[/bold orange_red1] {msg.content}")

            if any_success:
                set_status("Thinking...")
            continue

    clear_status()
    if had_error:
        console.print(Rule(style="yellow"))
        console.print("[bold yellow]  Finished with errors[/bold yellow]")
        console.print(Rule(style="yellow"))
    else:
        console.print(Rule(style="green"))
        console.print("[bold green]  Saved successfully![/bold green]")
        console.print(Rule(style="green"))


if __name__ == "__main__":
    main()
