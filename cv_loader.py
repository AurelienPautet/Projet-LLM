import questionary
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from cv_graph import cv_parser_graph

console = Console()


def load_cv_flow():
    filepath = questionary.text(
        "Please enter the path to your CV (PDF, TXT, MD):").ask()
    if not filepath:
        return

    status = console.status("[dim]Processing CV...[/dim]", spinner="dots")
    status.start()

    had_error = False

    try:
        state = {"cv_path": filepath, "messages": []}
        for updates in cv_parser_graph.stream(state, stream_mode="updates"):
            node = next(iter(updates))
            if node == "llm_node":
                last = updates[node].get("messages", [])[-1]
                if getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        status.update(
                            f"[dim]Adding experience from CV...[/dim]")
                        args = ", ".join(f"{k}={v!r}" for k,
                                         v in tc.get("args", {}).items())
                        console.print(
                            f"  [bold cyan]>[/bold cyan] [dim]Extracted:[/dim] [yellow]{tc['name']}[/yellow][dim]({args})[/dim]")
            elif node == "tool_node":
                messages = updates[node].get("messages", [])
                for msg in messages:
                    if "Error" in msg.content:
                        had_error = True
                        console.print(f"[red]Tool error: {msg.content}[/red]")
    except Exception as e:
        console.print(f"[red]Error parsing CV: {e}[/red]")
        had_error = True
    finally:
        status.stop()

    if not had_error:
        console.print(
            "[bold green]Successfully loaded experiences from CV![/bold green]")
    else:
        console.print(
            "[bold yellow]Finished loading CV with some errors.[/bold yellow]")
