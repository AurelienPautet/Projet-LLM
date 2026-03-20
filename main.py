import questionary
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from graph.experienceGraph import career_graph
from graph.cvGraph import cv_graph
from graph.baseGraph import runGraph


console = Console()


def main():
    console.print(Rule(style="blue"))
    console.print(Panel("[bold blue]Career Copilot[/bold blue]",
                  border_style="blue", padding=(0, 4)))
    console.print(Rule(style="blue"))

    while True:
        category = questionary.select(
            "What do you want to do?",
            choices=[
                "Add/modify a professional experience",
                "Create a cv for a specific job offer (not implemented yet)",
                "Quit",
            ]
        ).ask()

        if category == "Quit" or category is None:
            console.print("[bold blue]Goodbye![/bold blue]")
            break

        if category == "Add/modify a professional experience":
            runGraph(career_graph, {"messages": [], "status": ""})
        elif category == "Create a cv for a specific job offer (not implemented yet)":
            runGraph(cv_graph, {"messages": [], "status": ""})


if __name__ == "__main__":
    main()
