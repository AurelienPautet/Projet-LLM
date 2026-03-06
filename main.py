import os
import sqlite3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import questionary

console = Console()
load_dotenv()


def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )
    choix = questionary.select(
        "Que souhaitez-vous ajouter ?",
        choices=[
            "Expérience professionnelle",
            "Projet académique/perso",
            "Éducation",
            "Retour au menu principal"
        ]
    ).ask()

    console.print(
        Panel.fit("[bold blue]Career Copilot[/bold blue]", border_style="blue"))

    question = Prompt.ask(
        "[bold magenta]Entrez votre question[/bold magenta]")
    with console.status("[bold cyan]🔎 Analyste : Lecture de l'offre en cours...[/bold cyan]") as status:
        model_response = model.invoke(question)
    console.print(Panel(
        f"[bold green]✅ CV Généré avec succès ![/bold green]\nRéponse : {model_response.content}", title="Réponse"))


if __name__ == "__main__":
    main()
