from langchain_core.tools import tool
from db import engine, Experience, ExperienceBase
from sqlmodel import Session


@tool
def add_experience(experience: ExperienceBase) -> str:
    """Add a professional experience to the CV database."""
    with Session(engine) as session:
        db_experience = Experience.model_validate(experience)
        session.add(db_experience)
        session.commit()
        session.refresh(db_experience)

    return f"Experience added: {db_experience.title} at {db_experience.company_or_institution or 'N/A'}"
