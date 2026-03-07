from typing import List

from langchain_core.tools import tool
from sqlmodel import Session, select
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast

from db.db import engine, Experience, ExperienceBase, ExperienceResult
from graph.embedding import createEmbeddingFromText


def experienceToEmbeddingText(experience: ExperienceBase) -> str:
    technos = ", ".join(experience.technos or [])
    return "\n".join([
        f"Title: {experience.title}",
        f"Description: {experience.description}",
        f"Technologies: {technos}",
        f"Start date: {experience.start_date or ''}",
        f"End date: {experience.end_date or ''}",
        f"Organization: {experience.company_or_institution or ''}",
        f"Location: {experience.location or ''}",
    ])


@tool
def addExperience(experience: ExperienceBase) -> str:
    """Add a professional experience to the CV database."""
    try:
        embedding = createEmbeddingFromText(
            experienceToEmbeddingText(experience)
        )

        with Session(engine) as session:
            db_experience = Experience.model_validate(experience)
            db_experience.embedding = embedding
            session.add(db_experience)
            session.commit()
            session.refresh(db_experience)

        return f"Experience added: {db_experience.title} at {db_experience.company_or_institution or 'N/A'}"
    except Exception as exc:
        return f"Error: addExperience failed: {exc}"


@tool
def searchExperiences(query: str, limit: int = 5) -> str:
    """Search experiences semantically using a natural language query. Returns the most relevant experiences."""
    try:
        queryEmbedding = createEmbeddingFromText(query)

        with Session(engine) as session:
            results = session.exec(
                select(Experience)
                .order_by(Experience.embedding.op("<=>")(cast(queryEmbedding, Vector(3072))))
                .limit(limit)
            ).all()

        if not results:
            return "No experiences found."

        out: List[str] = []
        for exp in results:
            technos = ", ".join(exp.technos or [])
            out.append(
                f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
                f"({exp.start_date or '?'} - {exp.end_date or '?'}) | {technos}"
            )
        return "\n".join(out)
    except Exception as exc:
        return f"Error: searchExperiences failed: {exc}"


@tool
def editExperience(id: int, experience: ExperienceBase) -> str:
    """Edit an existing experience by its id. All fields will be replaced with the provided values."""
    try:
        with Session(engine) as session:
            dbExperience = session.get(Experience, id)
            if dbExperience is None:
                return f"Error: no experience found with id={id}"

            for field, value in experience.model_dump().items():
                setattr(dbExperience, field, value)

            dbExperience.embedding = createEmbeddingFromText(
                experienceToEmbeddingText(experience)
            )

            session.add(dbExperience)
            session.commit()
            session.refresh(dbExperience)

        return f"Experience updated: {dbExperience.title} at {dbExperience.company_or_institution or 'N/A'}"
    except Exception as exc:
        return f"Error: editExperience failed: {exc}"
