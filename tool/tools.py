import os
from typing import List

import fitz
from langchain_core.tools import tool
from sqlmodel import Session, select
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast

from db.db import engine, Experience, ExperienceBase, ExperienceResult
from graph.embedding import createEmbeddingFromText
from llmUtils import schemaToEmbeddingText


@tool
def addExperience(experience: ExperienceBase) -> str:
    """Add a professional experience to the CV database. Best practice: check if the experience already exists using searchExperiences before adding to avoid duplicates."""
    try:
        embedding = createEmbeddingFromText(
            schemaToEmbeddingText(experience)
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
                f"({str(exp.start_date) if exp.start_date else '?'} - {str(exp.end_date) if exp.end_date else '?'}) | {technos}"
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
                schemaToEmbeddingText(experience)
            )

            session.add(dbExperience)
            session.commit()
            session.refresh(dbExperience)

        return f"Experience updated: {dbExperience.title} at {dbExperience.company_or_institution or 'N/A'}"
    except Exception as exc:
        return f"Error: editExperience failed: {exc}"


@tool
def getAllExperiences() -> str:
    """Get all existing experiences from the CV database. WARNING: Use sparingly as this retrieves all records. Prefer searchExperiences for targeted queries."""
    try:
        with Session(engine) as session:
            results = session.exec(select(Experience)).all()

        if not results:
            return "No experiences found."

        out: List[str] = []
        for exp in results:
            technos = ", ".join(exp.technos or [])
            out.append(
                f"[id={exp.id}] {exp.title} at {exp.company_or_institution or 'N/A'} "
                f"({str(exp.start_date) if exp.start_date else '?'} - {str(exp.end_date) if exp.end_date else '?'}) | {technos}"
            )
        return "\n".join(out)
    except Exception as exc:
        return f"Error: getAllExperiences failed: {exc}"


@tool
def getExperienceCount() -> str:
    """Get the total number of experiences in the CV database."""
    try:
        with Session(engine) as session:
            count = len(session.exec(select(Experience)).all())
        return f"Total experiences: {count}"
    except Exception as exc:
        return f"Error: getExperienceCount failed: {exc}"


@tool
def deleteExperience(id: int) -> str:
    """Delete an experience by its id."""
    try:
        with Session(engine) as session:
            dbExperience = session.get(Experience, id)
            if dbExperience is None:
                return f"Error: no experience found with id={id}"

            session.delete(dbExperience)
            session.commit()

        return f"Experience with id={id} deleted successfully"
    except Exception as exc:
        return f"Error: deleteExperience failed: {exc}"


@tool
def loadCvFromFile(filepath: str) -> str:
    """Read a CV file (PDF, TXT, or MD) and return its text content so you can extract and add experiences from it."""
    filepath = filepath.strip().strip("'\"")
    if not os.path.exists(filepath):
        return f"Error: file not found at path: {filepath}"
    ext = filepath.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as exc:
            return f"Error reading PDF: {exc}"
    elif ext in ['txt', 'md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as exc:
            return f"Error reading file: {exc}"
    else:
        return "Error: unsupported file format. Please provide a PDF, TXT, or MD file."
