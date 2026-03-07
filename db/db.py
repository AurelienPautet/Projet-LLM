import os
from typing import List, Optional
from enum import Enum
from dotenv import load_dotenv
from sqlmodel import SQLModel, Field, ARRAY, String, create_engine, text
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column

load_dotenv()


class ExperienceType(str, Enum):
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    PROJECT = "project"


class ExperienceBase(SQLModel):
    title: str = Field(
        description="the title of the experience, e.g., 'Software Engineer Intern'")
    kind: Optional[ExperienceType] = Field(
        description="the kind of experience, e.g., 'professional', 'educational', 'project'")
    description: str = Field(
        description="a detailed description of the experience, including responsibilities and achievements")
    technos: Optional[List[str]] = Field(sa_column=Column(ARRAY(
        String)), description="a list of technologies used during the experience, e.g., ['Python', 'Django', 'PostgreSQL']")
    start_date: Optional[str] = Field(
        description="the start date of the experience")
    end_date: Optional[str] = Field(
        description="the end date of the experience")
    company_or_institution: Optional[str] = Field(
        description="the company or institution where the experience was gained")
    location: Optional[str] = Field(
        description="the location of the experience, e.g., 'Paris, France'")


class Experience(ExperienceBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    embedding: Optional[List[float]] = Field(
        default=None, sa_column=Column(Vector(3072))
    )


class ExperienceResult(ExperienceBase):
    id: int


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/career-goat")
engine = create_engine(DATABASE_URL)


def createDbAndTables():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    SQLModel.metadata.create_all(engine)


def resetDbAndTables():
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    createDbAndTables()
