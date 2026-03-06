from typing import List, Optional
from sqlmodel import SQLModel, Field, ARRAY, String, create_engine, Session, text
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column


class Experience(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    titre: str
    description: str
    technos: List[str] = Field(sa_column=Column(ARRAY(String)))
    embedding: Optional[List[float]] = Field(
        sa_column=Column(Vector(1536))
    )


DATABASE_URL = "postgresql+psycopg://postgres:postgres@db:5432/career-goat"
engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    create_db_and_tables()
