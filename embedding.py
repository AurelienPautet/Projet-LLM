import os

from langchain_openai import OpenAIEmbeddings


def create_embedding_from_text(text: str) -> list[float]:
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=os.getenv("AI_EMBEDDING_API_KEY") or os.getenv("AI_API_KEY"),
        base_url=os.getenv(
            "AI_EMBEDDING_ENDPOINT") or os.getenv("AI_ENDPOINT"),
    )
    return embeddings.embed_query(text)
