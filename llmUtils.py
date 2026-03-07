import os
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from pydantic import BaseModel

load_dotenv()

LLM_MIN_TOKENS = int(os.getenv("AI_MIN_TOKENS", "2"))
LLM_MAX_RETRIES = int(os.getenv("AI_MIN_TOKENS_RETRIES", "10"))
LLM_RETRY_BASE_DELAY = float(os.getenv("AI_RETRY_BASE_DELAY", "1.0"))
console = Console()


def schemaToEmbeddingText(obj: BaseModel) -> str:
    lines = []
    for fieldName, fieldValue in obj.model_dump().items():
        if fieldValue is None:
            continue
        if isinstance(fieldValue, list):
            fieldValue = ", ".join(str(item) for item in fieldValue)
        label = fieldName.replace("_", " ").title()
        lines.append(f"{label}: {fieldValue}")
    return "\n".join(lines)


def buildModel(tools: list) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        timeout=float(os.getenv("AI_TIMEOUT_SECONDS", "120")),
        max_retries=1,
    ).bind_tools(tools)


def responseIsEmpty(response: AIMessage) -> bool:
    hasToolCalls = bool(getattr(response, "tool_calls", None))
    if hasToolCalls:
        return False
    content = response.content
    if isinstance(content, str):
        return len(content.split()) < LLM_MIN_TOKENS
    return False


def invokeModelWithRetries(model: ChatOpenAI, messages: list) -> AIMessage:
    for attempt in range(LLM_MAX_RETRIES):
        if attempt > 0:
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            console.print(
                f"[red]LLM returned empty response, retrying (attempt {attempt}/{LLM_MAX_RETRIES - 1}, delay {delay:.1f}s)...[/red]"
            )
            time.sleep(delay)
        response = model.invoke(messages)
        if not responseIsEmpty(response):
            return response
    return response
