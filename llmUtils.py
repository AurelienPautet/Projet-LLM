import os
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from pydantic import BaseModel

load_dotenv()

LLM_MIN_TOKENS = 2
LLM_MAX_RETRIES = 10
LLM_RETRY_BASE_DELAY = 1.0
LLM_CLIENT_MAX_RETRIES = 5
LLM_TIMEOUT_SECONDS = 120

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


def buildChatModel(timeoutSeconds: float | None = None, maxRetries: int | None = None) -> ChatOpenAI:
    resolvedTimeout = timeoutSeconds if timeoutSeconds is not None else LLM_TIMEOUT_SECONDS
    resolvedRetries = maxRetries if maxRetries is not None else LLM_CLIENT_MAX_RETRIES
    return ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        timeout=resolvedTimeout,
        max_retries=resolvedRetries,
    )


def buildModel(tools: list) -> ChatOpenAI:
    return buildChatModel().bind_tools(tools)


def responseIsEmpty(response: AIMessage) -> bool:
    if bool(getattr(response, "tool_calls", None)):
        return False
    content = response.content
    if isinstance(content, str):
        return len(content.split()) < LLM_MIN_TOKENS
    return False


def invokeAgentWithRetries(model: ChatOpenAI, systemPrompt: str, messages: list, schema: type[BaseModel] | None = None) -> AIMessage | BaseModel:
    invoker = model.with_structured_output(schema) if schema else model
    for attempt in range(LLM_MAX_RETRIES):
        if attempt > 0:
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            console.print(
                f"[red]LLM returned empty response, retrying (attempt {attempt}/{LLM_MAX_RETRIES - 1}, delay {delay:.1f}s)...[/red]"
            )
            time.sleep(delay)
        response = invoker.invoke(
            [{"role": "system", "content": systemPrompt}] + messages)
        if schema or not responseIsEmpty(response):
            return response
    return response


def toDict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {"message": str(value)}


def extractStructuredOutput(result: dict) -> dict:
    structured = result.get("structured_response")
    if structured is None:
        return {}
    return toDict(structured)


def invokeStructuredAgent(agent, inputPayload: dict, config=None, recursionLimit: int = 80) -> dict:
    cfg = dict(config) if config else {}
    cfg.setdefault("recursion_limit", recursionLimit)
    return agent.invoke(inputPayload, config=cfg)


def invokeStructuredAgentWithEnforcedResponseTool(agent, inputMessages: list, config, schemaName: str, recursionLimit: int = 80) -> dict:
    result = invokeStructuredAgent(
        agent,
        {"messages": inputMessages},
        config,
        recursionLimit=recursionLimit,
    )
    structured = extractStructuredOutput(result)
    if structured:
        return result
    enforcementMessage = HumanMessage(
        content=(
            f"SYSTEM ENFORCEMENT: Your previous turn did not include the final structured response tool call. "
            f"Retry now and finish by calling {schemaName} exactly once as the final action."
        )
    )
    resultMessages = result.get("messages", [])
    baseMessages = resultMessages if resultMessages else inputMessages
    if baseMessages and isinstance(baseMessages[-1], HumanMessage):
        baseMessages = list(baseMessages) + [AIMessage(content="[Acknowledged, retrying structured output.]")]
    enforcedMessages = list(baseMessages) + [enforcementMessage]
    return invokeStructuredAgent(
        agent,
        {"messages": enforcedMessages},
        config,
        recursionLimit=recursionLimit,
    )


def formatLlmError(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if "429" in text or "rate-limit" in lowered or "rate limited" in lowered or "temporarily rate-limited" in lowered:
        return "The current model provider is rate-limited right now. Please retry in a few seconds."
    return f"LLM error: {text}"


def handleNodeError(exc: Exception, fieldName: str | None = None, defaultValues: dict | None = None) -> dict:
    from langgraph.errors import GraphRecursionError
    isRecursion = isinstance(exc, GraphRecursionError)
    if isRecursion:
        msg = "I could not finish this step because tool calls exceeded the safety limit. Please check tool availability and try again."
    else:
        msg = formatLlmError(exc)

    res = {"messages": [AIMessage(content=msg)], "status": ""}
    if fieldName:
        nodeOutput = {"message": msg}
        if defaultValues:
            nodeOutput.update(defaultValues)
        res[fieldName] = nodeOutput
        res["structured_response"] = nodeOutput
    return res
