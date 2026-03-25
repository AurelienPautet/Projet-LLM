from typing import Any

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel
import os


class BaseState(MessagesState):
    status: str


def runGraph(graph, initial_state: dict, agentName: str = "Assistant", firstQuestion: str | None = None, allowUserInput: bool = True):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.prompt import Prompt
    import json

    console = Console()
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    hadError = False
    statusIndicator = None
    lastToolHadError = False
    lastStatusText = ""
    lastPrintedName = ""
    lastPrintedText = ""

    history = list(initial_state.get("messages", []))
    seenMessageIds = {
        getattr(message, "id", None)
        for message in history
        if getattr(message, "id", None)
    }

    def setStatus(text: str):
        nonlocal statusIndicator, lastStatusText
        if statusIndicator is not None:
            statusIndicator.stop()
        lastStatusText = text
        if text:
            statusIndicator = console.status(
                f"[bold cyan]{text}[/bold cyan]", spinner="dots")
            statusIndicator.start()

    def clearStatus():
        nonlocal statusIndicator, lastStatusText
        if statusIndicator is not None:
            statusIndicator.stop()
            statusIndicator = None
        lastStatusText = ""

    def toText(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str)
                else item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else ""
                for item in content
            )
        return ""

    def formatToolArgValue(value: object) -> str:
        text = repr(value)
        if len(text) <= 40:
            return text
        return f"{text[:40]}..."

    def printToolCalls(message):
        toolCalls = getattr(message, "tool_calls", None)
        if not toolCalls:
            return
        for tc in toolCalls:
            args = ", ".join(
                f"{k}={formatToolArgValue(v)}" for k, v in tc.get("args", {}).items())
            console.print(
                f"\n[bold green]Tool:[/bold green] {tc['name']}({args})")

    def resolveDisplayName(nodeName: str, parentNodeName: str | None = None) -> str:
        if nodeName in ("agent", "model") and parentNodeName:
            nodeName = parentNodeName
        name = nodeName.replace("agentNode", "").replace("_", " ")
        if nodeName in ("agent", "model"):
            name = agentName
        return name

    def shouldPrintMessage(name: str, text: str) -> bool:
        nonlocal lastPrintedName, lastPrintedText
        normalizedText = text.strip()
        if not normalizedText:
            return False
        if lastPrintedName == name and lastPrintedText == normalizedText:
            return False
        lastPrintedName = name
        lastPrintedText = normalizedText
        return True

    def printAgentOutput(nodeName: str, message, parentNodeName: str | None = None):
        nonlocal hadError
        text = toText(getattr(message, "content", ""))

        if text.startswith("LLM error:"):
            hadError = True
            clearStatus()
            console.print(Markdown(f"**Error:** {text}"), style="red")
            return

        if isinstance(message, BaseModel) and not isinstance(message, (AIMessage, ToolMessage, BaseMessage)):
            modelText = getattr(message, "message", None)
            if modelText and shouldPrintMessage(resolveDisplayName(nodeName, parentNodeName), str(modelText)):
                clearStatus()
                name = resolveDisplayName(nodeName, parentNodeName)
                console.print(f"\n[bold blue]{name}:[/bold blue]")
                console.print(Markdown(str(modelText)))

        if text and shouldPrintMessage(resolveDisplayName(nodeName, parentNodeName), text):
            clearStatus()
            name = resolveDisplayName(nodeName, parentNodeName)
            console.print(f"\n[bold blue]{name}:[/bold blue]")
            console.print(Markdown(text))

        printToolCalls(message)

    def toStructuredPayload(structuredResponse):
        if hasattr(structuredResponse, "model_dump"):
            return structuredResponse.model_dump()
        if isinstance(structuredResponse, dict):
            return structuredResponse
        return {"message": str(structuredResponse)}

    def printStructuredMessage(nodeName: str, structuredResponse, parentNodeName: str | None = None):
        clearStatus()
        payload = toStructuredPayload(structuredResponse)
        message = payload.get("message", "")
        if not message:
            return
        name = resolveDisplayName(nodeName, parentNodeName)
        if not shouldPrintMessage(name, str(message)):
            return
        console.print(f"\n[bold blue]{name}:[/bold blue]")
        console.print(Markdown(str(message)))

    def printStructuredOutput(nodeName: str, structuredResponse, parentNodeName: str | None = None):
        clearStatus()
        name = resolveDisplayName(nodeName, parentNodeName)
        payload = toStructuredPayload(structuredResponse)
        console.print(f"\n[bold blue]{name}:[/bold blue]")
        console.print(json.dumps(payload, ensure_ascii=True, indent=2))

    def printToolOutput(messages):
        nonlocal hadError, lastToolHadError
        toolHadError = False
        for msg in messages:
            if isinstance(msg, ToolMessage):
                if "Error" in msg.content:
                    hadError = True
                    toolHadError = True
                    clearStatus()
                    console.print(
                        Markdown(f"**Tool error:** {msg.content}"), style="red")
                if DEBUG:
                    clearStatus()
                    console.print(
                        f"[bold orange_red1]Tool output:[/bold orange_red1] {msg.content}")
        lastToolHadError = toolHadError

    def processUpdates(updates, parentNodeName: str | None = None):
        if not updates:
            return None

        nodeName = next(iter(updates))
        nodeData = updates[nodeName]
        effectiveNodeName = parentNodeName if nodeName in (
            "agent", "model") and parentNodeName else nodeName

        if "status" in nodeData:
            newStatus = nodeData["status"]
            if newStatus:
                setStatus(newStatus)
            else:
                clearStatus()

        messages = nodeData.get("messages", [])
        structuredResponse = nodeData.get("structured_response")
        hasStructuredOutput = structuredResponse is not None
        aiMessages = [
            msg for msg in messages if isinstance(msg, AIMessage)]

        showStructuredOutput = hasStructuredOutput
        if hasStructuredOutput:
            payload = toStructuredPayload(structuredResponse)
            structuredMessage = str(payload.get("message", "")).strip()
            isFallbackStructuredError = structuredMessage.startswith(
                "I could not produce a structured"
            )
            hasRealAiText = any(
                (toText(getattr(msg, "content", "")).strip()) and
                (not toText(getattr(msg, "content", "")
                            ).strip().startswith("LLM error:"))
                for msg in aiMessages
            )
            if isFallbackStructuredError and hasRealAiText:
                showStructuredOutput = False

        if not messages:
            return []

        hasToolCalls = any(getattr(msg, "tool_calls", None)
                           for msg in messages)
        hasToolMessages = any(isinstance(msg, ToolMessage) for msg in messages)

        if effectiveNodeName == "tools" or effectiveNodeName == "toolNode":
            setStatus("Running tool...")
        elif hasToolCalls:
            setStatus("Running tool...")
        elif hasToolMessages and lastToolHadError:
            setStatus("Handling error...")
        elif hasToolMessages or effectiveNodeName == "agent" or "agentNode" in effectiveNodeName or effectiveNodeName == "model":
            setStatus("Evaluating tool output...")

        if hasToolMessages:
            toolMessages = [
                msg for msg in messages if isinstance(msg, ToolMessage) and getattr(msg, "id", None) not in seenMessageIds]
            printToolOutput(toolMessages)
            if lastToolHadError:
                setStatus("Handling error...")
            else:
                setStatus("Evaluating tool output...")

        if "agentNode" in effectiveNodeName or effectiveNodeName == "agent" or effectiveNodeName == "model":
            aiMessages = [
                msg for msg in aiMessages if getattr(msg, "id", None) not in seenMessageIds
            ]
            suppressNestedPdfModelOutput = (
                parentNodeName == "agentNodePdf_Generator" and nodeName in (
                    "agent", "model")
            )
            if aiMessages:
                structuredMessage = ""
                if showStructuredOutput:
                    payload = toStructuredPayload(structuredResponse)
                    structuredMessage = str(payload.get("message", "")).strip()
                suppressRawAiText = showStructuredOutput and not DEBUG

                if not suppressNestedPdfModelOutput:
                    for msg in aiMessages[:-1]:
                        msgText = toText(getattr(msg, "content", "")).strip()
                        if suppressRawAiText:
                            if getattr(msg, "tool_calls", None):
                                printToolCalls(msg)
                            continue
                        if msgText or getattr(msg, "tool_calls", None):
                            printAgentOutput(effectiveNodeName,
                                             msg, parentNodeName)

                    lastAi = aiMessages[-1]
                    lastText = toText(getattr(lastAi, "content", "")).strip()
                    if DEBUG:
                        printAgentOutput(effectiveNodeName,
                                         lastAi, parentNodeName)
                    else:
                        if suppressRawAiText:
                            if getattr(lastAi, "tool_calls", None):
                                printToolCalls(lastAi)
                        elif lastText.startswith("LLM error:"):
                            printAgentOutput(effectiveNodeName,
                                             lastAi, parentNodeName)
                        elif lastText and lastText != structuredMessage:
                            printAgentOutput(effectiveNodeName,
                                             lastAi, parentNodeName)
                        elif getattr(lastAi, "tool_calls", None):
                            printToolCalls(lastAi)

        isAgentNode = "agentNode" in effectiveNodeName or effectiveNodeName in ("agent", "model")
        if showStructuredOutput and isAgentNode:
            if DEBUG:
                printStructuredOutput(
                    effectiveNodeName, structuredResponse, parentNodeName)
            else:
                printStructuredMessage(
                    effectiveNodeName, structuredResponse, parentNodeName)

        return messages

    history = list(initial_state.get("messages", []))
    seenMessageIds = {
        getattr(message, "id", None)
        for message in history
        if getattr(message, "id", None)
    }

    def processQuestion(question: str) -> bool:
        cleaned = question.strip().lower()
        if cleaned in {"quit", "exit", "back"}:
            return False

        userMessage = HumanMessage(content=question)
        history.append(userMessage)
        setStatus("Thinking...")

        streamInput = dict(initial_state)
        streamInput["messages"] = history

        try:
            for event in graph.stream(streamInput, stream_mode="updates", subgraphs=True):
                parentNodeName = None
                if isinstance(event, tuple) and len(event) == 2:
                    path, updates = event
                    if isinstance(path, tuple) and len(path) > 0:
                        lastPathItem = path[-1]
                        if isinstance(lastPathItem, str) and lastPathItem:
                            parentNodeName = lastPathItem.split(":", 1)[0]
                else:
                    updates = event

                if updates:
                    nodeName = next(iter(updates))
                    nodeData = updates[nodeName]
                    for key, value in nodeData.items():
                        if key == "messages":
                            continue
                        initial_state[key] = value

                newMessages = processUpdates(
                    updates, parentNodeName=parentNodeName)
                if not newMessages:
                    continue
                for message in newMessages:
                    messageId = getattr(message, "id", None)
                    if messageId:
                        if messageId in seenMessageIds:
                            continue
                        seenMessageIds.add(messageId)
                    history.append(message)
        except Exception as exc:
            clearStatus()
            from llmUtils import formatLlmError
            errorMsg = formatLlmError(exc)
            console.print(
                f"[bold red]Graph execution error:[/bold red] {errorMsg}")
            history.append(AIMessage(
                content=f"An error occurred while processing your request: {errorMsg}"))

        initial_state["messages"] = history

        clearStatus()
        return True

    if firstQuestion and firstQuestion.strip():
        if not processQuestion(firstQuestion):
            clearStatus()
            return history

    if allowUserInput:
        while True:
            try:
                question = Prompt.ask("\n[bold magenta]You[/bold magenta]")
            except (EOFError, KeyboardInterrupt):
                break

            if question is None:
                break

            if not processQuestion(question):
                break

    clearStatus()
    return history
