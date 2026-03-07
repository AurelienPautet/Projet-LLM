import os
import fitz
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from tool.tools import addExperience
from graph.baseGraph import BaseState
from llmUtils import buildModel, invokeModelWithRetries

load_dotenv()

TOOLS = [addExperience]


class CVState(BaseState):
    cv_path: str


def extractCvText(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "File not found."
    ext = filepath.lower().split('.')[-1]
    if ext == 'pdf':
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"
    elif ext in ['txt', 'md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading text file: {e}"
    else:
        return "Unsupported file format. Please provide a PDF, TXT, or MD file."


def processCvNode(state: CVState) -> Command:
    cvText = extractCvText(state["cv_path"])
    sysPrompt = SystemMessage(
        content="You are an expert CV extractor. Your goal is to read the provided CV text and extract all professional experiences, adding them to the database using the addExperience tool. Extract as much detail as possible (title, description, start and end dates, company, location, technologies used)."
    )
    humanMsg = HumanMessage(
        content=f"Here is the CV text:\n{cvText}\n\nPlease extract all experiences and insert them."
    )
    return Command(
        goto="llmNode",
        update={
            "messages": [sysPrompt, humanMsg],
            "status": "Analyzing CV content..."
        }
    )


def llmNode(state: CVState) -> dict:
    try:
        model = buildModel(TOOLS)
        response = invokeModelWithRetries(model, state["messages"])
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


def afterLlmNode(state: CVState) -> Command:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return Command(goto="toolNode", update={"status": "Running tools..."})
    return Command(goto=END, update={"status": ""})


def afterToolNode(state: CVState) -> Command:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return Command(goto="llmNode", update={"status": "Analysing tool output..."})
    return Command(goto="llmNode", update={"status": "Handling error..."})


def buildCvGraph() -> StateGraph:
    graph = StateGraph(CVState)

    graph.add_node("processCvNode", processCvNode)
    graph.add_node("llmNode", llmNode)
    graph.add_node("toolNode", ToolNode(TOOLS))

    graph.add_node("afterLlmNode", afterLlmNode)
    graph.add_node("afterToolNode", afterToolNode)

    graph.add_edge(START, "processCvNode")
    graph.add_edge("llmNode", "afterLlmNode")
    graph.add_edge("toolNode", "afterToolNode")

    return graph.compile()


cv_parser_graph = buildCvGraph()
