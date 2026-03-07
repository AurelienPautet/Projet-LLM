import os
from typing import Annotated, TypedDict

import fitz
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import add_experience
from llm_utils import build_model, invoke_model_with_retries

load_dotenv()

TOOLS = [add_experience]


class CVState(TypedDict):
    cv_path: str
    messages: Annotated[list[BaseMessage], add_messages]


def extract_cv_text(filepath: str) -> str:
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


def process_cv_node(state: CVState) -> dict:
    cv_text = extract_cv_text(state["cv_path"])
    sys_prompt = SystemMessage(content="You are an expert CV extractor. Your goal is to read the provided CV text and extract all professional experiences, adding them to the database using the add_experience tool. Extract as much detail as possible (title, description, start and end dates, company, location, technologies used).")
    human_msg = HumanMessage(
        content=f"Here is the CV text:\n{cv_text}\n\nPlease extract all experiences and insert them.")
    return {"messages": [sys_prompt, human_msg]}


def llm_node(state: CVState) -> dict:
    try:
        model = build_model(TOOLS)
        response = invoke_model_with_retries(model, state["messages"])
        return {"messages": [response]}
    except Exception as exc:
        return {"messages": [AIMessage(content=f"LLM error: {exc}")]}


tool_node = ToolNode(TOOLS)


def after_llm(state: CVState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool_node"
    return END


def after_tool(state: CVState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and "Error" not in last.content:
        return "llm_node"
    return "llm_node"


def build_cv_graph() -> StateGraph:
    graph = StateGraph(CVState)

    graph.add_node("process_cv_node", process_cv_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "process_cv_node")
    graph.add_edge("process_cv_node", "llm_node")

    graph.add_conditional_edges(
        "llm_node",
        after_llm,
        {"tool_node": "tool_node", END: END},
    )

    graph.add_conditional_edges(
        "tool_node",
        after_tool,
        {"llm_node": "llm_node"},
    )

    return graph.compile()


cv_parser_graph = build_cv_graph()
