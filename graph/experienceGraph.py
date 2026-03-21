import os

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

from tool.tools import addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile
from llmUtils import buildChatModel

load_dotenv()

TOOLS = [addExperience, searchExperiences,
         editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile]


SYSTEM_PROMPT = "You are an expert at collecting and organizing professional experiences. Your goal is to help the user add or modify professional experiences in their database. Always use available tools when the user asks for data retrieval, creation, deletion, counting, or updates. Do not claim an operation succeeded unless a tool call confirms it. If a tool returns an error, explain it clearly and ask for the smallest next action to fix it, then retry with tools when the user confirms. You can add experiences manually by gathering details from the user (title, description, start and end dates, company, location, technologies used), or load them from a CV file using the loadCvFromFile tool if the user provides a file path. Ask clarifying questions if needed."


def buildGraph():
    model = buildChatModel()
    return create_react_agent(model=model, tools=TOOLS, prompt=SYSTEM_PROMPT)


career_graph = buildGraph()
