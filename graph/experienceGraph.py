from dotenv import load_dotenv
from langchain.agents import create_agent

from tool.tools import addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile, upsertPersonalInfo, getPersonalInfo, getAllPersonalInfo
from llmUtils import buildChatModel

load_dotenv()

TOOLS = [addExperience, searchExperiences, editExperience, deleteExperience, getAllExperiences, getExperienceCount, loadCvFromFile, upsertPersonalInfo, getPersonalInfo, getAllPersonalInfo]

SYSTEM_PROMPT = "You are an expert at collecting and organizing CV data. Your goal is to help the user add or modify professional experiences and personal information in their database. Always use available tools when the user asks for data retrieval, creation, deletion, counting, or updates. Do not claim an operation succeeded unless a tool call confirms it. If a tool returns an error, explain it clearly and ask for the smallest next action to fix it, then retry with tools when the user confirms. You can add experiences manually by gathering details from the user (title, description, start and end dates, company, location, technologies used), or load them from a CV file using the loadCvFromFile tool if the user provides a file path. You can also save personal information fields using upsertPersonalInfo and retrieve them with getPersonalInfo or getAllPersonalInfo. Ask clarifying questions if needed."


def buildGraph():
    model = buildChatModel()
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


career_graph = buildGraph()
