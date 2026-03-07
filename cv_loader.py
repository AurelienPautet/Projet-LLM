import questionary
from cv_graph import cv_parser_graph
from base_graph import runGraph


def loadCvFlow():
    filepath = questionary.text(
        "Please enter the path to your CV (PDF, TXT, MD):").ask()
    if not filepath:
        return

    state = {"cv_path": filepath, "messages": [], "status": "Reading file..."}
    runGraph(cv_parser_graph, state)
