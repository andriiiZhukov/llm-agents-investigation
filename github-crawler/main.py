import os
import requests
import tempfile
import importlib.util
from fastapi import FastAPI, Query
from langgraph.graph import StateGraph
from typing import Dict, Optional

class WorkflowState(Dict):
    user_input: str
    result: Optional[str] = None

def fetch_remote_agent(repo_url: str, file_path: str, function_name: str):
    """Downloads a Python file from a remote GitHub repository and imports the specified function."""
    raw_url = repo_url.replace("github.com", "raw.githubusercontent.com").rstrip("/") + "/master/" + file_path

    print(f"raw_url = {raw_url}")

    response = requests.get(raw_url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch agent code. Error code: {response.status_code}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(response.content)
        temp_filename = temp_file.name
    
    spec = importlib.util.spec_from_file_location("remote_agent", temp_filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.remove(temp_filename)
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Function {function_name} not found in the loaded code.")
    
    return getattr(module, function_name)

def llm_agent(state: WorkflowState):
    repo_url = "https://github.com/andriiiZhukov/import-llm-agent"
    file_path = "main.py"
    function_name = "custom_agent_function"
    
    agent_function = fetch_remote_agent(repo_url, file_path, function_name)
    state["result"] = agent_function(state)
    return state

workflow = StateGraph(WorkflowState)
workflow.add_node("llm_agent", llm_agent)
workflow.set_entry_point("llm_agent")
workflow.set_finish_point("llm_agent")
graph = workflow.compile()

app = FastAPI()

@app.get("/query")
def process_query(q: str = Query(..., description="Enter query")):
    """FastAPI endpoint to process requests via LangGraph"""
    initial_state = {"user_input": q}
    result = graph.invoke(initial_state)
    
    return result
