from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.server import create_mcp_app
from typing import TypedDict, Optional

class ToolState(TypedDict, total=False):
    input_text: Optional[str]
    response_text: Optional[str]

app = FastAPI()

graph = StateGraph(ToolState)

def process_request(state: ToolState) -> dict:
    input_text = state.get("input_text", "No input provided")
    response_text = f"MCP Processed: {input_text}"
    return {"response_text": response_text}

graph.add_node("process_request", process_request)

graph.add_edge(START, "process_request")
graph.add_edge("process_request", END)

graph.set_entry_point("process_request")

compiled_graph = graph.compile()

mcp_app = create_mcp_app(compiled_graph)
app.mount("/mcp", mcp_app)
