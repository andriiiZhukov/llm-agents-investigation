from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional
import uvicorn

class ToolState(TypedDict, total=False):
    input_text: Optional[str]
    response_text: Optional[str]

app = FastAPI()

graph = StateGraph(ToolState)

def process_request(state: ToolState) -> dict:
    input_text = state.get("input_text", "No input provided")
    response_text = f"Processed: {input_text}"
    return {"response_text": response_text}

graph.add_node("process_request", process_request)

graph.add_edge(START, "process_request")
graph.add_edge("process_request", END)

graph.set_entry_point("process_request")

compiled_graph = graph.compile()

@app.post("/mcp")
async def mcp_handler(request: ToolState):
    response = compiled_graph.invoke(request)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
