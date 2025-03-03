import json
from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Defining the state type
class MCPState(TypedDict):
    messages: List[Any]  # Message history
    context_requests: Optional[List[Dict[str, Any]]]  # MCP requests
    context_results: Optional[Dict[str, Any]]  # MCP request results
    current_node: str  # Current node for routing

# Simulating external data providers
class DataProviders:
    @staticmethod
    def search(parameters: Dict[str, Any]) -> str:
        """Simulated search provider"""
        query = parameters.get("query", "")
        if "python" in query.lower():
            return """
            Python is a high-level programming language. Latest versions:
            - Python 3.11: Performance improvements up to 60%, better error messages
            - Python 3.12: Faster f-strings, improved GIL, new debugging features
            """
        elif "javascript" in query.lower():
            return """
            JavaScript is a programming language for web development. Recent updates:
            - ES2022: Top-level await, RegExp match indices, Object.hasOwn()
            - ES2023: New array methods like findLast, toReversed, toSorted
            """
        else:
            return f"Search results for: {query} not found"
    
    @staticmethod
    def database(parameters: Dict[str, Any]) -> str:
        """Simulated database"""
        table = parameters.get("table", "")
        query = parameters.get("query", "")
        
        if table == "products":
            return """
            Query results from the products table:
            1. iPhone 15 Pro - $999
            2. Samsung Galaxy S23 - $899
            3. Google Pixel 8 - $799
            """
        else:
            return f"Table {table} not found or query not executed"

# MCP request handler
def process_mcp_requests(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Processes MCP requests and returns results"""
    results = {}
    providers = {
        "search": DataProviders.search,
        "database": DataProviders.database
    }
    
    for request in requests:
        request_id = request.get("request_id")
        provider = request.get("provider")
        parameters = request.get("parameters", {})
        
        if provider in providers:
            results[request_id] = providers[provider](parameters)
        else:
            results[request_id] = f"Provider {provider} not found"
    
    return results

# Graph components

def analyze_query(state: MCPState) -> MCPState:
    """Analyzes the user's query and determines necessary MCP requests"""
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        state["current_node"] = "generate_response"
        return state
    
    user_query = last_message.content
    
    # Request to LLM for generating MCP requests
    system_prompt = """
    Analyze the user's query and generate MCP requests to fetch necessary information.
    Available providers:
    1. search - Retrieves information from a search engine (parameter: query)
    2. database - Queries a database (parameters: table, query)
    
    Return the result in JSON format:
    {
      "context_requests": [
        {
          "request_id": "unique_id",
          "provider": "provider_name",
          "parameters": {
            "param1": "value1"
          }
        }
      ]
    }
    """
    
    request_message = f"User query: {user_query}"
    mcp_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=request_message)
    ])
    
    try:
        mcp_request = json.loads(mcp_response.content)
        state["context_requests"] = mcp_request.get("context_requests", [])
        state["current_node"] = "fetch_context"
    except json.JSONDecodeError:
        # If an error occurs, generate a response without context
        state["current_node"] = "generate_response"
        
    return state

def fetch_context(state: MCPState) -> MCPState:
    """Fetches external context based on MCP requests"""
    if state.get("context_requests"):
        state["context_results"] = process_mcp_requests(state["context_requests"])
    state["current_node"] = "generate_response"
    return state

def generate_response(state: MCPState) -> MCPState:
    """Generates a response based on the query and retrieved context"""
    messages = state["messages"]
    context_results = state.get("context_results", {})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return state
    
    user_query = last_message.content
    
    # Constructing system prompt with context
    system_prompt = "You are an AI assistant."
    
    if context_results:
        system_prompt += "\n\nAvailable external information sources:\n"
        for request_id, result in context_results.items():
            system_prompt += f"\n--- Result {request_id} ---\n{result}\n"
    
    # Generating response
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ])
    
    # Adding response to message history
    state["messages"].append(AIMessage(content=response.content))
    state["current_node"] = "end"
    
    return state

# Routing function
def router(state: MCPState) -> List[str]:
    """Determines the next step in the graph"""
    return [state["current_node"]]

# Creating and configuring the graph
def create_mcp_agent():
    """Creates and configures the MCP agent graph"""
    # Create graph
    workflow = StateGraph(MCPState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("fetch_context", fetch_context)
    workflow.add_node("generate_response", generate_response)
    
    # Configure routing
    workflow.add_edge("analyze_query", router)
    workflow.add_edge("fetch_context", router)
    workflow.add_edge("generate_response", router)
    
    # Set entry and exit points
    workflow.set_entry_point("analyze_query")
    workflow.set_finish_point("end")
    
    return workflow.compile()

# Example usage
def chat_with_mcp_agent(query: str):
    """Interacts with the MCP agent"""
    graph = create_mcp_agent()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "context_requests": None,
        "context_results": None,
        "current_node": "analyze_query"
    }
    
    # Execute the graph
    final_state = graph.invoke(initial_state)
    
    # Return response
    return final_state["messages"][-1].content

# Example queries
print(chat_with_mcp_agent("Tell me about the latest Python versions"))
print("\n--- New Query ---\n")
print(chat_with_mcp_agent("What smartphones are in our database?"))
