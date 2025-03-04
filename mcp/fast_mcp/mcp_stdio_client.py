import asyncio
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client



server_params = StdioServerParameters(
    command="python",
    args=["path/to/mcp_stdio_server.py"],
)

llm = ChatOllama(model="llama3.1:8b", temperature=0)


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


async def run_mcp_client():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            llm_with_tools = llm.bind_tools(tools)

            def call_agent(state: GraphState):
                messages = state['messages']
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}

            def should_continue(state):
                messages = state['messages']
                last_message = messages[-1]

                if last_message.tool_calls:
                    return "tools"

                return END

            workflow = StateGraph(GraphState)
            workflow.add_node("agent", call_agent)
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges("agent", should_continue)
            workflow.add_edge("tools", "agent")

            app = workflow.compile()

            input_text = "Generate random number in range between 1 and 100 and store it to the database"
            async for event in app.astream({"messages": [HumanMessage(content=input_text)]}, stream_mode="updates"):
                print(event)
                print("\n")


asyncio.run(run_mcp_client())
