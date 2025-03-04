import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult


async def run_mcp_client():
    streams_context = sse_client(url="http://127.0.0.1:8080/sse")
    streams = await streams_context.__aenter__()

    session_context = ClientSession(*streams)
    session = await session_context.__aenter__()
    await session.initialize()

    response = await session.list_tools()
    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])

    result: CallToolResult = await session.call_tool("generate_random_number", {"range_min": 1, "range_max": 100})
    print("\nCalling generate_random_number tool. Generated random number: " + str(result.content[0].text))

    print("\nConverting mcp tools to langchain tools")
    tools = await load_mcp_tools(session)
    print(tools)

    print("\nCalling langchain tool:")
    res = await tools[0].ainvoke({"range_min": 1, "range_max": 100})
    print(res)

    print("\nReading a resource from mcp server:")
    resource = await session.read_resource("resource://numbers")
    print(resource)

    print("\nGetting a prompt from mcp server:")
    prompt = await session.get_prompt("my prompt")
    print(prompt)

    #close session and streams
    if session_context:
        await session_context.__aexit__(None, None, None)
    if streams_context:
        await streams_context.__aexit__(None, None, None)


asyncio.run(run_mcp_client())
