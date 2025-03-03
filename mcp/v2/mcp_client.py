from langchain_mcp_adapters.client import MCPClient

MCP_SERVER_URL = "http://localhost:8000/mcp"

client = MCPClient(MCP_SERVER_URL)

response = client.invoke({"input_text": "Hello, MCP Server!"})

print("Server Response:", response)
