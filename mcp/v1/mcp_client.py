import requests

MCP_SERVER_URL = "http://localhost:8000/mcp"

request_data = {"input_text": "Hello, MCP Server!"}
response = requests.post(MCP_SERVER_URL, json=request_data)

print("Server Response:", response.json())
