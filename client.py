from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
from langgraph.prebuilt import create_react_agent
from fastmcp import Client
import asyncio

from utils.constants import MCP_SERVER_URL, OLLAMA_BASE_URL

llm = OllamaLLM(model="llama3", base_url=OLLAMA_BASE_URL)

async def main():
    mcp = Client("MCP Server", server_url=MCP_SERVER_URL)
    async with mcp:
        tools = await mcp.list_tools()

        agent_tools = [
            Tool(name=t.name, func=lambda **args: asyncio.run(mcp.call_tool(t.name, args)), description=t.description)
            for t in tools
        ]
        agent = initialize_agent(llm=llm, tools=agent_tools, agent_type="zero-shot-react-description", verbose=True)
        result = agent.invoke("Tell me the quantity of tasks based on status")
        print(result)

asyncio.run(main())
