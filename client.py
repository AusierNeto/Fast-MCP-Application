import requests

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import BaseTool

from fastmcp import Client


client = Client("http://localhost:8000/mcp/")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        
        print("Tools:", tools)
        print("Resources:", resources)
        print("Prompts:", prompts)
        
        # Execute operations
        result = await client.call_tool("list_tasks")#, {"param": "value"})
        print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

