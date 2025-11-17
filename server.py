import asyncio
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from maestro_client import MaestroClient


load_dotenv()

mcp = FastMCP("Bots", stateless_http=True)

client = MaestroClient(
    login=os.getenv("LOGIN"),
    key=os.getenv("KEY"),
)


@mcp.tool(
    name="list_tasks",           # Custom tool name for the LLM
    description="Retorna as informações das tarefas que foram executadas e estão em execução no sistema", # Custom description
    tags={"automation", "tasks"},      # Optional tags for organization/filtering
    meta={"version": "1.0", "author": "product-team"}  # Custom metadata
)
async def list_tasks() -> dict:
    """List Botcity tasks."""
    tasks = await client.tasks.list()
    print("Tasks fetched")
    return tasks.data

@mcp.tool(name="ping", description="Health check endpoint")
def ping() -> dict:
    """Health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
