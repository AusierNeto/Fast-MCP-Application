import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from maestro_client import MaestroClient

load_dotenv()

mcp = FastMCP("Bots")

client = MaestroClient(
    login=os.getenv("LOGIN"),
    key=os.getenv("KEY"),
)

@mcp.tool
def list_tasks() -> dict:
    """List Botcity tasks."""
    return client.tasks.list().data

@mcp.tool
def ping() -> dict:
    """Health check"""
    return {"status": "ok"}


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
