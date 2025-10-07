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
def greet(name: str) -> str:
    return f"Hello, {name}!"

@mcp.tool
def list_tasks(status: str = None) -> dict:
    """List BotCity tasks by status."""
    return client.tasks.list(status=status).data

@mcp.tool
def cancel_task(task_id: str) -> dict:
    """Cancel a BotCity task."""
    return client.tasks.cancel(task_id).data

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
