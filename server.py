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
def list_tasks(status: str = None) -> dict:
    """List BotCity tasks by status."""
    return client.tasks.list(status=status).data

@mcp.tool
def get_bots() -> dict:
    """List registered bots."""
    return client.bots.list().data

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
