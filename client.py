from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
from langgraph.prebuilt import create_react_agent
from fastmcp import Client
import asyncio

from utils.constants import MCP_SERVER_URL, OLLAMA_BASE_URL

llm = OllamaLLM(model="llama3", base_url=OLLAMA_BASE_URL)

async def main():
    mcp = Client(MCP_SERVER_URL)
    async with mcp:
        tools = await mcp.list_tools()

        agent_tools = []
        # Create per-tool synchronous wrappers (they will run the async call inside a fresh event loop)
        for t in tools:
            def _tool_sync(*args, _tool_name=t.name, **kwargs):
                # Normalize input into a dict to pass to mcp.call_tool
                if args and not kwargs:
                    if len(args) == 1:
                        payload = args[0]
                        # If the agent returned a placeholder meaning "no input needed",
                        # treat it as empty args (don't send an 'input' key that the tool
                        # doesn't expect).
                        if isinstance(payload, str):
                            s = payload.strip().lower()
                            if s in ("", "none", "n/a") or "don't need" in s or "do not need" in s or s.startswith("none"):
                                args_dict = {}
                            else:
                                args_dict = {"input": payload}
                        elif isinstance(payload, dict):
                            args_dict = payload
                        else:
                            args_dict = {"arg": payload}
                    else:
                        args_dict = {"args": list(args)}
                else:
                    args_dict = kwargs or {}

                # Run the async MCP call in a fresh event loop (safe inside a worker thread)
                return asyncio.run(mcp.call_tool(_tool_name, args_dict))

            agent_tools.append(Tool(name=t.name, func=_tool_sync, description=t.description))

        agent = initialize_agent(llm=llm, tools=agent_tools, agent_type="zero-shot-react-description", verbose=True)

        print("Invoking agent...")
        # Run the (synchronous) agent.invoke in a thread so tool wrappers can use asyncio.run safely.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: agent.invoke("Tell me the quantity of tasks that have succeeded"))
        print("Agent result:\n", result)

asyncio.run(main())
