import asyncio
import os
import sys
import re
import google.generativeai as genai
from fastmcp.client.client import Client
from utils.prompt_templates import MAESTRO_AUTOMATION_PROMPT
from dotenv import load_dotenv


# ------------------------------------------------------
# Configuration
# ------------------------------------------------------

load_dotenv()
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8000/mcp/")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Environment variable GOOGLE_API_KEY is required.")

# Configure Gemini SDK
genai.configure(api_key=GOOGLE_API_KEY)



# ------------------------------------------------------
# Helper: Execute a REACT agent step
# ------------------------------------------------------

async def run_react_step(llm, prompt):
    """Executes one step of REACT reasoning using Gemini."""
    response = llm.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            candidate_count=1,
            max_output_tokens=4096
        )
    )

    return response.text



# ------------------------------------------------------
# Agent implementation (REACT-style)
# ------------------------------------------------------

async def run_agent(llm, mcp_client, query: str, system_prompt: str):
    """
    Executes a full REACT agent loop against the MCP server.
    """

    # Build initial prompt
    prompt = system_prompt.replace("{input}", query)

    # Append initial "Thought:"
    prompt += "\nThought: "

    print("\n🔍 AGENT RUN")
    print("------------------")

    while True:
        step_output = await run_react_step(llm, prompt)
        print(step_output)

        # Check if model reached final answer
        if "Final Answer:" in step_output:
            print("\n🎯 Final Answer Found!")
            return step_output.split("Final Answer:")[1].strip()

        # Otherwise, expect an Action
        action_match = re.search(r"Action: (\w+)", step_output)
        action_input_match = re.search(r"Action Input: (.*)", step_output)

        if not action_match:
            raise RuntimeError("LLM did not provide an Action step.")

        tool_name = action_match.group(1)
        tool_input_raw = action_input_match.group(1) if action_input_match else "{}"

        # FastMCP expects dict input
        try:
            import json
            tool_input = json.loads(tool_input_raw)
        except:
            tool_input = {}

        # Execute tool
        print(f"\n🔧 Calling MCP Tool: {tool_name}")
        observation = await mcp_client.call_tool(tool_name, tool_input)

        print(f"📝 Observation: {observation}")

        # Append Observation to prompt and continue
        prompt += (
            step_output
            + f"\nObservation: {observation}\nThought: "
        )



# ------------------------------------------------------
# Setup Agent
# ------------------------------------------------------

async def setup_agent():
    print(f"Connecting to MCP server at {MCP_URL}")

    mcp_client = Client(MCP_URL)

    # IMPORTANT: open the MCP session
    async with mcp_client:
        print("Fetching available tools...")
        tools = await mcp_client.list_tools()
        print(f"Found {len(tools)} tools")

        # Monta um bloco de descrição das tools para o prompt
        tools_block = "\n".join(
            f"- {t.name}: {t.description or ''}"
            for t in tools
        )

        # Monta a lista de nomes de tools que o modelo PODE usar em Action:
        tool_names = ", ".join(t.name for t in tools)

        # Prepare Gemini LLM
        llm = genai.GenerativeModel(MODEL_NAME)

        # Build system prompt
        system_prompt = (
            MAESTRO_AUTOMATION_PROMPT.format(
                tools=tools_block,
                tool_names=tool_names,
                input="{input}"
            )
        )

        # Return both the LLM and an *already connected* MCP client session
        return llm, mcp_client, system_prompt


# ------------------------------------------------------
# Main loop
# ------------------------------------------------------

async def main():
    print("\n BotCity AI Assistant (Gemini Official SDK Edition) ")
    print("-" * 50)
    print("Ask me anything about BotCity Maestro tasks and operations!")
    print("\nType 'exit' to quit.")
    print("-" * 50)

    llm, mcp_client, system_prompt = await setup_agent()

    # keep the MCP session alive for the whole conversation
    async with mcp_client:
        print("Ready for actions!")

        while True:
            user_query = input("\n🔍 Your query: ")

            if user_query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if not user_query.strip():
                continue

            try:
                answer = await run_agent(llm, mcp_client, user_query, system_prompt)
                print("\n💡 ANSWER:")
                print(answer)

            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
