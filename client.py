import requests

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import BaseTool

class MCPTool(BaseTool):
    name: str
    description: str
    endpoint: str  # URL completa do recurso no seu MCP Server

    def _run(self, query: str):
        """Executa a chamada HTTP para o MCP endpoint."""
        response = requests.post(
            self.endpoint,
            json={"input": query},
            timeout=20
        )
        response.raise_for_status()
        return response.json()

    async def _arun(self, query: str):
        return self._run(query)

llm = Ollama(
    model="llama3", 
    base_url="http://localhost:11434"
)

tools = [
    MCPTool(
        name="get_tasks_info",
        description="Retorna as informações das tarefas que foram executadas e estão em execução no sistema",
        endpoint="http://localhost:8000/mcp/resources/list_tasks"
    ),
    # MCPTool(
    #     name="get_system_status",
    #     description="Consulta o status atual do sistema",
    #     endpoint="http://localhost:8000/mcp/resources/get_system_status"
    # ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

if __name__ == '__main__':
    question = "Quais as tasks que mais resultaram em falha nos últimos 7 dias?"
    response = agent.run(question)
    print(response)