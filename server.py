import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from maestro_client import MaestroClient

load_dotenv()

mcp = FastMCP("MaestroSDK", stateless_http=True)


# Helper to instantiate client
def get_client():
    client = MaestroClient(
        login=os.getenv("LOGIN"),
        key=os.getenv("KEY"),
    )
    return client


# -------------------------
# TASKS
# -------------------------

@mcp.tool(description="Lista todas as tasks.")
async def list_tasks(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.tasks.list()
    return resp.data


@mcp.tool(description="Obtém detalhes de uma task pelo ID.")
async def get_task_by_id(input: dict):
    client = get_client()
    await client.authenticate()
    task_id = input.get("task_id")
    resp = await client.tasks.get(task_id)
    return resp.data


@mcp.tool(description="Lista tasks filtradas por status (ex: FINISHED, ERROR, RUNNING).")
async def get_tasks_by_status(input: dict):
    status = input.get("status")
    print(status)
    client = get_client()
    await client.authenticate()
    resp = await client.tasks.list(Status=status)
    print(resp.data)
    return resp.data


@mcp.tool(description="Encerra uma task com sucesso.")
async def finish_task(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.tasks.finish(input["task_id"])
    return resp.data


# -------------------------
# AUTOMATIONS
# -------------------------

@mcp.tool(description="Lista todas as automações.")
async def list_automations(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.automations.list()
    return resp.data


@mcp.tool(description="Obtém uma automação pelo label.")
async def get_automation(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.automations.get(input["label"])
    return resp.data


# -------------------------
# RUNNERS
# -------------------------

@mcp.tool(description="Lista todos os runners.")
async def list_runners(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.runners.list()
    return resp.data


@mcp.tool(description="Obtém informações detalhadas de um runner.")
async def get_runner_info(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.runners.get_info(input["runner_id"])
    return resp.data


# -------------------------
# LOGS
# -------------------------

@mcp.tool(description="Lista logs.")
async def list_logs(input: dict):
    page = input.get("page", 1)
    size = input.get("size", 50)
    client = get_client()
    await client.authenticate()
    resp = await client.logs.list(page=page, size=size)
    return resp.data


# -------------------------
# ERRORS
# -------------------------

@mcp.tool(description="Lista erros recentes.")
async def list_errors(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.errors.list()
    return resp.data


@mcp.tool(description="Obtém um erro específico.")
async def get_error(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.errors.get(input["error_id"])
    return resp.data


# -------------------------
# DATAPOOLS
# -------------------------

@mcp.tool(description="Lista datapools.")
async def list_datapools(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.datapools.list()
    return resp.data


@mcp.tool(description="Adiciona item a um datapool.")
async def datapool_push_item(input: dict):
    client = get_client()
    await client.authenticate()
    dp = input["datapool"]
    data = input["data"]
    resp = await client.datapools.add_item(dp, **data)
    return resp.data


# -------------------------
# ARTIFACTS
# -------------------------

@mcp.tool(description="Lista artifacts.")
async def list_artifacts(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.result_files.list()
    return resp.data


@mcp.tool(description="Obtém artifact pelo ID.")
async def get_artifact(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.result_files.get(input["artifact_id"])
    return resp.data


# -------------------------
# SCHEDULES
# -------------------------

@mcp.tool(description="Lista schedules.")
async def list_schedules(input:dict={}):
    client = get_client()
    await client.authenticate()
    resp = await client.schedules.list()
    return resp.data


@mcp.tool(description="Cria schedule.")
async def create_schedule(input: dict):
    client = get_client()
    await client.authenticate()
    resp = await client.schedules.create(**input)
    return resp.data


# -------------------------
# SERVER START
# -------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
