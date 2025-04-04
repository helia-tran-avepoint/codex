import os
import sys

import uvicorn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

venv_path = "/home/celiu/projects/codex_v2/codex/agent_service/.venv"

os.environ["VIRTUAL_ENV"] = venv_path
os.environ["PATH"] = f"{venv_path}/bin:{os.environ['PATH']}"

site_packages = os.path.join(
    venv_path,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
sys.path.insert(0, site_packages)

print("Starting Agent Service...")

import agent_service
from agent_service import app_config
from shared.utils import generate_service_port


def add_host_mapping():
    with open("h", "r") as f:
        content = f.readlines()
        if not [i for i in content if "agent_service" in i]:
            with open("h", "a") as f:
                f.write("\n127.0.0.1    agent_service")


if __name__ == "__main__":
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
    print(f"Python executable: {sys.executable}")
    print(f"sys.path: {sys.path}")
    # add_host_mapping()
    agent_port, index_port, web_port, analysis_port, websocket_port = (
        generate_service_port("agent_service")
    )
    app_config.index_service_port = index_port
    app_config.analysis_service_port = analysis_port
    app_config.web_socket_port = websocket_port

    uvicorn.run("agent_service.app:app", host="0.0.0.0", port=agent_port)
