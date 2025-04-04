import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

venv_path = "/home/celiu/projects/codex_v2/codex/webui/.venv"

os.environ["VIRTUAL_ENV"] = venv_path
os.environ["PATH"] = f"{venv_path}/bin:{os.environ['PATH']}"

site_packages = os.path.join(
    venv_path,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
sys.path.insert(0, site_packages)

print("Starting Web UI...")

from streamlit.web import cli as stcli

import webui
from shared.utils import generate_service_port
from webui import app_config


def add_host_mapping():
    with open("/etc/hosts", "r") as f:
        content = f.readlines()
        if not [i for i in content if "web_service" in i]:
            with open("/etc/hosts", "a") as f:
                f.write("\n127.0.0.1    web_service")


if __name__ == "__main__":
    print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
    print(f"Python executable: {sys.executable}")
    print(f"sys.path: {sys.path}")
    # add_host_mapping()
    agent_port, index_port, web_port, analysis_port, websocket_port = (
        generate_service_port("web_service")
    )
    app_config.index_service_port = index_port
    app_config.agent_service_port = agent_port
    app_config.analysis_service_port = analysis_port
    app_config.web_socket_port = websocket_port

    sys.argv = [
        "streamlit",
        "run",
        "./webui/app.py",
        f"--server.port={web_port}",
        "--server.address=0.0.0.0",
    ]
    stcli.main()
