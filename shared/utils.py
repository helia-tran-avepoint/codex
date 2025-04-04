import os
import random
import re
import socket
from pathlib import Path

import pydantic
from dotenv import load_dotenv

import shared.constants as constants

DEVELOPMENT = False


def load_service_env(service_name=Path.cwd().name):
    load_dotenv()

    DEVELOPMENT = pydantic.TypeAdapter(bool).validate_python(os.getenv(constants.DEBUG))

    env_path = ensure_subdir(Path.cwd(), service_name)
    env_file = f"{env_path}/.env.{'dev' if DEVELOPMENT else 'prod' }"

    if env_path.exists():
        print(f"Loading environment variables from {env_file}")
        load_dotenv(env_file, override=True)
    else:
        print(
            f"Environment file {env_file} not found. Using default environment variables."
        )


def ensure_subdir(base_path, subdir_name):
    base_path = Path(base_path)
    if base_path.name != subdir_name:
        base_path = base_path / subdir_name
    return base_path.resolve()


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def check_port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except socket.error as e:
            return False


def generate_port(unuse_port=[8080]):
    port = random.randint(1024, 49151)
    while port in unuse_port or check_port_in_use("127.0.0.1", port):
        port = random.randint(1024, 49151)

    return port


def patch_port_file(
    port_mapping_path, index_port, web_port, agent_port, analysis_port, websocket_port
):
    os.environ["INDEX_SERVICE_PORT"] = str(index_port)
    os.environ["AGENT_SERVICE_PORT"] = str(agent_port)
    os.environ["ANALYSIS_SERVICE_PORT"] = str(analysis_port)
    os.environ["WEB_SOCKET_PORT"] = str(websocket_port)

    with open(port_mapping_path, "w") as f:
        f.writelines(
            [
                f"index_service:{index_port}\n",
                f"agent_service:{agent_port}\n",
                f"web_service:{web_port}\n",
                f"analysis_service:{analysis_port}\n",
                f"websocket_service:{websocket_port}",
            ]
        )


def generate_all_port(port_mapping_path):
    index_port = generate_port()
    web_port = generate_port()
    agent_port = generate_port()
    analysis_port = generate_port()
    websocket_port = generate_port()

    patch_port_file(
        port_mapping_path,
        index_port,
        web_port,
        agent_port,
        analysis_port,
        websocket_port,
    )
    return index_port, agent_port, web_port, analysis_port, websocket_port


def generate_service_port(service_name):
    port_mapping_path = "./shared/service_port_mapping"
    if os.path.exists(port_mapping_path):
        with open(port_mapping_path, "r") as f:
            lines = f.readlines()

        if not lines:
            index_port, agent_port, web_port, analysis_port, websocket_port = (
                generate_all_port(port_mapping_path)
            )
        else:
            generate_new_port = False
            for line in lines:
                name, port = line.split(":")
                port = int(port)
                if name == service_name:
                    generate_new_port = (
                        generate_new_port
                        if not check_port_in_use("127.0.0.1", port)
                        else True
                    )

                if generate_new_port:
                    port = generate_port()
                if name == "index_service":
                    index_port = port
                if name == "agent_service":
                    agent_port = port
                if name == "web_service":
                    web_port = port
                if name == "analysis_service":
                    analysis_port = port
                if name == "websocket_service":
                    websocket_port = port

            patch_port_file(
                port_mapping_path,
                index_port,
                web_port,
                agent_port,
                analysis_port,
                websocket_port,
            )

    else:
        index_port, agent_port, web_port, analysis_port, websocket_port = (
            generate_all_port(port_mapping_path)
        )

    return agent_port, index_port, web_port, analysis_port, websocket_port
