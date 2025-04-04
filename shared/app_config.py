import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, SecretStr
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings

# from sqlalchemy import literal


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def try_convert(value, t: type):
    bool_flag = ("true", "false", "t", "f", "yes", "no", "y", "n")
    if t == bool and value.lower() not in bool_flag:
        return None
    if t == float and "." not in value:
        return None
    try:
        return t(value)
    except Exception as e:
        return None


def replace_last_occurrence(text: str, old: str, new: str) -> str:
    index = text.rfind(old)
    if index == -1:
        return text
    return text[:index] + new + text[index + len(old) :]


class AutoEnvSettings(BaseSettings):
    class Config:
        case_sensitive = False
        extra = "allow"

    @classmethod
    def from_config(cls, env_file, flag="base"):
        """
        run this func to generate class vars, before runtime...
        """
        allow_type = [float, int, bool, str]
        insert_list = []

        file_name = cls.__module__
        file_path = getattr(sys.modules.get(file_name), "__file__", None)
        if file_path is None:
            return
        with open(file_path, "r") as f:
            code = f.read()

        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        for _t in allow_type:
                            v = try_convert(value, _t)
                            if v is not None:
                                if _t == str:
                                    v = f'"{v}"'
                                if key.lower() not in code:
                                    insert_list.append(
                                        f"    {key.lower()}: Optional[{_t.__name__}] = Field(default={v})"
                                    )
                                    break

            insert_list = [c for c in insert_list if c not in code] + [
                f"    # need *{flag}* replace"
            ]
            new_code = replace_last_occurrence(
                code, f"    # need *{flag}* replace", "\n".join(insert_list)
            )
            with open(file_path, "w") as f:
                f.write(new_code)

    @classmethod
    def load_env(cls, service_name=Path.cwd().name):
        if __name__ != "__main__":
            load_service_env(service_name)


class AppConfig(AutoEnvSettings):

    environment: Literal["dev", "prod"] = Field(default="dev")
    debug: bool = Field(default=True)
    graph_db_username: str = Field(default="neo4j")
    graph_db_password: str = Field(default="1qaz2wsxE")
    graph_db_url: str = Field(default="bolt://10.1.70.240:7688")
    graph_db_database_name: str = Field(default="neo4j")
    local_llm_model_name: str = Field(default="llama3.1")
    local_llm_url: str = Field(default="10.1.71.1:11434")
    persist_dir: str = Field(default="./shared_data/chroma_db")
    documentdir: str = Field(default="./test/")
    tavily_api_key: str = Field(default="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")
    bing_api_key: str = Field(default="7b921374511f44caad31c03ae2b25f88")
    nebula_user: str = Field(default="root")
    nebula_password: str = Field(default="1qaz2wsxE")
    nebula_address: str = Field(default="10.1.70.240:9669")
    space_name: str = Field(default="llamaindex_nebula_property_graph")
    shared_path: str = Field(default="./shared_data")
    agent_service_port: int = Field(default=8001)
    index_service_port: int = Field(default=8002)
    analysis_service_port: int = Field(default=8003)
    azure_openai_model_name: str = Field(default="gpt-4o")
    azure_openai_deployment_name: str = Field(default="gpt-4o")
    azure_openai_host: str = Field(default="https://azurechatgpt.openai.azure.com")
    azure_openai_api_version: str = Field(default="2024-05-01-preview")
    azure_openai_api_key: str
    web_socket_port: int = Field(default=8004)
    # need *base* replace


class WebUiAppConfig(AppConfig):
    max_history: int = Field(default=10)
    # need *webui* replace
    ...


class IndexAppConfig(AppConfig):
    num_files_limit: int = Field(default=10)
    # need *index* replace
    ...


class AgentAppConfig(AppConfig):
    # need *agent* replace
    ...


def patch_config(project_name):
    if project_name == "webui":
        return WebUiAppConfig()
    elif project_name == "agent_service":
        return AgentAppConfig()
    elif project_name == "index_service":
        return IndexAppConfig()
    else:
        return AppConfig()


if __name__ == "__main__":
    project_path = "."
    AppConfig.from_config(f"{project_path}/.env")
    WebUiAppConfig.from_config(f"{project_path}/webui/.env.dev", "webui")
    IndexAppConfig.from_config(f"{project_path}/index_service/.env.dev", "index")
    AgentAppConfig.from_config(f"{project_path}/agent_service/.env.dev", "agent")
else:
    from shared.utils import load_service_env
