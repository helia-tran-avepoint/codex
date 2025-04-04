import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import pydantic
from pydantic import BaseModel, Field

from shared import app_config

PROJECT_CONFIGS_PATH = f"{app_config.shared_path}/project_configs"
os.makedirs(PROJECT_CONFIGS_PATH, exist_ok=True)


################################ Web UI Model Begin ##############################################################


def load_project_names():
    project_files = [f for f in os.listdir(PROJECT_CONFIGS_PATH) if f.endswith(".json")]
    return [os.path.splitext(f)[0] for f in project_files]


class PrivateKnowledge:
    def __init__(
        self,
        knowledge_dir: str,
        knowledge_type: str,
        knowledge_name: str,
        indexed: bool,
        last_index_time: str,
        index_dir: str,
    ):
        self.knowledge_dir = knowledge_dir
        self.knowledge_type = knowledge_type
        self.knowledge_name = knowledge_name
        self.indexed = indexed
        self.last_index_time = last_index_time
        self.index_dir = index_dir

    def to_dict(self):
        return {
            "knowledge_dir": self.knowledge_dir,
            "knowledge_type": self.knowledge_type,
            "knowledge_name": self.knowledge_name,
            "indexed": self.indexed,
            "last_index_time": self.last_index_time,
            "index_dir": self.index_dir,
        }

    @staticmethod
    def from_dict(data: Dict):
        return PrivateKnowledge(
            knowledge_dir=data["knowledge_dir"],
            knowledge_type=data["knowledge_type"],
            knowledge_name=data["knowledge_name"],
            indexed=pydantic.TypeAdapter(bool).validate_python(
                data.get("indexed", False)
            ),
            last_index_time=data.get("last_index_time", ""),
            index_dir=data.get("index_dir", ""),
        )


class Project:
    def __init__(
        self,
        name: str,
        knowledges: list[PrivateKnowledge],
    ):
        self.name = name
        self.knowledges = knowledges

    def save(self):
        project_data = {
            "name": self.name,
            "knowledges": [k.to_dict() for k in self.knowledges],
        }
        with open(os.path.join(PROJECT_CONFIGS_PATH, f"{self.name}.json"), "w") as f:
            json.dump(project_data, f, indent=4)

    @staticmethod
    def load(name: str):
        with open(os.path.join(PROJECT_CONFIGS_PATH, f"{name}.json"), "r") as f:
            project_data = json.load(f)
        knowledges = [PrivateKnowledge.from_dict(k) for k in project_data["knowledges"]]
        return Project(name=project_data["name"], knowledges=knowledges)

    def update_knowledge(self, knowledge: PrivateKnowledge):
        for i, _ in enumerate(self.knowledges):
            if self.knowledges[i].knowledge_name == knowledge.knowledge_name:
                self.knowledges[i] = knowledge
                continue

        self.save()


################################ Web UI Model End ################################################################

################################ Index Service Model Begin #######################################################


class IndexQueryRequest(BaseModel):
    query: str
    data_type: Literal["document", "source_code"]
    evaluate: bool = False


class BuildIndexRequest(BaseModel):
    path: str
    data_type: Literal["document", "source_code"]
    is_parallel: bool = False
    tasks: Optional[List[Literal["vector", "tree", "graph"]]] = None
    project_name: str
    knowledge_name: str


class LoadIndexRequest(BaseModel):
    project_name: str


################################ Index Service Model End #########################################################


################################ Agent Service Model Begin #######################################################


class AgentQueryRequest(BaseModel):
    query: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, {...}]


################################ Agent Service Model End #########################################################
