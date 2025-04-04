from typing import Any, Dict

from llama_index.llms.ollama import Ollama
from llama_index.readers.file.docs import DocxReader

from index_service import app_config, logger
from index_service.custom_reader import (
    PandasCSVReader,
    PandasExcelReader,
    SourceCodeJsonReader,
)

INFERENCE_LOCAL_LLM = Ollama(
    model="mistral:7b",
    base_url=app_config.local_llm_url,
    keep_alive="5m",
    request_timeout=60000,
    temperature=0.1,
)

KNOWLEDGE_EXTRACT_LOCAL_LLM = Ollama(
    model="codellama:13b",
    base_url=app_config.local_llm_url,
    keep_alive="5m",
    request_timeout=60000,
    temperature=0.1,
)


def get_data_config(data_type: str) -> Dict[str, Any]:
    if data_type == "document":
        return {
            "file_extractor": {
                ".xls": PandasExcelReader(),
                ".xlsx": PandasExcelReader(),
                ".csv": PandasCSVReader(),
                ".docx": DocxReader(),
            },
            # "required_exts": [".xlsx", ".pdf", ".csv", ".log", ".docx", ".txt", ".md"],
            "required_exts": [".xlsx", ".csv", ".docx", ".txt", ".md"],
        }
    elif data_type == "source_code":
        return {
            "file_extractor": {
                ".json": SourceCodeJsonReader(),
            },
            # "file_extractor": None,
            "required_exts": [".cs", ".json"],  # [".json"],
        }
    else:
        raise ValueError(f"Unknown data type: {data_type}")
