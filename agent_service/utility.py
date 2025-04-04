import os
from urllib.parse import urljoin

import chromadb
import chromadb.config
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.ollama import Ollama
from pydantic import SecretStr

from agent_service import app_config

model_name = app_config.local_llm_model_name
model_url = app_config.local_llm_url

llm_config = {
    "config_list": [
        {
            "model": model_name,
            "api_type": "ollama",
            "client_host": model_url,
        }
    ]
}

code_llm_config = {
    "config_list": [
        {
            "model": "deepseek-r1:32b",
            "api_type": "ollama",
            "client_host": model_url,
        }
    ]
}

azure_openai_llm_config = {
    "config_list": [
        {
            "model": app_config.azure_openai_deployment_name,
            "api_type": "azure",
            "api_key": app_config.azure_openai_api_key,
            "base_url": app_config.azure_openai_host,
            "api_version": app_config.azure_openai_api_version,
        }
    ]
}

# Ollama is referenced from index, ChatOllama is referenced from chain
llm = Ollama(
    model=model_name, base_url=model_url
)  # keep_alive="5m", request_timeout=120


code_llm = Ollama(model="deepseek-r1:32b", base_url=model_url, temperature=0.1)

general_llm = Ollama(model="mistral:7b", base_url=model_url, temperature=0.1)

routing_llm = Ollama(model="mixtral:8x7b", base_url=model_url, temperature=0.1)

chat_llm = ChatOllama(base_url=model_url, model=model_name, temperature=0.1)

chat_code_llm = ChatOllama(model="deepseek-r1:32b", base_url=model_url, temperature=0.1)

chat_general_llm = ChatOllama(model="mistral:7b", base_url=model_url, temperature=0.1)

chat_routing_llm = ChatOllama(model="mixtral:8x7b", base_url=model_url, temperature=0.1)

azure_openai_chat_llm = AzureChatOpenAI(
    azure_deployment=app_config.azure_openai_deployment_name,
    model=app_config.azure_openai_model_name,
    api_key=SecretStr(app_config.azure_openai_api_key),
    azure_endpoint=app_config.azure_openai_host,
    api_version=app_config.azure_openai_api_version,
)

azure_openai_llm = AzureOpenAI(
    model=app_config.azure_openai_model_name,
    deployment_name=app_config.azure_openai_deployment_name,
    api_key=app_config.azure_openai_api_key,
    azure_endpoint=app_config.azure_openai_host,
    api_version=app_config.azure_openai_api_version,
)


chroma_settings = chromadb.config.Settings(
    anonymized_telemetry=False,
)

embedding_func = OllamaEmbeddingFunction(
    url=urljoin(model_url, "api/embeddings"), model_name=model_name
)

chroma_vectordb = ChromaVectorDB(
    path=app_config.persist_dir,
    settings=chroma_settings,
    embedding_function=embedding_func,
)
