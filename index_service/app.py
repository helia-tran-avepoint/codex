import asyncio
import json
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

import multiprocessing
import os
import random
from asyncio import Lock
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import date
from enum import Enum
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.composability.graph import ComposableGraph
from pydantic import BaseModel, Field

from index_service import app_config, index, logger
from index_service.config import INFERENCE_LOCAL_LLM
from index_service.custom_cache import EmbeddingCache, LLMCache
from index_service.custom_embedding import OllamaEmbedding

# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
from index_service.custom_ollama import Ollama
from index_service.custom_query import GraphRAGQueryEngine
from index_service.custom_transform import SentenceSplitter
from index_service.evaluate import LlamaIndexOllamaEvaluator
from index_service.index import get_collection_name, get_or_create_vector_store
from index_service.utils import construct_dir_path, safe_save_file
from shared import constants
from shared.concurrent_utils import run_jobs_threadpool
from shared.models import (
    PROJECT_CONFIGS_PATH,
    BuildIndexRequest,
    IndexQueryRequest,
    LoadIndexRequest,
    Project,
    load_project_names,
)
from shared.utils import DEVELOPMENT

executor = ThreadPoolExecutor()

app = FastAPI(debug=DEVELOPMENT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

build_index_task_lock = Lock()

embedding_cache_persist_path = "./transform_cache/embedding"
llm_cache_persist_path = "./transform_cache/llm"
Path("./transform_cache").mkdir(parents=True, exist_ok=True)


def init():
    global embedding_cache
    global llm_cache
    global ollama_embedding

    logger.debug("Begin init app.")

    # multiprocessing.set_start_method("fork", force=True)

    # original_get_context = multiprocessing.get_context

    # def patched_get_context(method):
    #     if method == "spawn":
    #         return original_get_context("fork")
    #     return original_get_context(method)

    # multiprocessing.get_context = patched_get_context

    model_name = app_config.local_llm_model_name
    model_url = app_config.local_llm_url

    embedding_cache = (
        EmbeddingCache.from_persist_path(embedding_cache_persist_path)
        if Path(embedding_cache_persist_path).exists()
        else EmbeddingCache()
    )

    llm_cache = (
        LLMCache.from_persist_path(llm_cache_persist_path)
        if Path(llm_cache_persist_path).exists()
        else LLMCache()
    )

    ollama_embedding = OllamaEmbedding(
        model_name="bge-m3:latest",
        cache=embedding_cache,
        base_url=model_url,
        ollama_additional_kwargs={"mirostat": 0, "keep_alive": "5m"},
    )

    Settings.node_parser = SentenceSplitter()
    Settings.chunk_overlap = 200
    Settings.embed_model = ollama_embedding
    Settings.llm = Ollama(
        model=model_name,
        cache=llm_cache,
        base_url=model_url,
        keep_alive="5m",
        request_timeout=1200,
        temperature=0.1,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    async_scheduler = AsyncIOScheduler()
    async_scheduler.add_job(
        safe_build_indexes,
        "cron",
        hour=12,
        minute=0,
    )
    async_scheduler.add_job(persist_cache, "interval", minutes=10)  # seconds=10

    async_scheduler.start()
    yield

    await persist_cache()
    async_scheduler.shutdown()
    logger.debug("Shutdown completed")


async def persist_cache():
    """Persist cache periodically."""
    logger.debug("Persisting cache...")
    run_jobs_threadpool(
        [
            lambda func=func, path=path: safe_save_file(func, path)
            for func, path in zip(
                [embedding_cache.persist, llm_cache.persist],
                [embedding_cache_persist_path, llm_cache_persist_path],
            )
        ],
        workers=2,
    )
    # loop = asyncio.get_running_loop()
    # await loop.run_in_executor(executor, sync_persist_cache)


def sync_persist_cache():
    safe_save_file(embedding_cache.persist, embedding_cache_persist_path)
    safe_save_file(llm_cache.persist, llm_cache_persist_path)
    logger.debug("Cache persisted successfully.")


async def safe_build_indexes():
    if build_index_task_lock.locked():
        logger.info("Previous task is still running. Skipping this run.")
        return
    async with build_index_task_lock:
        for project_file_name in os.listdir(
            f"{app_config.shared_path}/project_configs"
        ):
            project_name = project_file_name.split(".json")[0]
            project = Project.load(project_name)
            for knowledge in project.knowledges:
                await index.build_indexes(
                    path=knowledge.knowledge_dir,
                    project_name=project_name,
                    knowledge_name=knowledge.knowledge_name,
                    force_rebuild=True,
                    is_parallel=False,
                    data_type=knowledge.knowledge_type,
                    tasks=(
                        ["vector"]
                        if knowledge.knowledge_type == "document"
                        else ["vector", "graph"]
                    ),
                )


async def load_query_engine(
    knowledge_path_list: List[tuple[str, str]],
    data_type: str,
    project_name: str,
) -> BaseQueryEngine:
    logger.debug(f"Begin load query engine for project {project_name} {data_type}")

    all_indices = {}
    knowledge_names = []
    indexs_list = run_jobs_threadpool(
        [
            index.build_indexes(
                path=path,
                project_name=project_name,
                knowledge_name=knowledge_name,
                force_rebuild=False,
                is_parallel=True,
                data_type=data_type,
                tasks=["vector"] if data_type == "document" else ["vector", "graph"],
            )
            for path, knowledge_name in knowledge_path_list
        ],
        workers=len(knowledge_path_list),
    )
    for indexs in indexs_list:
        for key, value in indexs.items():
            new_key = key
            counter = 1
            while new_key in all_indices:
                new_key = f"{key}_{counter}"
                counter += 1

            all_indices[new_key] = value

    for path, knowledge_name in knowledge_path_list:
        knowledge_names.append(knowledge_name)

    # TODO enhance the query engine to support multi index
    # composable_graph = ComposableGraph(all_indices=all_indices, root_id="vector")
    # return composable_graph.as_query_engine()

    if data_type == "document":
        return all_indices["vector"].as_query_engine()
    elif data_type == "source_code":
        extra_method_name_vector_stores = [
            get_or_create_vector_store(
                get_collection_name(
                    project_name, data_type, knowledge_name, "method_name"
                )
            )
            for knowledge_name in knowledge_names
        ]

        extra_community_summary_vector_stores = [
            get_or_create_vector_store(
                get_collection_name(
                    project_name, data_type, knowledge_name, "community_summary"
                )
            )
            for knowledge_name in knowledge_names
        ]

        graph_store = all_indices["graph"].property_graph_store
        query_engine = GraphRAGQueryEngine(
            vector_query_engine=all_indices["vector"].as_query_engine(),
            extra_method_name_vector_stores=extra_method_name_vector_stores,
            extra_community_summary_vector_stores=extra_community_summary_vector_stores,
            graph_store=graph_store,
            # llm=Settings.llm,
            llm=INFERENCE_LOCAL_LLM,
            index=all_indices["graph"],
            similarity_top_k=10,
        )
        return query_engine
    else:
        raise Exception("Not Support data type")


@app.post("/retrieve")
async def a_retrieve(request: IndexQueryRequest):
    if request.evaluate:
        results, record = evaluator.query(
            query=request.query,
            data_type=request.data_type,
            evaluate=request.evaluate,
        )
        response = {"query": request.query, "results": results}
        if record:
            response["feedback"] = record.feedbacks
        return response
    else:
        if request.data_type == "document":
            response = await document_query_engine.aquery(request.query)
        elif request.data_type == "source_code":
            response = await source_code_query_engine.aquery(request.query)
        else:
            raise ValueError(f"Unknown data type: {request.data_type}")

        return response


@app.post("/build_index")
async def a_build_index(request: BuildIndexRequest):
    logger.debug("Begin build index")

    if build_index_task_lock.locked():
        logger.info("Previous task is still running. Skipping this run.")
        return
    async with build_index_task_lock:
        await index.build_indexes(
            path=request.path,
            data_type=request.data_type,
            project_name=request.project_name,
            knowledge_name=request.knowledge_name,
            force_rebuild=True,
            is_parallel=request.is_parallel,
            tasks=(
                ["vector"] if request.data_type == "document" else ["vector", "graph"]
            ),
        )
        return {"status": "success", "message": "Index built successfully"}


@app.post("/load_index")
async def a_load_index(request: LoadIndexRequest):
    global document_query_engine
    global source_code_query_engine
    global evaluator
    global knowledges_indexed

    logger.debug("Begin load index")

    project = Project.load(request.project_name)

    document_knowledge_paths = [
        (knowledge.knowledge_dir, knowledge.knowledge_name)
        for knowledge in project.knowledges
        if knowledge.knowledge_type == "document"
    ]
    document_query_engine = await load_query_engine(
        document_knowledge_paths, "document", request.project_name
    )

    source_code_knowledge_paths = [
        (knowledge.knowledge_dir, knowledge.knowledge_name)
        for knowledge in project.knowledges
        if knowledge.knowledge_type == "source_code"
    ]
    source_code_query_engine = await load_query_engine(
        source_code_knowledge_paths, "source_code", request.project_name
    )

    evaluator = LlamaIndexOllamaEvaluator(
        document_query_engine=document_query_engine,
        source_code_query_engine=source_code_query_engine,
        embedding_model=ollama_embedding,
    )


async def save_file(
    file: UploadFile,
    relative_path: str,
    project_name: str,
    knowledge_name: str,
    knowledge_type: str,
) -> None:
    target_folder_path = construct_dir_path(
        app_config.shared_path,
        ["project_datas", project_name, knowledge_type, knowledge_name],
    )
    target_path = target_folder_path.joinpath(relative_path)
    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)


@app.post("/upload_files")
async def upload_files(
    files: List[UploadFile] = File(...),
    relative_paths: List[str] = Form(...),
    project_name: str = Form(...),
    knowledge_name: str = Form(...),
    knowledge_type: str = Form(...),
):
    if knowledge_name:
        logger.debug("Begin upload file")
        run_jobs_threadpool(
            [
                save_file(
                    file,
                    relative_path,
                    project_name,
                    knowledge_type,
                    knowledge_name,
                )
                for file, relative_path in zip(files, relative_paths)
            ],
            workers=max(len(files), 100),
            show_progress=True,
            desc="Upload files to Server",
        )

    return {
        "message": f"Files for project '{project_name}' and knowledge '{knowledge_name}' uploaded successfully"
    }


@app.get("/run-dashboard")
def run_dashboard(background_tasks: BackgroundTasks):
    try:
        port = random.randint(10000, 30000)

        background_tasks.add_task(evaluator.show_dashboard, port=port)
        return {
            "message": f"Trulens dashboard is running. Access it at http://localhost:{port}"
        }
    except Exception as e:
        return {"error": str(e)}


app.router.lifespan_context = lifespan
