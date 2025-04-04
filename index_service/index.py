import asyncio
import json
import multiprocessing
import os
import re
from concurrent.futures import ThreadPoolExecutor
from ctypes import util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin

import aiohttp
import chromadb
import chromadb.config
import networkx as nx
import qdrant_client
from chromadb import PersistentClient
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion.cache import IngestionCache
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_SUMMARY_PROMPT
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.utils import get_tqdm_iterable
from llama_index.graph_stores.nebula.nebula_property_graph import DEFAULT_PROPS_SCHEMA
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams

from index_service import app_config, logger
from index_service.config import (
    INFERENCE_LOCAL_LLM,
    KNOWLEDGE_EXTRACT_LOCAL_LLM,
    get_data_config,
)
from index_service.custom_indexer import (
    KnowledgeGraphIndex,
    PropertyGraphIndex,
    TreeIndex,
    VectorStoreIndex,
)
from index_service.custom_store import (
    GraphRAGStore,
    NebulaGraphStore,
    NebulaPropertyGraphStore,
    Neo4jGraphStore,
)
from index_service.custom_transform import (
    KG_TRIPLET_EXTRACT_TMPL,
    GraphRAGExtractor,
    ImplicitPathExtractor,
    SentenceSplitter,
    SimpleLLMPathExtractor,
    SimpleRoslynPathExtractor,
)
from index_service.model import MethodInfo
from index_service.utils import (
    DocCacheManager,
    construct_dir_path,
    get_collection_name,
    get_embedding_size,
    get_text_hash,
    get_text_md5,
    rename_files_in_directory,
)
from shared import constants, utils
from shared.concurrent_utils import run_jobs_threadpool
from shared.models import Project

persist_directory = app_config.persist_dir

CUSTOM_PROPS_SCHEMA = f"{DEFAULT_PROPS_SCHEMA},`name` STRING,`description` STRING,`inheritsFrom` STRING,`className` STRING,`namespaceName` STRING,`comments` STRING,`returnType` STRING"


def build_vector_index(
    documents, force_rebuild, data_type, project_name, knowledge_name
):
    logger.debug("Begin build vector index")

    vector_path = construct_dir_path(
        parts=[
            project_name,
            "index",
            data_type,
            knowledge_name,
            "vector",
        ]
    )

    #   chroma_settings = chromadb.config.Settings(
    #       anonymized_telemetry=False,
    #   )

    #   chroma_client = PersistentClient(
    #       path=str(vector_path),
    #       settings=chroma_settings,
    #   )

    client = qdrant_client.QdrantClient(
        # you can use :memory: mode for fast and light-weight experiments,
        # it does not require to have Qdrant deployed anywhere
        # but requires qdrant-client >= 1.1.1
        # location=":memory:"
        # otherwise set Qdrant instance address with:
        # url="http://<host>:<port>"
        # otherwise set Qdrant instance with host and port:
        host="10.1.70.240",
        port=6333,
        # set API KEY for Qdrant Cloud
        # api_key="<qdrant-api-key>",
    )
    aclient = qdrant_client.AsyncQdrantClient(
        host="10.1.70.240",
        port=6333,
    )

    # TODO
    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=get_collection_name(project_name, data_type, knowledge_name),
    )

    #   chroma_collection = chroma_client.get_or_create_collection(
    #      constants.BUSINESS_DOCUMENT_COLLECTION
    #   )

    #   vector_store = ChromaVectorStore(
    #      chroma_collection=chroma_collection, persist_dir=str(vector_path)
    #  )

    if not force_rebuild:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(vector_path)
        )
        logger.info("Load existing VectorStoreIndex...")
        index = load_index_from_storage(storage_context)
        # make sure this property is true even though set it in init method
        index._store_nodes_override = True
        # index.refresh_ref_docs(documents=documents)
        logger.info("VectorStoreIndex load successfully.")
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Building VectorStoreIndex...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.debug("Save VectorStoreIndex storage context...")
        index.storage_context.persist(vector_path)
        logger.debug("VectorStoreIndex storage context save successfully.")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(vector_path)
        )
        clean_up_indexes(storage_context)
        logger.info("VectorStoreIndex built successfully.")

    return index


def build_tree_index(documents, force_rebuild, data_type, project_name, knowledge_name):
    logger.debug("Begin build tree index")

    tree_path = construct_dir_path(
        parts=[
            project_name,
            "index",
            data_type,
            knowledge_name,
            "tree",
        ]
    )

    if not force_rebuild:
        storage_context = StorageContext.from_defaults(persist_dir=str(tree_path))

        logger.info("Load existing TreeIndex...")
        tree_index = load_index_from_storage(storage_context)
        logger.info("TreeIndex load successfully.")
    else:
        storage_context = StorageContext.from_defaults()

        logger.info("Building TreeIndex...")
        tree_index = TreeIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.debug("Save TreeIndex storage context...")
        tree_index.storage_context.persist(tree_path)
        logger.debug("TreeIndex storage context save successfully.")
        storage_context = StorageContext.from_defaults(persist_dir=str(tree_path))
        clean_up_indexes(storage_context)
        logger.info("TreeIndex built successfully.")

    return tree_index


def build_graph_index(
    documents, force_rebuild, data_type, project_name, knowledge_name
):
    logger.debug("Begin build graph index")

    # space_name = app_config.space_name

    # graph_store = GraphRAGStore(
    #     space=space_name,
    #     overwrite=True,
    #     username=app_config.nebula_user,
    #     password=app_config.nebula_password,
    #     url=f"nebula://{app_config.nebula_address}",
    #     props_schema=CUSTOM_PROPS_SCHEMA,
    # )
    community_summary_vector_store = get_or_create_vector_store(
        get_collection_name(
            project_name, data_type, knowledge_name, "community_summary"
        )
    )
    graph_store = GraphRAGStore(
        username=app_config.graph_db_username,
        password=app_config.graph_db_password,
        url=app_config.graph_db_url,
        database=app_config.graph_db_database_name,
        refresh_schema=True,
        sanitize_query_output=False,
        community_summary_vector_store=community_summary_vector_store,
    )

    graph_path = construct_dir_path(
        parts=[
            project_name,
            "index",
            data_type,
            knowledge_name,
            "graph",
        ]
    )

    collection_name = get_collection_name(project_name, data_type, knowledge_name)
    vector_store = get_or_create_vector_store(collection_name)
    method_name_vector_store = get_or_create_vector_store(
        get_collection_name(project_name, data_type, knowledge_name, "method_name")
    )

    entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
    relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

    def parse_fn(response_str: str) -> Any:
        entities = re.findall(entity_pattern, response_str)
        relationships = re.findall(relationship_pattern, response_str)
        return entities, relationships

    kg_extractor = GraphRAGExtractor(
        llm=KNOWLEDGE_EXTRACT_LOCAL_LLM,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=20,
        parse_fn=parse_fn,
    )

    if not force_rebuild:
        storage_context = StorageContext.from_defaults(
            property_graph_store=graph_store, persist_dir=str(graph_path)
        )
        logger.info("Load existing PropertyGraphIndex...")

        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            vector_store=vector_store,
            method_name_vector_store=method_name_vector_store,
            storage_context=storage_context,
            use_async=False,
            llm=INFERENCE_LOCAL_LLM,
            # llm=AzureOpenAI(
            #     model=app_config.azure_openai_model_name,
            #     deployment_name=app_config.azure_openai_deployment_name,
            #     api_key=app_config.azure_openai_api_key,
            #     azure_endpoint=app_config.azure_openai_host,
            #     api_version=app_config.azure_openai_api_version,
            # ),
            kg_extractors=(
                # [kg_extractor, SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                [SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                if data_type == "source_code"
                else [
                    SimpleLLMPathExtractor(),
                    ImplicitPathExtractor(),
                ]
            ),
        )
    else:
        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
        logger.info("Building PropertyGraphIndex...")

        graph_index = PropertyGraphIndex.from_documents(
            documents,
            use_async=False,
            property_graph_store=graph_store,
            vector_store=vector_store,
            method_name_vector_store=method_name_vector_store,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            show_progress=True,
            kg_extractors=(
                # [kg_extractor, SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                [SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                if data_type == "source_code"
                else [
                    SimpleLLMPathExtractor(),
                    ImplicitPathExtractor(),
                ]
            ),
            # cache=cache,
        )

        graph_index.property_graph_store.build_communities()  # type: ignore

        logger.info("Begin persist context")
        graph_index.storage_context.persist(graph_path)
        # graph_index.storage_context.vector_store.persist(graph_path)
        storage_context = StorageContext.from_defaults(
            property_graph_store=graph_store, persist_dir=str(graph_path)
        )
        clean_up_indexes(storage_context)
        logger.info("PropertyGraphIndex built successfully.")

    return graph_index


def build_graph_index_obsolete(
    documents, force_rebuild, data_type, project_name, knowledge_name
):
    logger.debug("Begin build graph index")

    # graph_store = Neo4jGraphStore(
    #     username=app_config.graph_db_username,
    #     password=app_config.graph_db_password,
    #     url=app_config.graph_db_url,
    #     database=app_config.graph_db_database_name,
    # )
    space_name = app_config.space_name
    # edge_types, rel_prop_names = ["relationship"], ["relationship"]
    # tags = ["entity"]
    # graph_store = NebulaGraphStore(
    #     space_name=space_name,
    #     edge_types=edge_types,
    #     rel_prop_names=rel_prop_names,
    #     tags=tags,
    # )

    graph_store = NebulaPropertyGraphStore(
        space=space_name,
        overwrite=True,
        username=app_config.nebula_user,
        password=app_config.nebula_password,
        url=f"nebula://{app_config.nebula_address}",
        props_schema=CUSTOM_PROPS_SCHEMA,
    )

    graph_path = construct_dir_path(
        parts=[
            project_name,
            "index",
            data_type,
            knowledge_name,
            "graph",
        ]
    )

    chroma_settings = chromadb.config.Settings(
        anonymized_telemetry=False,
    )

    chroma_client = PersistentClient(
        path=str(graph_path),
        settings=chroma_settings,
    )
    chroma_collection = chroma_client.get_or_create_collection(space_name)

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection, persist_dir=str(graph_path)
    )

    if not force_rebuild:
        storage_context = StorageContext.from_defaults(
            property_graph_store=graph_store, persist_dir=str(graph_path)
        )
        logger.info("Load existing PropertyGraphIndex...")
        # graph_index = load_index_from_storage(storage_context)

        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            vector_store=vector_store,
            storage_context=storage_context,
            use_async=False,
            kg_extractors=(
                [SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                if data_type == "source_code"
                else [
                    SimpleLLMPathExtractor(),
                    ImplicitPathExtractor(),
                ]
            ),
        )
        logger.info("PropertyGraphIndex load successfully.")

    else:
        # cache_persist_path = f"./transform_cache/{project_name}/{knowledge_name}/graph"
        # cache = (
        #     IngestionCache.from_persist_path(cache_persist_path)
        #     if Path(cache_persist_path).exists()
        #     else IngestionCache()
        # )

        storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
        logger.info("Building PropertyGraphIndex...")

        graph_index = PropertyGraphIndex.from_documents(
            documents,
            use_async=False,
            property_graph_store=graph_store,
            vector_store=vector_store,
            storage_context=storage_context,
            max_triplets_per_chunk=2,
            show_progress=True,
            kg_extractors=(
                [SimpleRoslynPathExtractor(), ImplicitPathExtractor()]
                if data_type == "source_code"
                else [
                    SimpleLLMPathExtractor(),
                    ImplicitPathExtractor(),
                ]
            ),
            # cache=cache,
        )

        # graph_index = KnowledgeGraphIndex.from_documents(
        #     documents,
        #     storage_context=storage_context,
        #     show_progress=True,
        #     max_triplets_per_chunk=2,
        #     space_name=space_name,
        #     edge_types=edge_types,
        #     rel_prop_names=rel_prop_names,
        #     tags=tags,
        # )
        # cache.persist(cache_persist_path)
        logger.debug("Save PropertyGraphIndex storage context...")
        graph_index.storage_context.persist(graph_path)
        logger.debug("PropertyGraphIndex storage context save successfully.")
        # graph_index.storage_context.vector_store.persist(graph_path)
        storage_context = StorageContext.from_defaults(
            property_graph_store=graph_store, persist_dir=str(graph_path)
        )
        clean_up_indexes(storage_context)
        logger.info("PropertyGraphIndex built successfully.")

    return graph_index


def get_or_create_vector_store(collection_name):
    client = qdrant_client.QdrantClient(
        host="10.1.70.240",
        port=6333,
    )
    aclient = qdrant_client.AsyncQdrantClient(
        host="10.1.70.240",
        port=6333,
    )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name,
            vectors_config=VectorParams(
                size=get_embedding_size(), distance=Distance.COSINE
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=collection_name,
    )

    return vector_store


def clean_up_indexes(storage_context: StorageContext):
    # index_ids = storage_context.index_store.index_structs()
    # for index_id in index_ids:
    #     storage_context.index_store.delete_index_struct(index_id.index_id)
    pass


async def analysis_source_code(path):
    logger.debug("Begin analysis source code")

    url = f"http://{constants.ANALYSIS_SERVICE_URL}:{app_config.analysis_service_port}/analysis"
    payload = {"Path": path}
    headers = {"Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                logger.debug("Analysis source code successfuly.")
                return await response.json()
            else:
                logger.debug("Analysis source code failed.")
                response.raise_for_status()


def update_index_path(project_name: str, knowledge_name: str, data_type: str):
    logger.debug("Begin update index path")

    project = Project.load(project_name)
    for knowledge in project.knowledges:
        if knowledge.knowledge_name == knowledge_name:
            knowledge.index_dir = "./" + str(
                Path(persist_directory)
                / project_name
                / "index"
                / data_type
                / knowledge_name
            )
            knowledge.indexed = True
            knowledge.last_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            project.update_knowledge(knowledge)
            logger.debug(
                f"Update project: {project_name}, knowledge: {knowledge_name} successfully."
            )
            continue


def generate_method_summary(text):
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are an expert in software development and code documentation. Your task is to analyze the given C# method and generate a comprehensive summary that captures every important detail. The summary should be clear, structured, and suitable for documentation."
                "Please ensure that your summary includes:"
                "- The method’s **purpose** (what it does and why it exists)."
                "- A detailed description of **input parameters**, including their types and expected values."
                "- A detailed description of the **return value**, including its type and possible outputs."
                "- Any **exceptions** the method might throw and under what conditions."
                "- Any **dependencies** or related methods/classes that it interacts with."
                "- A high-level explanation of the **algorithm or logic** used in the method."
                "- Any **edge cases** or special considerations when using this method."
                "**Example format for the response:**"
                "Method Name: {MethodName}"
                "Summary: {A clear and structured explanation of what the method does, why it is needed, and how it works.}"
                "Parameters:"
                "{param1_name} ({param1_type}): {Description of what this parameter represents and how it affects the method.}"
                "{param2_name} ({param2_type}): {Description of what this parameter represents and how it affects the method.} ..."
                "Return Value: {ReturnType} - {Description of the returned data and possible variations.}"
                "Exceptions:"
                "{ExceptionType1}: {Condition under which this exception is thrown.}"
                "{ExceptionType2}: {Condition under which this exception is thrown.}"
                "Dependencies & Interactions: {Describe any classes, libraries, or methods that this function interacts with.}"
                "Logic & Algorithm Explanation: {Step-by-step breakdown of the key operations performed within the method.}"
                "Edge Cases & Considerations: {List any unusual conditions, boundary values, or specific use cases where the function behaves differently.}"
                "**Now, analyze the following C# method and generate its summary accordingly:**"
            ),
        ),
        ChatMessage(role="user", content=f"```csharp:\n{text}```"),
    ]
    response = INFERENCE_LOCAL_LLM.chat(messages)
    response = re.sub(r"<think>.*</think>", "", str(response), flags=re.DOTALL)
    clean_response = re.sub(r"^assistant:\s*", "", response).strip()
    return clean_response


async def build_indexes(
    path: str,
    data_type: str,
    project_name: str,
    knowledge_name: str,
    force_rebuild: bool = False,
    is_parallel: bool = False,
    tasks: Optional[List[Literal["vector", "tree", "graph"]]] = None,
) -> Dict[str, BaseIndex]:
    logger.debug(
        f"Begin build index -- force rebuild {force_rebuild} -- project name {project_name} -- knowledage name {knowledge_name}"
    )

    Settings.chunk_size = 1024 * 2
    if data_type == "source_code" and force_rebuild:
        Settings.chunk_size = 1024 * 10

        source_code_root = Path(path).parent / f"{Path(path).name}_converted"

        if not source_code_root.exists():
            analysis_result = await analysis_source_code(path)

            if analysis_result:
                methodInfos_with_progress = get_tqdm_iterable(
                    analysis_result["methodInfos"],
                    show_progress=True,
                    desc="Summarize and dump methods",
                )
                for methodInfo in methodInfos_with_progress:
                    if methodInfo is not None:
                        methodObj = MethodInfo.model_validate(methodInfo)
                        methodObj.description = generate_method_summary(
                            methodObj.sourceCode
                        )
                        file_path = (
                            source_code_root
                            / f"{get_text_hash(methodObj.namespaceName)}"
                            / f"{get_text_hash(methodObj.className)}"
                            / f"{get_text_hash(methodObj.name + methodObj.returnType)}.json"
                        )

                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.parent.chmod(0o774)

                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(methodObj.model_dump(), f, indent=4)

        path = str(source_code_root)

        logger.debug(f"Format file name which in {path}...")
        rename_files_in_directory(path)

    data_config = get_data_config(data_type)

    task_map = {
        "vector": build_vector_index,
        "tree": build_tree_index,
        "graph": build_graph_index,
    }

    merged_tasks = tasks or task_map.keys()
    selected_tasks = {name: task_map[name] for name in merged_tasks if name in task_map}

    results = {}

    if force_rebuild:
        logger.debug("Load documents...")
        documents = SimpleDirectoryReader(
            path,
            recursive=True,
            num_files_limit=(
                app_config.num_files_limit if app_config.num_files_limit >= 0 else None
            ),
            file_extractor=data_config["file_extractor"],
            required_exts=data_config["required_exts"],
        ).load_data(
            show_progress=True
        )  # , num_workers=multiprocessing.cpu_count()
    else:
        logger.debug("Use empty documents.")
        documents = []

    tmp_documents = []

    with DocCacheManager() as cache:
        for doc in documents:
            md5_text = get_text_md5(doc.text)
            if md5_text in cache.doc_cache:
                logger.info(f"------- ignore document : {doc.extra_info}")
            else:
                cache.doc_cache[md5_text] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                doc.metadata["data_type"] = data_type
                tmp_documents.append(doc)

    documents = tmp_documents.copy()

    if is_parallel:
        logger.info("Running tasks in parallel...")
        max_workers = len(selected_tasks)
    else:
        logger.info("Running tasks sequentially...")
        max_workers = 1

    results_list = run_jobs_threadpool(
        [
            lambda task=task: task(
                documents,
                force_rebuild,
                data_type,
                project_name,
                knowledge_name,
            )
            for task in selected_tasks.values()
        ],
        workers=max_workers,
        show_progress=True,
        desc="Build/Load Indexs",
    )
    results = dict(zip(selected_tasks.keys(), results_list))
    if force_rebuild and results_list:
        update_index_path(project_name, knowledge_name, data_type)

    return results
