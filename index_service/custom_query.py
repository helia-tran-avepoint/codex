import json
import re
import time
from typing import List

import numpy as np
from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore

from index_service import logger
from index_service.custom_indexer import PropertyGraphIndex
from index_service.custom_retriever import LLMSynonymRetriever
from index_service.custom_store import GraphRAGStore
from shared.concurrent_utils import run_jobs_threadpool


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    vector_query_engine: BaseQueryEngine
    extra_method_name_vector_stores: List[QdrantVectorStore]
    extra_community_summary_vector_stores: List[QdrantVectorStore]
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 5

    def query_community_by_entity(self, query_str: str, query_embeding: List[float]):
        """Query community by method name"""
        entities, entities_info = self.get_entities(
            query_str, query_embeding, self.similarity_top_k
        )

        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )

        return {"ids": community_ids, "info": entities_info}

    def query_community_by_summary(self, query_str: str, query_embeding: List[float]):
        """Query community by community summary"""
        logger.warning(f"Only use first extra method vector store.")
        communities = self.extra_community_summary_vector_stores[0].query(
            VectorStoreQuery(
                query_embedding=query_embeding,
                # query_str=query_str,
                similarity_top_k=2,
                # mode=VectorStoreQueryMode.HYBRID,
            )
        )
        community_ids = communities.ids
        return community_ids

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""

        query_embeding = Settings.embed_model.get_text_embedding(query_str)
        community_ids = run_jobs_threadpool(
            [
                lambda query_action=query_action: query_action(
                    query_str, query_embeding
                )
                for query_action in [
                    self.query_community_by_entity,
                    self.query_community_by_summary,
                ]
            ],
            show_progress=True,
            desc="Query community...",
        )
        entities_info = {}
        if community_ids:
            merge_community_ids = []
            for i in community_ids:
                if isinstance(i, dict):
                    merge_community_ids.extend(i.get("ids", []))
                    entities_info = i.get("info", {})
                else:
                    merge_community_ids.extend(i)
            community_ids = list(set(merge_community_ids))
        if len(community_ids) == 0:
            community_answers = []
            return "No relevant information found."
        else:
            community_vector_store = self.extra_community_summary_vector_stores[0]
            logger.debug("Get node from community vector store")
            community_summaries = community_vector_store.get_nodes(
                node_ids=community_ids
            )
            logger.debug("Generate answer from comunity...")
            community_answers = run_jobs_threadpool(
                [
                    lambda community_summary=community_summary: self.generate_answer_from_summary(
                        community_summary.text, query_str
                    )
                    for community_summary in community_summaries
                ],
                show_progress=True,
                desc="Generate answer from each community summary...",
            )
            # vector_result = self.vector_query_engine.query(query_str)

            # final_answer = self.aggregate_answers(
            #     structured_results, entities, community_answers, vector_result, query_str
            # )
            # return final_answer if final_answer else "No relevant information found."
            for method_name in entities_info.keys():
                if method_name in query_str.split(" "):
                    return (
                        "There are method implementation and code location which need show in answer.\n"
                        + entities_info[method_name]
                    )
            logger.debug("Merge all answers...")
            start_time = time.time()
            community_answers.extend(list(entities_info.values()))
            final_answer = self.aggregate_answers(community_answers)
            logger.debug(
                f"Merge all answers spend: {time.time() - start_time:.4f} seconds"
            )

            return final_answer

    def generate_cypher_query(self, query_str: str) -> str:
        schema_json = self.graph_store.get_schema_json()

        prompt = f"""
        You are a Neo4j database expert. The Neo4j version is 5.15.0. Your task is to convert a natural language query into a Cypher query.

        **Database Schema:**
        ```json
        {schema_json}
        ```

        **User Query:**
        "{query_str}"

        **Generate a valid Cypher query that retrieves data matching the user's request.**
        **Only return the Cypher query, no extra text.**
        """

        messages = [
            ChatMessage(
                role="system", content="You are a 5.15.0 version Neo4j expert."
            ),
            ChatMessage(role="user", content=prompt),
        ]

        response = self.llm.chat(messages)
        response = re.sub(r"<think>.*</think>", "", str(response), flags=re.DOTALL)

        match = re.search(r"```cypher\n(.*?)\n```", str(response), re.DOTALL)
        if match:
            cypher_query = match.group(1).strip()

        match = re.search(
            r"(?i)(MATCH|CALL|CREATE|MERGE|RETURN)[^`]*", str(response), re.DOTALL
        )
        if match:
            cypher_query = match.group(0).strip()

        if cypher_query.lower().startswith("match") or "return" in cypher_query.lower():
            return cypher_query
        else:
            return ""

    def get_method_location(self, path):
        change_path = path.replace("/app/", "./")
        with open(change_path, "r") as f:
            method_info = json.load(f)

        if "fileLocation" in method_info:
            return method_info["fileLocation"]
        else:
            return ""

    def get_entities(self, query_str, query_embeding, similarity_top_k):
        if not query_str:
            return []
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        logger.warning(f"Only use first extra method vector store.")
        nodes_by_method_name = self.extra_method_name_vector_stores[0].query(
            VectorStoreQuery(
                query_embedding=query_embeding,
                # query_str=query_str,
                similarity_top_k=similarity_top_k,
                # mode=VectorStoreQueryMode.HYBRID,
            )
        )

        enitites = set()
        enitites_info = {}
        similary_nodes = self.graph_store.get(ids=nodes_by_method_name.ids)
        for node in similary_nodes:
            if node.properties["name"] not in enitites:
                enitites_info[node.properties["name"]] = (
                    "**Source Code**\n"
                    + node.properties["sourceCode"]
                    + "\n**Location**\n"
                    + self.get_method_location(node.properties["file_path"])
                )
            enitites.add(node.properties["name"])

        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites), enitites_info

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

        # def aggregate_answers(
        #     self, structured_results, entities, community_answers, vector_results, query_str
        # ):

    def aggregate_answers(self, community_answers):
        prompt = (
            "You are analyzing code explanations. Your goal is to consolidate the key points "
            "from multiple intermediate answers into a **concise but precise response**.\n\n"
            "**Rules:**\n"
            "1. **DO NOT** invent or create new methods that do not exist in the given answers.\n"
            "2. **Only extract and combine** existing method names, parameters, and code locations.\n"
            "3. **Preserve important details** such as `PersonalInfoCallback`, `sgSingpassService.CreateSingpassLoginUrl()`, etc.\n"
            "4. **If multiple answers mention the same method, merge their explanations.**\n"
            "5. **Your response should focus on identifying relevant code locations and their responsibilities.**\n\n"
            "Now, based on the following Intermediate answers, generate a final, refined response.\n"
        )

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"\n **Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        final_response = re.sub(
            r"<think>.*</think>", "", str(final_response), flags=re.DOTALL
        )
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response

        # results_summary = []

        # if structured_results:
        #     formatted_structured = "\n".join(
        #         [
        #             ", ".join(f"{k}: {v}" for k, v in res.items())
        #             for res in structured_results
        #         ]
        #     )
        #     results_summary.append(f"Structured Data:\n{formatted_structured}")

        # if entities:
        #     formatted_entities = ", ".join(entities)
        #     results_summary.append(f"Related Entities: {formatted_entities}")

        # if community_answers:
        #     results_summary.append(
        #         f"Community Insights:\n{'\n'.join(community_answers)}"
        #     )

        # if vector_results:
        #     results_summary.append(f"Vector Query Result:\n{'\n'.join(vector_results)}")

        # return "\n".join(results_summary)
        # if results_summary:
        #     messages = [
        #         ChatMessage(role="system", content=prompt),
        #         ChatMessage(
        #             role="user",
        #             content=(
        #                 "Intermediate answers:\n"
        #                 f"Here is all the retrieved knowledge:\n{'\n'.join(results_summary)}\n\n"
        #                 f"Provide a final concise answer to this user query: '{query_str}'"
        #             ),
        #         ),
        #     ]
        #     final_response = self.llm.chat(messages)
        #     final_response = re.sub(
        #         r"<think>.*</think>", "", str(final_response), flags=re.DOTALL
        #     )
        #     cleaned_final_response = re.sub(
        #         r"^assistant:\s*", "", str(final_response)
        #     ).strip()
        #     return cleaned_final_response

        # return ""
