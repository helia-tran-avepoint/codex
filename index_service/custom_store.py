import json
import os
import random
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import community.community_louvain as community_louvain
import neo4j
import networkx as nx
import numpy as np
from graspologic.partition import hierarchical_leiden
from llama_index.core import Settings
from llama_index.core.graph_stores.types import (
    ChunkNode,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
    Triplet,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.graph_stores import nebula, neo4j
from llama_index.graph_stores.nebula.nebula_graph_store import (
    QUOTE,
    escape_str,
    hash_string_to_rank,
)
from llama_index.graph_stores.neo4j.neo4j_property_graph import (
    BASE_ENTITY_LABEL,
    BASE_NODE_LABEL,
    CHUNK_SIZE,
    EXCLUDED_LABELS,
    EXCLUDED_RELS,
    EXHAUSTIVE_SEARCH_LIMIT,
    LIST_LIMIT,
    LONG_TEXT_THRESHOLD,
    node_properties_query,
    rel_properties_query,
    rel_query,
    remove_empty_values,
)
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from torch import embedding
from tqdm import tqdm

from index_service import app_config, logger
from index_service.config import INFERENCE_LOCAL_LLM
from index_service.utils import load_json_file, safe_save_file, save_json_file

ENTIYT_INFO_KEY = "./transform_cache/entity_info.json"
NEO4J_SCHEMA_CACHE = "./transform_cache/neo4j_structured_schema_cache.json"


class Neo4jGraphStore(neo4j.Neo4jGraphStore):
    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[:`%s`]->(n2)
        """

        prepared_statement = query % (
            self.node_label,
            self.node_label,
            rel.replace(" ", "_").upper(),
        )

        try:
            with self._driver.session(database=self._database) as session:
                session.run(prepared_statement, {"subj": subj, "obj": obj})  # type: ignore
        except Exception as e:
            logger.error(e)


class NebulaGraphStore(nebula.NebulaGraphStore):
    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        # Note, to enable leveraging existing knowledge graph,
        # the (triplet -- property graph) mapping
        #   makes (n:1) edge_type.prop_name --> triplet.rel
        # thus we have to assume rel to be the first edge_type.prop_name
        # here in upsert_triplet().
        # This applies to the type of entity(tags) with subject and object, too,
        # thus we have to assume subj to be the first entity.tag_name

        # lower case subj, rel, obj
        subj = escape_str(subj)
        rel = escape_str(rel)
        obj = escape_str(obj)
        if self._vid_type == "INT64":
            assert all(
                [subj.isdigit(), obj.isdigit()]
            ), "Subject and object should be digit strings in current graph store."
            subj_field = subj
            obj_field = obj
        else:
            subj_field = f"{QUOTE}{subj}{QUOTE}"
            obj_field = f"{QUOTE}{obj}{QUOTE}"
        edge_field = f"{subj_field}->{obj_field}"

        edge_type = self._edge_types[0]
        rel_prop_name = self._rel_prop_names[0]
        entity_type = self._tags[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {subj_field}:({QUOTE}{subj}{QUOTE});"
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {obj_field}:({QUOTE}{obj}{QUOTE});"
            f"INSERT EDGE `{edge_type}`(`{rel_prop_name}`) "
            f"  VALUES "
            f"{edge_field}"
            f"@{rel_hash}:({QUOTE}{rel}{QUOTE});"
        )
        logger.debug(f"upsert_triplet()\nDML query: {dml_query}")

        try:
            result = self.execute(dml_query)
            assert (
                result and result.is_succeeded()
            ), f"Failed to upsert triplet: {subj} {rel} {obj}, query: {dml_query}"
        except Exception as e:
            logger.error(e)


class NebulaPropertyGraphStore(nebula.NebulaPropertyGraphStore):
    def upsert_relations(self, relations: List[Relation]) -> None:
        """Add relations."""
        schema_ensurence_cache = set()
        for relation in relations:
            keys, values_k, values_params = self._construct_property_query(
                relation.properties
            )
            stmt = f'INSERT EDGE `Relation__` (`label`,{keys}) VALUES "{relation.source_id}"->"{relation.target_id}":("{relation.label}",{values_k});'
            # if relation.label not in schema_ensurence_cache:
            #     if ensure_relation_meta_schema(
            #         relation.source_id,
            #         relation.target_id,
            #         relation.label,
            #         self.structured_schema,
            #         self.client,
            #         relation.properties,
            #     ):
            #         self.refresh_schema()
            #         schema_ensurence_cache.add(relation.label)
            self.structured_query(stmt, param_map=values_params)


class GraphRAGStore(neo4j.Neo4jPropertyGraphStore):
    max_cluster_size = 5

    def __init__(
        self,
        community_summary_vector_store: BasePydanticVectorStore,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.community_summary_vector_store = community_summary_vector_store
        self._entity_info = {}

    @property
    def entity_info(self):
        if self._entity_info is None or self._entity_info == {}:
            self._load_entity_info()
        return self._entity_info

    @entity_info.setter
    def entity_info(self, value):
        self._entity_info = value
        self._cache_entity_info()

    def refresh_schema(self) -> None:
        """Refresh the schema."""
        logger.info("Refresh schema")
        if not self.structured_schema and os.path.exists(NEO4J_SCHEMA_CACHE):
            logger.info("Load structured_schema cache...")
            self.structured_schema = load_json_file(NEO4J_SCHEMA_CACHE)
        else:
            node_query_results = self.structured_query(
                node_properties_query,
                param_map={
                    "EXCLUDED_LABELS": [
                        *EXCLUDED_LABELS,
                        BASE_ENTITY_LABEL,
                        BASE_NODE_LABEL,
                    ]
                },
            )
            node_properties = (
                [el["output"] for el in node_query_results]
                if node_query_results
                else []
            )

            rels_query_result = self.structured_query(
                rel_properties_query, param_map={"EXCLUDED_LABELS": EXCLUDED_RELS}
            )
            rel_properties = (
                [el["output"] for el in rels_query_result] if rels_query_result else []
            )

            rel_objs_query_result = self.structured_query(
                rel_query,
                param_map={
                    "EXCLUDED_LABELS": [
                        *EXCLUDED_LABELS,
                        BASE_ENTITY_LABEL,
                        BASE_NODE_LABEL,
                    ]
                },
            )
            relationships = (
                [el["output"] for el in rel_objs_query_result]
                if rel_objs_query_result
                else []
            )

            # Get constraints & indexes
            try:
                constraint = self.structured_query("SHOW CONSTRAINTS")
                index = self.structured_query(
                    "CALL apoc.schema.nodes() YIELD label, properties, type, size, "
                    "valuesSelectivity WHERE type = 'RANGE' RETURN *, "
                    "size * valuesSelectivity as distinctValues"
                )
            except (
                neo4j.exceptions.ClientError
            ):  # Read-only user might not have access to schema information
                constraint = []
                index = []

            logger.info("Load schema from graph db...")
            self.structured_schema = {
                "node_props": {
                    el["labels"]: el["properties"] for el in node_properties
                },
                "rel_props": {el["type"]: el["properties"] for el in rel_properties},
                "relationships": relationships,
                "metadata": {"constraint": constraint, "index": index},
            }
            logger.info("Load all node and rel from graph db...")
            schema_counts = self.structured_query(
                "CALL apoc.meta.subGraph({}) YIELD nodes, relationships "
                "RETURN nodes, [rel in relationships | {name:apoc.any.property"
                "(rel, 'type'), count: apoc.any.property(rel, 'count')}]"
                " AS relationships"
            )
            logger.info("Load schema from graph db success.")
            # Update node info
            logger.info("Update node info...")
            for node in tqdm(
                schema_counts[0].get("nodes", []), desc="Update node info"
            ):
                # Skip bloom labels
                if node["name"] in EXCLUDED_LABELS:
                    continue
                node_props = self.structured_schema["node_props"].get(node["name"])
                if not node_props:  # The node has no properties
                    continue
                enhanced_cypher = self._enhanced_schema_cypher(
                    node["name"], node_props, node["count"] < EXHAUSTIVE_SEARCH_LIMIT
                )
                enhanced_info = self.structured_query(enhanced_cypher)[0]["output"]
                for prop in node_props:
                    # Map to custom types
                    # Text
                    if prop["type"] == "STRING" and any(
                        len(value) >= LONG_TEXT_THRESHOLD
                        for value in enhanced_info[prop["property"]]["values"]
                    ):
                        enhanced_info[prop["property"]]["type"] = "TEXT"
                    # Embedding
                    if (
                        prop["type"] == "LIST"
                        and enhanced_info[prop["property"]]["max_size"] > LIST_LIMIT
                    ):
                        enhanced_info[prop["property"]]["type"] = "EMBEDDING"
                    if prop["property"] in enhanced_info:
                        prop.update(enhanced_info[prop["property"]])
            # Update rel info
            logger.info("Update rel info...")
            for rel in tqdm(
                schema_counts[0].get("relationships", []), desc="Update rel info"
            ):
                # Skip bloom labels
                if rel["name"] in EXCLUDED_RELS:
                    continue
                rel_props = self.structured_schema["rel_props"].get(rel["name"])
                if not rel_props:  # The rel has no properties
                    continue
                enhanced_cypher = self._enhanced_schema_cypher(
                    rel["name"],
                    rel_props,
                    rel["count"] < EXHAUSTIVE_SEARCH_LIMIT,
                    is_relationship=True,
                )
                try:
                    enhanced_info = self.structured_query(enhanced_cypher)[0]["output"]
                    for prop in rel_props:
                        if prop["property"] in enhanced_info:
                            prop.update(enhanced_info[prop["property"]])
                except neo4j.exceptions.ClientError:
                    # Sometimes the types are not consistent in the db
                    pass
            logger.info("Save structured schema...")
            safe_save_file(
                save_json_file, path=NEO4J_SCHEMA_CACHE, content=self.structured_schema
            )

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        # Lists to hold separated types
        entity_dicts: List[dict] = []
        chunk_dicts: List[dict] = []

        # Sort by type
        for item in nodes:
            if isinstance(item, EntityNode):
                entity_dicts.append({**item.dict(), "id": item.id})
            elif isinstance(item, ChunkNode):
                chunk_dicts.append({**item.dict(), "id": item.id})
            else:
                # Log that we do not support these types of nodes
                # Or raise an error?
                pass

        if chunk_dicts:
            for index in range(0, len(chunk_dicts), CHUNK_SIZE):
                chunked_params = chunk_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.id}})
                    SET c.text = row.text, c:Chunk
                    WITH c, row
                    SET c += row.properties
                    WITH c, row.embedding AS embedding
                    WHERE embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(c, 'embedding', embedding)
                    RETURN count(*)
                    """,
                    param_map={"data": chunked_params},
                )

        if entity_dicts:
            for index in range(0, len(entity_dicts), CHUNK_SIZE):
                chunked_params = entity_dicts[index : index + CHUNK_SIZE]
                self.structured_query(
                    f"""
                    UNWIND $data AS row
                    MERGE (e:{BASE_NODE_LABEL} {{id: row.id}})
                    SET e += apoc.map.clean(row.properties, [], [])
                    SET e.name = row.name, e:`{BASE_ENTITY_LABEL}`
                    WITH e, row
                    CALL apoc.create.addLabels(e, [row.label])
                    YIELD node
                    WITH e, row WHERE row.embedding IS NOT NULL
                    CALL {{
                        WITH e, row
                        CALL db.create.setNodeVectorProperty(e, 'embedding', row.embedding)
                        RETURN count(*) AS count
                    }}
                    WITH e, row WHERE row.properties.triplet_source_id IS NOT NULL
                    MERGE (c:{BASE_NODE_LABEL} {{id: row.properties.triplet_source_id}})
                    MERGE (e)<-[:MENTIONS]-(c)
                    """,
                    param_map={"data": chunked_params},
                )

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        # TODO: handle ids of chunk nodes
        cypher_statement = f"MATCH (e:`{BASE_ENTITY_LABEL}`) "

        params = {}
        if entity_names or properties or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND ".join(prop_list)

        #    WHERE toLower(type(r)) CONTAINS 'call'
        return_statement = f"""
        WITH e
        CALL {{
            WITH e
            MATCH (e)-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]->(t:`{BASE_ENTITY_LABEL}`)
            RETURN e.name AS source_id, [l in labels(e) WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS source_type,
                   e{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   r{{.*}} AS rel_properties,
                   t.name AS target_id, [l in labels(t) WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS target_type,
                   t{{.* , embedding: Null, name: Null}} AS target_properties
            UNION ALL
            WITH e
            MATCH (e)<-[r{':`' + '`|`'.join(relation_names) + '`' if relation_names else ''}]-(t:`{BASE_ENTITY_LABEL}`)
            RETURN t.name AS source_id, [l in labels(t) WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS source_type,
                   t{{.* , embedding: Null, name: Null}} AS source_properties,
                   type(r) AS type,
                   r{{.*}} AS rel_properties,
                   e.name AS target_id, [l in labels(e) WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS target_type,
                   e{{.* , embedding: Null, name: Null}} AS target_properties
        }}
        RETURN source_id, source_type, type, rel_properties, target_id, target_type, source_properties, target_properties"""
        cypher_statement += return_statement

        data = self.structured_query(cypher_statement, param_map=params)
        data = data if data else []

        triples = []
        for record in data:
            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_properties"]),
            )
            triples.append([source, rel, target])
        return triples

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                    "each entity is a method name."
                    "each method is provided in the `METHODS:` section below, formatted as:\n"
                    "```\nMethod Name: [method_name]\nMethod Summary:\n[method_summary]\n```"
                    "Community:\n"
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = INFERENCE_LOCAL_LLM.chat(messages)
        # response = Settings.llm.chat(messages)
        response = re.sub(r"<think>.*</think>", "", str(response), flags=re.DOTALL)
        clean_response = re.sub(r"^assistant:\s*", "", response).strip()
        return clean_response

    def compute_dynamic_cluster_size(self, nx_graph):
        """
        Dynamically compute max_cluster_size based on graph structure.
        """
        num_nodes = nx_graph.number_of_nodes()
        num_edges = nx_graph.number_of_edges()

        # clustering_coeff = nx.average_clustering(nx_graph)
        # logger.debug(f"Average Clustering Coefficient: {clustering_coeff}")

        avg_degree = np.mean([d for _, d in nx_graph.degree()])

        components = [len(c) for c in nx.connected_components(nx_graph)]
        avg_component_size = np.mean(components)

        modularity = avg_degree / num_nodes if num_nodes > 0 else 1

        max_cluster_size = int((avg_component_size + avg_degree) * modularity * 1.5)

        max_cluster_size = max(self.max_cluster_size, min(max_cluster_size, 500))

        logger.info(f"calculate optimized max_cluster_size: {max_cluster_size}")
        return max_cluster_size

    def find_best_resolution(self, nx_graph, resolution_range=(0.5, 2.0), step=0.1):
        best_resolution = 1.0
        best_modularity = -1
        resolution_scores = []

        for res in np.arange(resolution_range[0], resolution_range[1], step):
            partition = community_louvain.best_partition(nx_graph, resolution=res)
            modularity = community_louvain.modularity(partition, nx_graph)
            resolution_scores.append((res, modularity))

            if modularity > best_modularity:
                best_modularity = modularity
                best_resolution = res
                num_communities = len(set(partition.values()))
                logger.info(
                    f"Resolution: {res}, Modularity: {modularity}, Num communities: {num_communities}"
                )

        logger.info(
            f"Best Resolution: {best_resolution}, Best Modularity: {best_modularity}"
        )
        return best_resolution, resolution_scores

    def build_communities(self):
        """Builds communities from the graph and summarizes them."""

        logger.info("Building communities...")
        # optimize cluster size dynamicly
        nx_graph = self._create_nx_graph()

        if nx_graph.number_of_edges() == 0:
            logger.info("Skip building communities as the graph is empty.")
        else:
            best_resolution, resolution_scores = self.find_best_resolution(nx_graph)

            partition = community_louvain.best_partition(
                nx_graph, weight="weight", resolution=best_resolution, random_state=42
            )

            # self.max_cluster_size = self.compute_dynamic_cluster_size(nx_graph)

            # community_hierarchical_clusters = hierarchical_leiden(
            #     nx_graph, max_cluster_size=self.max_cluster_size
            # )
            self.entity_info, community_info = self._collect_community_info(
                nx_graph, partition  # community_hierarchical_clusters
            )
            community_summary = self._summarize_communities(community_info)
            # community_summary = self._concurrent_summarize_communities(community_info)
            self.generate_community_embedding(community_summary)

            logger.info("==============================")
            logger.info("Build communities completed.")
            logger.info("==============================")

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.id)  # type: ignore
            nx_graph.add_node(entity2.id)  # type: ignore
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                weight=(
                    10
                    if (
                        "call" in relation.label.lower()
                        or "invoke" in relation.label.lower()
                    )
                    else 1
                ),
                description=(
                    relation.properties["relationship_description"]
                    if "relationship_description" in relation.properties
                    else relation.label
                ),
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, partition):  # clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        cluster_id_mapping = {}

        for node, cluster_id_int in partition.items():
            if cluster_id_int in cluster_id_mapping:
                cluster_id = cluster_id_mapping[cluster_id_int]
            else:
                cluster_id = str(uuid.uuid4())
                cluster_id_mapping[cluster_id_int] = cluster_id

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    node_body = self.get(properties={"id": node})
                    neighbor_body = self.get(properties={"id": neighbor})

                    if len(node_body) > 1 or len(neighbor_body) > 1:
                        print(
                            "Warning: collect community info assume only one node for each name"
                        )
                    node_body_summary = node_body[0].properties.get("description", "")
                    neighbor_body_summary = neighbor_body[0].properties.get(
                        "description", ""
                    )
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']} \n\n METHODS --\nMethod Name: {node}\nMethod Summary:\n{node_body_summary}\n-- \n\n METHODS --\nMethod Name: {neighbor}\nMethod Summary:\n{neighbor_body_summary}\n--"
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_single_community(self, args):
        community_id, details, generate_fn = args
        details_text = "\n".join(details) + "."
        summary = generate_fn(details_text)
        return community_id, summary

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""

        communitys_with_progress = get_tqdm_iterable(
            community_info.items(), show_progress=True, desc="Summarize communities"
        )

        community_summary = {}
        for community_id, details in communitys_with_progress:
            if community_id not in community_summary:
                details_text = "\n".join(details) + "."
                community_summary[community_id] = self.generate_community_summary(
                    details_text
                )

        return community_summary

    def _concurrent_summarize_communities(self, community_info):
        from shared.concurrent_utils import run_jobs_threadpool

        community_summary = {}
        results = run_jobs_threadpool(
            [
                lambda community_id=community_id, details=details: self._summarize_single_community(
                    (community_id, details, self.generate_community_summary)
                )
                for community_id, details in community_info.items()
                if community_id not in community_summary
            ],
            workers=8,
            show_progress=True,
            desc="Summarize communities",
        )

        for community_id, summary in results:
            community_summary[community_id] = summary

        return community_summary

    def generate_community_embedding(self, community_summary):
        c_ids, c_summaries = zip(*community_summary.items())

        s_embeddings = Settings.embed_model.get_text_embedding_batch(c_summaries)

        communitys_with_progress = get_tqdm_iterable(
            zip(c_ids, c_summaries, s_embeddings),
            show_progress=True,
            desc="Summarize communities",
        )

        community_to_nodes = {}
        for node, clusters in self.entity_info.items():
            for cluster_id in clusters:
                community_to_nodes.setdefault(cluster_id, []).append(node)

        community_docs = []
        for community_id, summary, embedding in communitys_with_progress:
            community_docs.append(
                Document(
                    doc_id=str(community_id),
                    text=summary,
                    embedding=embedding,
                    extra_info={"nodes": community_to_nodes.get(community_id, [])},
                )
            )

        self.community_summary_vector_store.add(community_docs)

    def _cache_entity_info(self):
        with open(ENTIYT_INFO_KEY, "w") as f:
            json.dump(self.entity_info, f)

    def _load_entity_info(self):
        if os.path.exists(ENTIYT_INFO_KEY):
            with open(ENTIYT_INFO_KEY, "r") as f:
                self.entity_info = json.load(f)
            return True
        else:
            return False

    def get_schema_json(self, refresh: bool = False) -> str:
        schema = self.get_schema(refresh=refresh)
        schema_json = {"nodes": {}, "relationships": []}

        for node, props in schema["node_props"].items():
            schema_json["nodes"][node] = {
                "properties": {prop["property"]: prop["type"] for prop in props}
            }

        for rel in schema["relationships"]:
            schema_json["relationships"].append(
                {"start": rel["start"], "type": rel["type"], "end": rel["end"]}
            )

        return json.dumps(schema_json, indent=2)
