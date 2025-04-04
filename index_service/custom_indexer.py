import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import llama_index.core
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.data_structs import IndexLPG
from llama_index.core.data_structs.data_structs import IndexDict, IndexGraph
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    TRIPLET_SOURCE_KEY,
    VECTOR_SOURCE_KEY,
    EntityNode,
    LabelledNode,
    PropertyGraphStore,
    Relation,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.core.ingestion.pipeline import (
    arun_transformations,
    run_transformations,
)
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    IndexNode,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

if TYPE_CHECKING:
    from llama_index.core.indices.property_graph.sub_retrievers.base import (
        BasePGRetriever,
    )


class KnowledgeGraphIndex(llama_index.core.KnowledgeGraphIndex):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ValueError, TimeoutError)),
    )
    def _llm_extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        response = self._llm.predict(
            self.kg_triplet_extract_template,
            text=text,
        )
        return self._parse_triplet_response(
            response, max_length=self._max_object_length
        )


class VectorStoreIndex(llama_index.core.VectorStoreIndex):
    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        # vector store index params
        use_async: bool = False,
        store_nodes_override: bool = False,
        embed_model: Optional[EmbedType] = None,
        insert_batch_size: int = 2048,
        # parent class params
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexDict] = None,
        storage_context: Optional[StorageContext] = None,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        super().__init__(
            nodes=nodes,
            use_async=use_async,
            store_nodes_override=True,
            embed_model=embed_model,
            insert_batch_size=insert_batch_size,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            index_struct=index_struct,
            callback_manager=callback_manager,
            transformations=transformations,
            **kwargs,
        )

    def _build_index_from_nodes(
        self,
        nodes: Sequence[BaseNode],
        **insert_kwargs: Any,
    ) -> IndexDict:
        return super()._build_index_from_nodes(nodes, **insert_kwargs)


class TreeIndex(llama_index.core.TreeIndex):
    def _build_index_from_nodes(
        self, nodes: Sequence[BaseNode], **build_kwargs: Any
    ) -> IndexGraph:
        return super()._build_index_from_nodes(nodes, **build_kwargs)


class PropertyGraphIndex(llama_index.core.PropertyGraphIndex):
    def __init__(
        self,
        method_name_vector_store: BasePydanticVectorStore,
        method_summary_vector_store: Optional[BasePydanticVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        self.method_name_vector_store = method_name_vector_store
        self.method_summary_vector_store = method_summary_vector_store
        super().__init__(**kwargs)

    # def _insert_nodes(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
    #     """Insert nodes to the index struct."""
    #     if len(nodes) == 0:
    #         return nodes

    #     # run transformations on nodes to extract triplets
    #     if self._use_async:
    #         nodes = asyncio.run(
    #             arun_transformations(
    #                 nodes,
    #                 self._kg_extractors,
    #                 show_progress=self._show_progress,
    #                 # **kwargs,
    #             )
    #         )
    #     else:
    #         nodes = run_transformations(
    #             nodes,
    #             self._kg_extractors,
    #             show_progress=self._show_progress,  # , **kwargs
    #         )

    #     # ensure all nodes have nodes and/or relations in metadata
    #     assert all(
    #         node.metadata.get(KG_NODES_KEY) is not None
    #         or node.metadata.get(KG_RELATIONS_KEY) is not None
    #         for node in nodes
    #     )

    #     kg_nodes_to_insert: List[EntityNode] = []
    #     kg_rels_to_insert: List[Relation] = []
    #     for node in nodes:
    #         # remove nodes and relations from metadata
    #         kg_nodes = node.metadata.pop(KG_NODES_KEY, [])
    #         kg_rels = node.metadata.pop(KG_RELATIONS_KEY, [])

    #         kg_nodes = [
    #             EntityNode.model_validate(kg_node)
    #             for kg_node in kg_nodes
    #             if isinstance(kg_node, dict)
    #         ]
    #         kg_rels = [
    #             Relation.model_validate(kg_rel)
    #             for kg_rel in kg_rels
    #             if isinstance(kg_rel, dict)
    #         ]

    #         # add source id to properties
    #         for kg_node in kg_nodes:
    #             kg_node.properties[TRIPLET_SOURCE_KEY] = node.id_
    #         for kg_rel in kg_rels:
    #             kg_rel.properties[TRIPLET_SOURCE_KEY] = node.id_

    #         # add nodes and relations to insert lists
    #         kg_nodes_to_insert.extend(kg_nodes)
    #         kg_rels_to_insert.extend(kg_rels)

    #     # filter out duplicate kg nodes
    #     kg_node_ids = {node.id for node in kg_nodes_to_insert}
    #     existing_kg_nodes = self.property_graph_store.get(ids=list(kg_node_ids))
    #     existing_kg_node_ids = {node.id for node in existing_kg_nodes}
    #     kg_nodes_to_insert = [
    #         node for node in kg_nodes_to_insert if node.id not in existing_kg_node_ids
    #     ]

    #     # filter out duplicate llama nodes
    #     existing_nodes = self.property_graph_store.get_llama_nodes(
    #         [node.id_ for node in nodes]
    #     )
    #     existing_node_hashes = {node.hash for node in existing_nodes}
    #     nodes = [node for node in nodes if node.hash not in existing_node_hashes]

    #     # embed nodes (if needed)
    #     if self._embed_kg_nodes:
    #         # embed llama-index nodes
    #         node_texts = [
    #             node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
    #         ]

    #         if self._use_async:
    #             embeddings = asyncio.run(
    #                 self._embed_model.aget_text_embedding_batch(  # type: ignore
    #                     node_texts, show_progress=self._show_progress
    #                 )
    #             )
    #         else:
    #             embeddings = self._embed_model.get_text_embedding_batch(  # type: ignore
    #                 node_texts, show_progress=self._show_progress
    #             )

    #         for node, embedding in zip(nodes, embeddings):
    #             node.embedding = embedding

    #         # embed kg nodes
    #         kg_node_texts = [str(kg_node) for kg_node in kg_nodes_to_insert]

    #         if self._use_async:
    #             kg_embeddings = asyncio.run(
    #                 self._embed_model.aget_text_embedding_batch(  # type: ignore
    #                     kg_node_texts, show_progress=self._show_progress
    #                 )
    #             )
    #         else:
    #             kg_embeddings = self._embed_model.get_text_embedding_batch(  # type: ignore
    #                 kg_node_texts,
    #                 show_progress=self._show_progress,
    #             )

    #         for kg_node, embedding in zip(kg_nodes_to_insert, kg_embeddings):
    #             kg_node.embedding = embedding

    #     # if graph store doesn't support vectors, or the vector index was provided, use it
    #     if self.vector_store is not None and len(kg_nodes_to_insert) > 0:
    #         self._insert_nodes_to_vector_index(kg_nodes_to_insert)  # type: ignore

    #     if len(nodes) > 0:
    #         self.property_graph_store.upsert_llama_nodes(nodes)

    #     if len(kg_nodes_to_insert) > 0:
    #         self.property_graph_store.upsert_nodes(kg_nodes_to_insert)

    #     # important: upsert relations after nodes
    #     if len(kg_rels_to_insert) > 0:
    #         self.property_graph_store.upsert_relations(kg_rels_to_insert)

    #     # refresh schema if needed
    #     if self.property_graph_store.supports_structured_queries:
    #         self.property_graph_store.get_schema(refresh=True)

    #     return nodes

    def _build_index_from_nodes(
        self, nodes: Optional[Sequence[BaseNode]], **build_kwargs: Any
    ) -> IndexLPG:
        """Build index from nodes."""
        # nodes = self._insert_nodes(nodes or [], **build_kwargs)
        nodes = self._insert_nodes(nodes or [])

        # build vector index
        method_names = [node.metadata["name"] for node in nodes]

        method_name_docs = []
        method_name_embeddings = self._embed_model.get_text_embedding_batch(
            method_names, show_progress=self._show_progress
        )
        for node, method_name, embedding in zip(
            nodes, method_names, method_name_embeddings
        ):
            method_name_docs.append(
                Document(doc_id=node.id_, text=method_name, embedding=embedding)
            )

        self.method_name_vector_store.add(method_name_docs)

        # this isn't really used or needed
        return IndexLPG()

    def as_retriever(
        self,
        sub_retrievers: Optional[List["BasePGRetriever"]] = None,
        include_text: bool = True,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Return a retriever for the index.

        Args:
            sub_retrievers (Optional[List[BasePGRetriever]]):
                A list of sub-retrievers to use. If not provided, a default list will be used:
                `[LLMSynonymRetriever, VectorContextRetriever]` if the graph store supports vector queries.
            include_text (bool):
                Whether to include source-text in the retriever results.
            **kwargs:
                Additional kwargs to pass to the retriever.
        """
        from llama_index.core.indices.property_graph.retriever import PGRetriever
        from llama_index.core.indices.property_graph.sub_retrievers.vector import (
            VectorContextRetriever,
        )

        # from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import (
        #     LLMSynonymRetriever,
        # )
        from index_service.custom_retriever import LLMSynonymRetriever

        if sub_retrievers is None:
            sub_retrievers = [
                # LLMSynonymRetriever(
                #     graph_store=self.property_graph_store,
                #     include_text=include_text,
                #     llm=self._llm,
                #     **kwargs,
                # ),
            ]

            if self._embed_model and (
                self.property_graph_store.supports_vector_queries or self.vector_store
            ):
                sub_retrievers.append(
                    VectorContextRetriever(
                        graph_store=self.property_graph_store,
                        vector_store=self.vector_store,
                        include_text=include_text,
                        embed_model=self._embed_model,
                        **kwargs,
                    )
                )

        return PGRetriever(sub_retrievers, use_async=True, **kwargs)  # self._use_async

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        ref_doc_map = {}
        return ref_doc_map
