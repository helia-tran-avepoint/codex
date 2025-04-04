import asyncio
import json
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from llama_index.core import node_parser
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph import PropertyGraphIndex, transformations
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import (
    BaseNode,
    Document,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from llama_index.core.utils import get_tqdm_iterable

from shared.concurrent_utils import run_jobs_threadpool

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""


class SentenceSplitter(node_parser.SentenceSplitter):
    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        refined_docs = []

        for doc in documents:
            refined_doc = deepcopy(doc)
            for key in ["callees", "parameters", "variables", "fileLocation"]:
                refined_doc.metadata.pop(key, None)
            refined_docs.append(refined_doc)

        ## TO-DO: Should use summary for source code
        return super().get_nodes_from_documents(refined_docs, show_progress, **kwargs)

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        try:
            return super().split_text_metadata_aware(text, metadata_str)
        except Exception as e:
            pass
        return [""]


class SimpleRoslynPathExtractor(TransformComponent):
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        super().__init__(
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SimpleRoslynPathExtractor"

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress=show_progress, desc="Extracting paths from text"
        )
        return [self._extract(node) for node in nodes_with_progress]

    def _extract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert "file_path" in node.metadata

        file_path = node.metadata["file_path"]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                triples = self._extract_triples_from_json(data)
        except ValueError:
            triples = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for subj, rel, obj in triples:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            reverse_rel_node = Relation(
                label=f"{rel} (reverse)",
                source_id=obj_node.id,
                target_id=subj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.extend([rel_node, reverse_rel_node])

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node

    def _extract_triples_from_json(
        self, data: Dict[str, Any]
    ) -> List[Tuple[str, str, str]]:
        """Extract triples from JSON data."""
        triples = []
        method_name = data.get("name")
        for callee in data.get("callees", []):
            callee_name = callee.get("name")
            triples.append((method_name, "calls", callee_name))
            triples.append((callee_name, "caller is", method_name))
        return triples


class SimpleLLMPathExtractor(transformations.SimpleLLMPathExtractor):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        return super().__init__(
            llm=llm,
            extract_prompt=extract_prompt,
            parse_fn=parse_fn,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
        )

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Extract triples from nodes."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress=show_progress, desc="Extracting paths from text"
        )
        return [self._extract(node) for node in nodes_with_progress]

    def _extract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            llm_response = self.llm.predict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            triples = self.parse_fn(llm_response)
        except ValueError:
            triples = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        node.metadata.pop("data_type")
        metadata = node.metadata.copy()
        for subj, rel, obj in triples:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations

        return node


class ImplicitPathExtractor(transformations.ImplicitPathExtractor): ...


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""

        # return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

        # return asyncio.run_coroutine_threadsafe(
        #     self.acall(nodes, show_progress=show_progress, **kwargs),
        #     asyncio.get_running_loop(),
        # ).result()

        # nodes_with_progress = get_tqdm_iterable(
        #     nodes, show_progress=show_progress, desc="Extracting Graph from text"
        # )
        # return [self._extract(node) for node in nodes_with_progress]

        return run_jobs_threadpool(
            [lambda node=node: self._extract(node) for node in nodes],
            show_progress=show_progress,
            desc="Extracting Graph from text",
        )

    def _extract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            llm_response = self.llm.predict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
