from typing import List, Optional, Sequence

import fsspec
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc
from llama_index.core.storage.kvstore import SimpleKVStore as SimpleCache
from llama_index.core.storage.kvstore.types import BaseKVStore as BaseCache

DEFAULT_CACHE_NAME = "llama_cache"


class EmbeddingCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    embedding_key: str = "embedding"

    collection: str = Field(
        default=DEFAULT_CACHE_NAME, description="Collection name of the cache."
    )
    cache: BaseCache = Field(default_factory=SimpleCache, description="Cache to use.")

    # TODO: add async get/put methods?
    def put(
        self, key: str, embedding: List[float], collection: Optional[str] = None
    ) -> None:
        """Put a value into the cache."""
        collection = collection or self.collection

        val = {self.embedding_key: embedding}
        self.cache.put(key, val, collection=collection)

    def get(self, key: str, collection: Optional[str] = None) -> Optional[List[float]]:
        """Get a value from the cache."""
        collection = collection or self.collection
        embedding_dicts = self.cache.get(key, collection=collection)

        if embedding_dicts is None:
            return None

        return embedding_dicts[self.embedding_key]

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear the cache."""
        collection = collection or self.collection
        data = self.cache.get_all(collection=collection)
        for key in data:
            self.cache.delete(key, collection=collection)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the cache to a directory, if possible."""
        if isinstance(self.cache, SimpleCache):
            self.cache.persist(persist_path, fs=fs)
        else:
            print("Warning: skipping persist, only needed for SimpleCache.")

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        collection: str = DEFAULT_CACHE_NAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "EmbeddingCache":
        """Create a EmbeddingCache from a persist directory."""
        return cls(
            collection=collection,
            cache=SimpleCache.from_persist_path(persist_path, fs=fs),
        )


class LLMCache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_key: str = "llm"

    collection: str = Field(
        default=DEFAULT_CACHE_NAME, description="Collection name of the cache."
    )
    cache: BaseCache = Field(default_factory=SimpleCache, description="Cache to use.")

    # TODO: add async get/put methods?
    def put(self, key: str, content: str, collection: Optional[str] = None) -> None:
        """Put a value into the cache."""
        collection = collection or self.collection

        val = {self.llm_key: content}
        self.cache.put(key, val, collection=collection)

    def get(self, key: str, collection: Optional[str] = None) -> Optional[str]:
        """Get a value from the cache."""
        collection = collection or self.collection
        response_dicts = self.cache.get(key, collection=collection)

        if response_dicts is None:
            return None

        return response_dicts[self.llm_key]

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear the cache."""
        collection = collection or self.collection
        data = self.cache.get_all(collection=collection)
        for key in data:
            self.cache.delete(key, collection=collection)

    def persist(
        self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None
    ) -> None:
        """Persist the cache to a directory, if possible."""
        if isinstance(self.cache, SimpleCache):
            self.cache.persist(persist_path, fs=fs)
        else:
            print("Warning: skipping persist, only needed for SimpleCache.")

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        collection: str = DEFAULT_CACHE_NAME,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "LLMCache":
        """Create a LLMCache from a persist directory."""
        return cls(
            collection=collection,
            cache=SimpleCache.from_persist_path(persist_path, fs=fs),
        )
