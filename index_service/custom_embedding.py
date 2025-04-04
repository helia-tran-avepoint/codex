import asyncio
import gettext
import json
import os
from typing import Any, Dict, List, Optional

import llama_index.embeddings.ollama
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from ollama import AsyncClient, Client

from index_service.custom_cache import EmbeddingCache
from index_service.utils import get_text_hash

embedding_cache_persist_path = "./transform_cache/embedding"


class OllamaEmbedding(llama_index.embeddings.ollama.OllamaEmbedding):
    cache: EmbeddingCache = Field(description="The Embedding Cache.")

    def __init__(
        self,
        model_name: str,
        cache: EmbeddingCache,
        base_url: str = "http://localhost:11434",
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        ollama_additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            cache=cache,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            ollama_additional_kwargs=ollama_additional_kwargs or {},
            callback_manager=callback_manager,
            **kwargs,
        )

    def get_general_text_embedding(self, texts: str) -> List[float]:
        """Get Ollama embedding."""
        hash = get_text_hash(texts)
        cached_embedding = self.cache.get(hash, collection=None)
        if cached_embedding is not None:
            embedding = cached_embedding
        else:
            result = self._client.embeddings(
                model=self.model_name,
                prompt=texts,
                options=self.ollama_additional_kwargs,
            )
            embedding = result["embedding"]
            self.cache.put(hash, embedding, collection=None)

        return embedding

    async def aget_general_text_embedding(self, prompt: str) -> List[float]:
        """Asynchronously get Ollama embedding."""
        hash = get_text_hash(prompt)
        cached_embedding = self.cache.get(hash, collection=None)
        if cached_embedding is not None:
            embedding = cached_embedding
        else:
            result = await self._async_client.embeddings(
                model=self.model_name,
                prompt=prompt,
                options=self.ollama_additional_kwargs,
            )
            embedding = result["embedding"]
            self.cache.put(hash, embedding, collection=None)

        return embedding
