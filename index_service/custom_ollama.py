from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Sequence,
    Type,
    Union,
    get_args,
    runtime_checkable,
)

import llama_index.llms.ollama
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.llms.ollama.base import DEFAULT_REQUEST_TIMEOUT
from ollama import AsyncClient, Client
from pydantic import Field

from index_service.custom_cache import LLMCache
from index_service.utils import get_text_hash

llm_cache_persist_path = "./transform_cache/llm"


class Ollama(llama_index.llms.ollama.Ollama):
    cache: LLMCache = Field(description="The LLM Cache.")

    def __init__(
        self,
        model: str,
        cache: LLMCache,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.75,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        prompt_key: str = "prompt",
        json_mode: bool = False,
        additional_kwargs: Dict[str, Any] = {},
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
        is_function_calling_model: bool = True,
        keep_alive: Optional[Union[float, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            cache=cache,
            base_url=base_url,
            temperature=temperature,
            context_window=context_window,
            request_timeout=request_timeout,
            prompt_key=prompt_key,
            json_mode=json_mode,
            additional_kwargs=additional_kwargs,
            is_function_calling_model=is_function_calling_model,
            keep_alive=keep_alive,
            **kwargs,
        )

    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:

        hash = get_text_hash(f"{str(prompt)}{str(prompt_args)}")
        cached_response = self.cache.get(hash, collection=None)
        if cached_response is not None:
            response = cached_response
        else:
            response = super().predict(prompt, **prompt_args)
            self.cache.put(hash, response, collection=None)

        return response
