import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from autogen.logger.base_logger import BaseLogger
from openai.types.chat import ChatCompletion

from shared.websocket_server import send_data, send_data_sync

F = TypeVar("F", bound=Callable[..., Any])


class WebSocketLogger(BaseLogger):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)

    def start(self) -> str:
        message = f"[WebSocketLogger] Session {self.session_id} started."
        send_data_sync({"result": message})
        return self.session_id

    def stop(self) -> None:
        message = "[WebSocketLogger] Session stopped."
        send_data_sync({"result": message})

    def log_chat_completion(
        self,
        invocation_id: uuid.UUID,
        client_id: int,
        wrapper_id: int,
        source: Union[str, Any],
        request: Dict[str, Union[float, str, List[Dict[str, str]]]],
        response: Union[str, ChatCompletion],
        is_cached: int,
        cost: float,
        start_time: str,
    ) -> None:
        message = f"""
        <div style="color: blue;"><strong>{source.name} reply:</strong></div>
        <pre style="background: #f0f8ff; padding: 5px; border-radius: 4px;">{response.choices[0].message.content}</pre>
        """

        send_data_sync({"result": message})

    def log_event(self, source: str, event_name: str, **kwargs):
        # content = ""
        # if "message" in kwargs:
        #     if isinstance(kwargs["message"], str):
        #         content = kwargs["message"]
        #     elif isinstance(kwargs["message"], dict):
        #         content = kwargs.get("message", {}).get("content", "")
        #     else:
        #         content = ""

        # sender = kwargs.get("sender", "")
        # message = f"""
        # <div style="color: green;"><strong>{sender} -> {event_name}</strong></div>
        # <pre style="background: #eee; padding: 5px; border-radius: 4px;">{content}</pre>
        # """
        # send_data_sync(message)
        pass

    def log_function_use(
        self, source: Union[str, Any], function: F, args: Dict[str, Any], returns: Any
    ) -> None:
        # message = f"""
        # <div style="color: green;"><strong>{source} -> "function_use"</strong></div>
        # <pre style="background: #eee; padding: 5px; border-radius: 4px;">{function.__name__}</pre>
        # """
        # send_data_sync({"result": message})
        pass

    def log_new_agent(self, agent: Any, init_args: Dict[str, Any] = {}) -> None:
        # log_data = {
        #     "type": "new_agent",
        #     "agent_name": agent.name if hasattr(agent, "name") else "unknown",
        #     "init_args": init_args,
        # }
        # send_data_sync({"result": json.dumps(log_data)})
        pass

    def log_new_wrapper(self, wrapper: Any, init_args: Dict[str, Any] = {}) -> None:
        # log_data = {
        #     "type": "new_wrapper",
        #     "wrapper_id": id(wrapper),
        #     "init_args": init_args,
        # }
        # send_data_sync({"result": json.dumps(log_data)})
        pass

    def log_new_client(
        self, client: Any, wrapper: Any, init_args: Dict[str, Any]
    ) -> None:
        # log_data = {
        #     "type": "new_client",
        #     "client_class": type(client).__name__,
        #     "wrapper_id": id(wrapper),
        #     "init_args": init_args,
        # }
        # send_data_sync({"result": json.dumps(log_data)})
        pass

    def get_connection(self) -> None:
        return None
