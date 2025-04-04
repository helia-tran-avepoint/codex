import asyncio
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import websockets
from autogen import runtime_logging
from autogen.io.websockets import IOStream, IOWebsockets
from autogen.logger.base_logger import BaseLogger
from autogen.logger.logger_factory import LoggerFactory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from agent_service import app_config, logger
from agent_service.graph import graph
from agent_service.websocketlogger import WebSocketLogger
from shared import constants
from shared.models import AgentQueryRequest
from shared.websocket_server import (
    send_data,
    send_data_sync,
    start_websocket_server,
    stop_websocket_server,
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_thread = None
server_running = threading.Event()


def start_ws_server():

    def keep_ws_alive(ws: IOWebsockets):
        IOStream.set_global_default(ws)
        while True:
            time.sleep(3)

    with IOWebsockets.run_server_in_thread(
        host="0.0.0.0",
        port=app_config.web_socket_port,
        on_connect=lambda ws: keep_ws_alive(ws),
    ) as ws_uri:
        while True:
            time.sleep(3)
        print("WebSocket Server Stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_thread

    runtime_logging.start(logger=WebSocketLogger())
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    server_running.set()

    try:
        yield
    finally:
        print("WebSocket Server Stopping...")
        server_running.clear()
        if ws_thread and ws_thread.is_alive():
            ws_thread.join()


@app.post("/process")
async def process(request: AgentQueryRequest):
    global ws_thread
    if not ws_thread or not ws_thread.is_alive():
        ws_thread = threading.Thread(target=start_ws_server, daemon=True)
        ws_thread.start()

        server_running.set()

    await send_data({"result": f"Processing user query...\n"})

    logger.info(f"Process user query: {request.query}")
    role_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    messages = []
    for item in request.query:
        messages.append(
            role_map[item["role"]](content=item["content"]),
        )

    # messages = [
    #     {"role": item["role"], "content": item["content"]} for item in request.query
    # ]
    logger.debug("Invoke graph")
    response = graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": 42}},
    )

    return {"result": response["messages"][-1].content}


app.router.lifespan_context = lifespan
