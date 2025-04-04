import asyncio
import json

import websockets

from shared import app_config

clients = set()
server = None


async def register(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)


async def send_data(data):
    if clients:
        if isinstance(data, dict) and "result" in data:
            message = f"<div>{data['result']}<div>"
        else:
            message = data
        await asyncio.wait(
            [asyncio.create_task(client.send(message)) for client in clients]
        )


def send_data_sync(data):
    loop = asyncio.get_event_loop()
    loop.call_soon_threadsafe(asyncio.create_task, send_data(data))


async def websocket_handler(websocket):
    await register(websocket)


async def start_websocket_server():
    global server
    server = await websockets.serve(
        websocket_handler, "0.0.0.0", app_config.web_socket_port
    )
    return server


async def stop_websocket_server():
    global server
    if server:
        server.close()
        await server.wait_closed()
