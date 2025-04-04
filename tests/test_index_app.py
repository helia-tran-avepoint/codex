import asyncio
from contextlib import AsyncExitStack

import pytest
from fastapi.testclient import TestClient

from index_service.app import app, lifespan

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    loop = asyncio.get_event_loop()

    async def initialize():
        stack = AsyncExitStack()
        await stack.enter_async_context(lifespan(app))
        return stack

    stack = loop.run_until_complete(initialize())

    yield

    async def finalize():
        await stack.aclose()

    loop.run_until_complete(finalize())


def test_retrieve():
    response = client.post("/retrieve", json={"query": "PFP跟Rubrics有什么区别?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


def test_retrieve_hard():
    response = client.post("/retrieve", json={"query": "给我一个Rubrics出题的案例"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


def test_retrieve_empty_query():
    response = client.post("/retrieve", json={"query": ""})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


def test_retrieve_nonexistent_query():
    response = client.post("/retrieve", json={"query": "不存在的查询"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)


def test_build_index():
    response = client.post(
        "/build_index", json={"force_rebuild": True, "tasks": ["graph"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Index built successfully"


def test_build_source_code_index():
    response = client.post(
        "/build_index",
        json={"path": "./shared_data/test/sourcecode", "data_type": "source_code"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Index built successfully"
