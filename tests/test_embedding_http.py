"""Offline tests for the OpenAI-compatible HTTP embedding function."""

from __future__ import annotations

import json
from typing import Any

from agent_memory.rag_store import OpenAICompatibleEmbeddingFunction


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_http_embedding_request_and_response(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [3.0, 4.0]},
                    {"index": 0, "embedding": [1.0, 2.0]},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedding = OpenAICompatibleEmbeddingFunction(
        model="text-embedding-3-small",
        api_key="sk-not-logged",
        base_url="https://api.example.test/v1/",
    )
    vectors = embedding(["first", "second"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["url"] == "https://api.example.test/v1/embeddings"
    assert captured["auth"] == "Bearer sk-not-logged"
    assert captured["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }
    assert captured["timeout"] == 30.0
    assert "api_key" not in embedding.get_config()


def test_embed_query_is_available_for_chroma_query_path(monkeypatch: Any) -> None:
    """Chroma 1.x calls ``embed_query`` for queries instead of ``__call__``."""

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        return _FakeResponse({"data": [{"index": 0, "embedding": [0.5, 0.25]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    embedding = OpenAICompatibleEmbeddingFunction(
        model="embedding-3",
        api_key="sk-never-logged",
        base_url="https://api.example.test/v1",
    )

    assert embedding.embed_query(input=["我的幸运数字是多少"]) == [[0.5, 0.25]]
    assert embedding.embed_query(["我的幸运数字是多少"]) == [[0.5, 0.25]]
