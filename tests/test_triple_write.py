"""Real Chroma test for timestamp-bound user/assistant/fact triple writes."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from agent_memory.memory_agent import MemoryAgent
from agent_memory.rag_store import RAGStore

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


@register_embedding_function
class TripleHashEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        vectors: list[list[float]] = []
        for text in input:
            vector = [0.0] * self.dimension
            tokens = set(re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.lower()))
            tokens.update(a + b for a, b in pairwise(re.findall(r"[\u4e00-\u9fff]", text)))
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] & 1 else -1.0
            norm = sum(value * value for value in vector) ** 0.5
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "triple-write-test-hash-1024"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> Any:
        return TripleHashEmbedding(int(config.get("dimension", 1024)))

    def get_config(self) -> dict[str, Any]:
        return {"dimension": self.dimension}


class _MockCompletions:
    def create(self, **kwargs: Any) -> dict[str, Any]:
        system = kwargs["messages"][0]["content"]
        if "maintain long-term memory" in system:
            content = json.dumps(
                {"has_new_preference": True, "fact": "用户今天吃了火锅。"},
                ensure_ascii=False,
            )
        elif "rewrite a user query" in system:
            content = json.dumps({"keywords": "火锅 好吃 吗"}, ensure_ascii=False)
        elif "relevance judge" in system:
            content = json.dumps({"keep": [1]})
        else:
            content = "好吃吗？"
        return {"choices": [{"message": {"content": content}}]}


class _MockClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": _MockCompletions()})()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triple_write_shares_timestamp_and_retrieves_assistant() -> None:
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    db_path = TEST_TMP_ROOT / f"triple-{uuid.uuid4().hex}"
    store = RAGStore(
        persist_directory=db_path,
        collection_name=f"triple_{uuid.uuid4().hex}",
        embedding_function=TripleHashEmbedding(),
    )
    try:
        agent = MemoryAgent(
            store=store,
            llm_client=_MockClient(),
            system_prompt="You are a triple-write test agent.",
            candidate_k=10,
            top_k=3,
        )
        result = agent.chat_with_memory("我今天吃了火锅")

        assert result.reply == "好吃吗？"
        assert result.extracted_fact == "用户今天吃了火锅。"
        assert result.fact_memory_id
        assert result.user_memory_id
        assert result.assistant_memory_id

        raw = store._collection.get(include=["documents", "metadatas"])
        rows = list(zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])))
        assert len(rows) == 3
        by_source = {metadata["source_type"]: (document, metadata) for _, document, metadata in rows}
        assert set(by_source) == {"system_fact", "user_message", "assistant_message"}
        assert by_source["user_message"][0] == "我今天吃了火锅"
        assert by_source["assistant_message"][0] == "好吃吗？"
        timestamps = {int(metadata["timestamp_ms"]) for _, metadata in by_source.values()}
        assert len(timestamps) == 1

        print("\n=== TRIPLE_WRITE_ROWS ===")
        for memory_id, document, metadata in rows:
            print(
                json.dumps(
                    {
                        "id": memory_id,
                        "source_type": metadata.get("source_type"),
                        "timestamp_ms": metadata.get("timestamp_ms"),
                        "content": document,
                    },
                    ensure_ascii=False,
                )
            )

        hits = store.search_memory("好吃吗？", top_k=5, min_vector_score=0.0)
        print("=== ASSISTANT_SEARCH_HIT ===")
        for hit in hits:
            if hit.source_type == "assistant_message":
                print(
                    json.dumps(
                        {
                            "source_type": hit.source_type,
                            "content": hit.content,
                            "retrieval_source": hit.retrieval_source,
                        },
                        ensure_ascii=False,
                    )
                )
        assert any(
            hit.source_type == "assistant_message" and hit.content == "好吃吗？"
            for hit in hits
        )
        assert {hit.source_type for hit in hits} == {
            "system_fact",
            "user_message",
            "assistant_message",
        }
        assert hits[0].source_type == "system_fact"
        turn_timestamp = next(iter(timestamps))
        assert all(
            hit.metadata.get("timestamp_ms") == turn_timestamp for hit in hits
        )
    finally:
        store.close()
        shutil.rmtree(db_path, ignore_errors=True)
