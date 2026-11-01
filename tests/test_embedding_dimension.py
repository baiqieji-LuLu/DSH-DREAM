"""Dimension guard tests for switching embedding models."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from agent_memory.rag_store import RAGStore, RAGStoreError

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


@register_embedding_function
class FixedEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        return [[1.0] * self.dimension for _ in input]

    @staticmethod
    def name() -> str:
        return "dimension-guard-test"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> Any:
        return FixedEmbedding(int(config["dimension"]))

    def get_config(self) -> dict[str, int]:
        return {"dimension": self.dimension}


def _new_path() -> Path:
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / f"dimension-{uuid.uuid4().hex}"


def test_dimension_mismatch_raises_without_auto_rebuild() -> None:
    path = _new_path()
    collection = f"dimension_{uuid.uuid4().hex}"
    first = RAGStore(
        persist_directory=path,
        collection_name=collection,
        embedding_function=FixedEmbedding(8),
    )
    try:
        first.add_memory("keep me", "system_fact")
    finally:
        first.close()

    try:
        with pytest.raises(
            RAGStoreError,
            match="向量维度从 8 变成了 4.*勾选自动重嵌入",
        ):
            RAGStore(
                persist_directory=path,
                collection_name=collection,
                embedding_function=FixedEmbedding(4),
                auto_rebuild_collection=False,
            )
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_cloud_provider_requires_explicit_base_url() -> None:
    path = _new_path()
    try:
        with pytest.raises(RAGStoreError, match="必须填写 embedding_base_url"):
            RAGStore(
                persist_directory=path,
                collection_name=f"cloud_{uuid.uuid4().hex}",
                embedding_provider="openai_compatible",
                embedding_api_key="sk-test",
                embedding_base_url="",
            )
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_dimension_mismatch_rebuilds_when_enabled(capsys: pytest.CaptureFixture[str]) -> None:
    path = _new_path()
    collection = f"dimension_{uuid.uuid4().hex}"
    first = RAGStore(
        persist_directory=path,
        collection_name=collection,
        embedding_function=FixedEmbedding(8),
    )
    try:
        first.add_memory("old memory", "system_fact")
    finally:
        first.close()

    rebuilt = RAGStore(
        persist_directory=path,
        collection_name=collection,
        embedding_function=FixedEmbedding(4),
        auto_rebuild_collection=True,
    )
    try:
        assert rebuilt.count() == 1
        raw = rebuilt._collection.get(include=["documents", "metadatas"])
        assert "old memory" in raw["documents"]
        assert raw["metadatas"][0]["source_type"] == "system_fact"
    finally:
        rebuilt.close()
        shutil.rmtree(path, ignore_errors=True)
    captured = capsys.readouterr()
    assert "[WARNING] 维度不匹配，正在无损重嵌入旧数据..." in captured.err
