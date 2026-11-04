"""Smoke tests for the Phase 3 sqlite-vec backend."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from agent_memory.storage.sqlite_backend import SqliteVecBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = PROJECT_ROOT / ".test-tmp" / "sqlite-backend"


def test_sqlite_vec_round_trip() -> None:
    pytest.importorskip("sqlite_vec")
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP / f"db-{uuid.uuid4().hex}.sqlite3"

    def embed(texts: list[str]) -> list[list[float]]:
        # Deterministic three-dimensional toy embeddings.
        return [
            [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]
            for text in texts
        ]

    backend = SqliteVecBackend(
        path,
        dimension=3,
        embedding_function=embed,
    )
    try:
        backend.add(
            ids=["a", "b", "c"],
            documents=["alpha", "beta", "gamma"],
            metadatas=[
                {"source_type": "user_message", "timestamp_ms": 1},
                {"source_type": "system_fact", "timestamp_ms": 2},
                {"source_type": "tool_result", "timestamp_ms": 3},
            ],
        )
        assert backend.count() == 3

        result = backend.query(
            query_texts=["alpha"],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        assert len(result["ids"][0]) == 2
        assert "alpha" in result["documents"][0]
        assert result["metadatas"][0][0]["timestamp_ms"] in {1, 2, 3}

        rows = backend.get(ids=["b"], include=["documents", "metadatas"])
        assert rows["ids"] == ["b"]
        assert rows["documents"] == ["beta"]

        backend.delete(["b"])
        assert backend.count() == 2
        assert backend.get(ids=["b"], include=["documents"])["ids"] == []
    finally:
        backend.close()
        path.unlink(missing_ok=True)
