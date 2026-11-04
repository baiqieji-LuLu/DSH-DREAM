"""Independent proof that holographic dual-write survives the sqlite-vec migration.

Run with:
  pytest -q -s tests/test_holographic_storage.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from agent_memory.anchor_manager import AnchorManager
from agent_memory.plugin_entry import MemoryPlugin
from agent_memory.rag_store import RAGStore
from agent_memory.window_manager import WindowManager


class _FakeEmbedding:
    dimension = 4
    model_name = "test-holographic-embedding"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            raw = np.asarray(
                [
                    len(text) % 17 + 1.0,
                    sum(map(ord, text)) % 31 + 1.0,
                    text.count("哈") + 1.0,
                    (len(text) + 1) % 7 + 1.0,
                ],
                dtype=np.float32,
            )
            rows.append(raw / np.linalg.norm(raw))
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def _db_path(root: Path) -> Path:
    return root / "agent_memory.sqlite3"


def _read_records(root: Path) -> list[dict[str, object]]:
    connection = sqlite3.connect(str(_db_path(root)))
    try:
        rows = connection.execute(
            "SELECT memory_id, document, metadata_json FROM memories ORDER BY rowid"
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "memory_id": memory_id,
            "document": document,
            "metadata": json.loads(metadata_json),
        }
        for memory_id, document, metadata_json in rows
    ]


@pytest.fixture()
def holographic_plugin():
    root = (
        Path(__file__).resolve().parents[1]
        / ".test-tmp"
        / f"holographic-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    store = RAGStore(
        persist_directory=root / "store",
        storage_backend="sqlite-vec",
        embedding_backend=_FakeEmbedding(),
        auto_rebuild_collection=False,
        flat_max_records=0,
    )
    plugin = MemoryPlugin(
        anchor_manager=AnchorManager(root / "anchors.json"),
        window_manager=WindowManager(max_tokens=4000),
        rag_store=store,
        enable_memory=True,
    )
    try:
        yield plugin, root / "store"
    finally:
        store.close()
        shutil.rmtree(root, ignore_errors=True)


def test_holographic_dual_write_keeps_both_messages_and_timestamp(
    holographic_plugin,
) -> None:
    plugin, store_root = holographic_plugin
    user = "今天天气真不错，我想吃个苹果"
    assistant = "苹果很健康，祝你今天有个好心情"

    result = plugin.memory_add_turn(user, assistant)

    assert result["ok"] is True
    rows = _read_records(store_root)
    assert len(rows) == 2
    by_source = {
        str(row["metadata"]["source_type"]): row for row in rows
    }
    assert by_source["user_message"]["document"] == user
    assert by_source["assistant_message"]["document"] == assistant
    assert (
        by_source["user_message"]["metadata"]["timestamp_ms"]
        == by_source["assistant_message"]["metadata"]["timestamp_ms"]
        == result["timestamp_ms"]
    )
    print("\n[test 1] sqlite-vec rows:", json.dumps(rows, ensure_ascii=False))


def test_holographic_write_rejects_memory_hint_echo(holographic_plugin) -> None:
    plugin, store_root = holographic_plugin
    echo = (
        "<memory-hint>\n"
        "[System Note: optional background memories]\n"
        "1. fake injected memory\n"
        "</memory-hint>"
    )

    result = plugin.memory_add_turn(echo, "不应写入")

    assert result["ok"] is False
    rows = _read_records(store_root)
    assert all("<memory-hint>" not in str(row["document"]) for row in rows)
    assert len(rows) == 0
    print("\n[test 2] echo rejected; rows:", len(rows))


def test_holographic_write_preserves_long_chatter(holographic_plugin) -> None:
    plugin, store_root = holographic_plugin
    long_text = ("哈哈哈" * 220) + "。"  # > 500 characters
    assert len(long_text) > 500

    result = plugin.memory_add_turn(long_text, "收到")

    assert result["ok"] is True
    rows = _read_records(store_root)
    stored = [
        row for row in rows if row["metadata"]["source_type"] == "user_message"
    ]
    assert len(stored) == 1
    assert stored[0]["document"] == long_text
    assert len(str(stored[0]["document"])) == len(long_text)
    print(
        "\n[test 3] long user chat preserved:",
        len(str(stored[0]["document"])),
        "chars",
    )
