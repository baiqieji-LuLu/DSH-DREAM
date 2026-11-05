"""Dual-track memory regressions: latest-wins and event timeline."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from agent_memory.anchor_manager import AnchorManager
from agent_memory.plugin_entry import MemoryPlugin
from agent_memory.rag_store import RAGStore
from agent_memory.window_manager import WindowManager


class _BagEmbedding:
    dimension = 8
    model_name = "test-bag-embedding"

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        chars = list(text)
        for index, char in enumerate(chars):
            vector[(ord(char) + index) % self.dimension] += 1.0
            if index:
                vector[(ord(chars[index - 1]) + ord(char)) % self.dimension] += 2.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._one(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


@pytest.fixture()
def plugin():
    root = (
        Path(__file__).resolve().parents[1]
        / ".test-tmp"
        / f"dual-track-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    store = RAGStore(
        persist_directory=root / "store",
        storage_backend="sqlite-vec",
        embedding_backend=_BagEmbedding(),
        auto_rebuild_collection=False,
        flat_max_records=0,
    )
    instance = MemoryPlugin(
        anchor_manager=AnchorManager(root / "anchors.json"),
        window_manager=WindowManager(max_tokens=4000),
        rag_store=store,
        enable_memory=True,
    )
    try:
        yield instance
    finally:
        store.close()
        shutil.rmtree(root, ignore_errors=True)


def _user_texts(result: dict) -> list[str]:
    return [
        row["content"]
        for row in result["results"]
        if row["source_type"] == "user_message"
    ]


def test_preference_replacement_collapses_old_value(plugin) -> None:
    plugin.memory_add_turn("我最喜欢的菜是马蹄糕", "记住了")
    plugin.memory_add_turn("把马蹄糕换成柠檬牛肉球", "好的，已经更新")

    result = plugin.memory_search("我喜欢什么菜", top_k=10)
    texts = _user_texts(result)

    assert texts[0] == "把马蹄糕换成柠檬牛肉球"
    assert "我最喜欢的菜是马蹄糕" not in texts
    assert "柠檬牛肉球" in texts[0]
    # Latest-wins is an injection-time projection, never a physical delete.
    assert plugin.rag_store.count() == 4


def test_history_intent_returns_old_and_new_in_order(plugin) -> None:
    plugin.memory_add_turn("我最喜欢的菜是马蹄糕", "记住了")
    plugin.memory_add_turn("把马蹄糕换成柠檬牛肉球", "好的，已经更新")

    result = plugin.memory_search(
        "我以前喜欢吃什么，现在换成什么了？",
        top_k=10,
    )
    user_hits = [
        row
        for row in result["results"]
        if row["source_type"] == "user_message"
    ]
    texts = [row["content"] for row in user_hits]

    assert "我最喜欢的菜是马蹄糕" in texts
    assert "把马蹄糕换成柠檬牛肉球" in texts
    assert texts.index("我最喜欢的菜是马蹄糕") < texts.index(
        "把马蹄糕换成柠檬牛肉球"
    )
    stamps = [row["metadata"]["timestamp_ms"] for row in user_hits]
    assert stamps == sorted(stamps)


def test_time_intent_finds_replacement_date(plugin) -> None:
    plugin.memory_add_turn("我最喜欢的菜是马蹄糕", "记住了")
    plugin.memory_add_turn("把马蹄糕换成柠檬牛肉球", "好的，已经更新")

    result = plugin.memory_search(
        "我什么时候跟你说要换牛肉球的？",
        top_k=10,
    )
    texts = _user_texts(result)

    assert any("把马蹄糕换成柠檬牛肉球" in text for text in texts)
    assert all("timestamp_ms" in row["metadata"] for row in result["results"])


def test_event_evolution_is_preserved_in_timeline(plugin) -> None:
    plugin.memory_add_turn("我买了个杯子", "好的")
    plugin.memory_add_turn("我把杯子摔了", "可惜")
    plugin.memory_add_turn("我又买了个新的杯子", "这次小心一点")

    result = plugin.memory_search("杯子怎么了", top_k=10, timeline_mode=True)
    user_hits = [
        row
        for row in result["results"]
        if row["source_type"] == "user_message"
    ]
    texts = [row["content"] for row in user_hits]
    stamps = [row["metadata"]["timestamp_ms"] for row in user_hits]

    assert texts == ["我买了个杯子", "我把杯子摔了", "我又买了个新的杯子"]
    assert stamps == sorted(stamps)
    assert len(stamps) == 3


def test_replacement_context_exposes_latest_rebuy(plugin) -> None:
    plugin.memory_add_turn("我买了个杯子", "好的")
    plugin.memory_add_turn("我把杯子摔了", "可惜")
    plugin.memory_add_turn("我又买了个新的杯子", "这次小心一点")

    result = plugin.memory_search(
        "我打算买个新杯子，之前那个摔了",
        top_k=5,
    )
    texts = _user_texts(result)

    assert any("我又买了个新的杯子" in text for text in texts)
