"""Tests for the LLM-assisted MemoryAgent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_memory.memory_agent import (
    MemoryAgent,
    build_deepseek_client,
    filter_historical_questions,
)
from agent_memory.rag_store import MemoryHit
from agent_memory.window_manager import WindowManager


class _FakeStore:
    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []
        self.metadata: dict[str, dict[str, object]] = {}

    def add_memory(
        self,
        content: str,
        source_type: str,
        *,
        timestamp_ms: int | None = None,
    ) -> str:
        self.added.append((content, source_type))
        memory_id = f"mem-{len(self.added)}"
        self.metadata[memory_id] = {
            "source_type": source_type,
            "timestamp_ms": timestamp_ms if timestamp_ms is not None else len(self.added),
        }
        return memory_id

    def next_timestamp_ms(self) -> int:
        values = [
            int(metadata["timestamp_ms"])
            for metadata in self.metadata.values()
            if isinstance(metadata.get("timestamp_ms"), int)
        ]
        return max(values, default=0) + 1

    def get_memory(self, memory_id: str) -> MemoryHit | None:
        metadata = self.metadata.get(memory_id)
        if metadata is None:
            return None
        content, source_type = self.added[int(memory_id.split("-")[1]) - 1]
        return MemoryHit(
            memory_id=memory_id,
            content=content,
            source_type=source_type,  # type: ignore[arg-type]
            metadata=metadata,  # type: ignore[arg-type]
        )

    def search_memory(
        self,
        query: str,
        *,
        top_k: int,
        min_vector_score: float,
    ) -> list[MemoryHit]:
        assert query
        assert top_k == 10
        assert min_vector_score == 0.35
        return [
            MemoryHit(
                memory_id="c1",
                content="主人喜欢喝乌龙茶。",
                source_type="system_fact",
                metadata={"timestamp_ms": 1},
                distance=0.1,
            ),
            MemoryHit(
                memory_id="c2",
                content="主人喜欢安静的咖啡馆。",
                source_type="system_fact",
                metadata={"timestamp_ms": 2},
                distance=0.2,
            ),
            MemoryHit(
                memory_id="c3",
                content="今天天气不错。",
                source_type="user_message",
                metadata={"timestamp_ms": 3},
                distance=0.3,
            ),
            MemoryHit(
                memory_id="c4",
                content="主人喜欢在周末散步。",
                source_type="system_fact",
                metadata={"timestamp_ms": 4},
                distance=0.4,
            ),
        ]


class _FakeCompletions:
    def create(self, **kwargs: object) -> dict[str, object]:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        system = messages[0]["content"]
        if "maintain long-term memory" in system:
            content = '{"has_new_preference": true, "fact": "主人喜欢乌龙茶。"}'
        elif "rewrite a user query" in system:
            content = '{"keywords": "主人 喜欢 喝 乌龙茶"}'
        elif "relevance judge" in system:
            content = '{"keep": [2, 1]}'
        else:
            content = "最终回答：你最喜欢乌龙茶。"
        return {"choices": [{"message": {"content": content}}]}


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_memory_agent_full_cognitive_chain() -> None:
    store = _FakeStore()
    agent = MemoryAgent(
        store=store,  # type: ignore[arg-type]
        llm_client=_FakeClient(),
        window=WindowManager(max_tokens=500),
        system_prompt="You are a test agent.",
        candidate_k=10,
        top_k=3,
    )

    result = agent.chat_with_memory("我最喜欢喝什么饮料？")

    assert result.reply == "最终回答：你最喜欢乌龙茶。"
    assert result.extracted_fact == "主人喜欢乌龙茶。"
    assert result.rewritten_query == "主人 喜欢 喝 乌龙茶"
    assert [hit.memory_id for hit in result.selected] == ["c2", "c1"]
    assert len(result.candidates) == 4
    assert result.llm_calls == 4
    assert result.fact_memory_id == "mem-1"
    assert result.user_memory_id == "mem-2"
    assert result.assistant_memory_id == "mem-3"
    assert store.added == [
        ("主人喜欢乌龙茶。", "system_fact"),
        ("我最喜欢喝什么饮料？", "user_message"),
        ("最终回答：你最喜欢乌龙茶。", "assistant_message"),
    ]
    assert (
        store.metadata["mem-1"]["timestamp_ms"]
        == store.metadata["mem-2"]["timestamp_ms"]
        == store.metadata["mem-3"]["timestamp_ms"]
    )
    assert "主人喜欢喝乌龙茶" in result.prompt


def test_user_and_assistant_writes_are_unconditional() -> None:
    store = _FakeStore()
    agent = MemoryAgent(
        store=store,  # type: ignore[arg-type]
        llm_client=_FakeClient(),
        window=WindowManager(max_tokens=500),
        system_prompt="You are a test agent.",
        candidate_k=10,
        top_k=3,
        record_user_message=False,
    )

    result = agent.chat_with_memory("今天有什么安排？")

    sources = [source for _, source in store.added]
    assert "user_message" in sources
    assert "assistant_message" in sources
    assert result.user_memory_id is not None
    assert result.assistant_memory_id is not None


def test_extract_preference_deduplicates_repeated_fact() -> None:
    agent = MemoryAgent(
        store=_FakeStore(),  # type: ignore[arg-type]
        llm_client=_FakeClient(),
        window=WindowManager(max_tokens=500),
        system_prompt="You are a test agent.",
        candidate_k=10,
        top_k=3,
    )

    assert agent._extract_preference("我喜欢乌龙茶。") == "主人喜欢乌龙茶。"
    assert agent._extract_preference("再说一次我喜欢乌龙茶。") is None


def test_filter_historical_questions_keeps_facts_and_statements() -> None:
    hits = [
        MemoryHit(
            memory_id="q",
            content="我最喜欢喝什么饮料？",
            source_type="user_message",
            metadata={},
        ),
        MemoryHit(
            memory_id="fact",
            content="主人喜欢喝乌龙茶。",
            source_type="system_fact",
            metadata={},
        ),
        MemoryHit(
            memory_id="statement",
            content="我今天去了公园。",
            source_type="user_message",
            metadata={},
        ),
    ]
    filtered = filter_historical_questions(hits, "我最喜欢喝什么饮料")
    assert [hit.memory_id for hit in filtered] == ["fact", "statement"]


def test_build_deepseek_client_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    client, model = build_deepseek_client()

    assert client is not None
    assert model == "deepseek-chat"
    assert str(client.base_url).rstrip("/") == "https://api.deepseek.com/v1"


def test_memory_agent_degrades_without_client(capsys: object) -> None:
    store = _FakeStore()
    agent = MemoryAgent(
        store=store,  # type: ignore[arg-type]
        llm_client=None,
        window=WindowManager(max_tokens=500),
        system_prompt="You are a test agent.",
        fallback_responder=lambda query, hits: "本地兜底",
    )

    result = agent.chat_with_memory("我喜欢什么？")

    assert result.degraded is True
    assert result.reply == "本地兜底"
    assert result.llm_calls == 0
    assert result.user_memory_id == "mem-1"
    assert result.assistant_memory_id == "mem-2"
    assert (
        store.metadata["mem-1"]["timestamp_ms"]
        == store.metadata["mem-2"]["timestamp_ms"]
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "降级" in captured.err
