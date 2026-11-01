"""Tests for the DSH bridge configuration manager."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from agent_memory.config import (
    DEFAULT_MAX_WINDOW_TOKENS,
    DEFAULT_RETRIEVAL_TOP_K,
    ConfigError,
    MemoryConfig,
    load_config,
    resolve_config_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".test-tmp"


def _bridge_path(name: str) -> Path:
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}.json"


def test_defaults_come_from_manifest() -> None:
    # Do not depend on the user's live agent_memory_config.json bridge.
    config = load_config(TEST_TMP_ROOT / f"missing-{uuid.uuid4().hex}.json")
    assert config.enable_memory is True
    assert config.rag_path == (PROJECT_ROOT / ".chroma").resolve()
    assert config.max_window_tokens == DEFAULT_MAX_WINDOW_TOKENS
    assert config.retrieval_top_k == DEFAULT_RETRIEVAL_TOP_K
    assert config.embedding_provider == "local"
    assert config.embedding_model == ""
    assert config.embedding_api_key == ""
    assert config.embedding_base_url == ""
    assert config.auto_rebuild_collection is True
    assert config.holographic_mode is False


def test_bridge_file_overrides_defaults() -> None:
    bridge = _bridge_path("bridge")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "enable_memory": False,
                    "rag_path": "custom_memory",
                    "max_window_tokens": 2048,
                    "retrieval_top_k": 7,
                }
            ),
            encoding="utf-8",
        )

        config = load_config(bridge)

        assert config.enable_memory is False
        assert config.rag_path == (PROJECT_ROOT / "custom_memory").resolve()
        assert config.max_window_tokens == 2048
        assert config.retrieval_top_k == 7
        assert isinstance(config, MemoryConfig)
        assert config.to_dict()["rag_path"] == str(config.rag_path)
    finally:
        bridge.unlink(missing_ok=True)


def test_embedding_bridge_values_are_trimmed() -> None:
    bridge = _bridge_path("embedding")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "embedding_provider": " openai_compatible ",
                    "embedding_model": " text-embedding-3-small ",
                    "embedding_api_key": " sk-test ",
                    "embedding_base_url": " https://api.example.test/v1 ",
                    "auto_rebuild_collection": False,
                    "holographic_mode": True,
                }
            ),
            encoding="utf-8",
        )
        config = load_config(bridge)
        assert config.embedding_provider == "openai_compatible"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_api_key == "sk-test"
        assert config.embedding_base_url == "https://api.example.test/v1"
        assert config.auto_rebuild_collection is False
        assert config.holographic_mode is True
    finally:
        bridge.unlink(missing_ok=True)


def test_invalid_bridge_json_raises() -> None:
    bridge = _bridge_path("broken")
    try:
        bridge.write_text("{", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bridge)
    finally:
        bridge.unlink(missing_ok=True)


def test_numbers_are_clamped() -> None:
    bridge = _bridge_path("clamp")
    try:
        bridge.write_text(
            json.dumps({"max_window_tokens": 1, "retrieval_top_k": 999}),
            encoding="utf-8",
        )
        config = load_config(bridge)
        assert config.max_window_tokens == 128
        assert config.retrieval_top_k == 20
    finally:
        bridge.unlink(missing_ok=True)


def test_env_path_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    target = TEST_TMP_ROOT / f"env-config-{uuid.uuid4().hex}.json"
    monkeypatch.setenv("AGENT_MEMORY_CONFIG", str(target))
    assert resolve_config_path() == target.resolve()
