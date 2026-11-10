"""Tests for the DSH bridge configuration manager."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from agent_memory.config import (
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


def test_missing_rag_path_raises_instead_of_defaulting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Do not depend on the user's live bridge or shell environment.
    monkeypatch.delenv("AGENT_MEMORY_RAG_PATH", raising=False)
    with pytest.raises(ConfigError, match="rag_path is not configured"):
        load_config(TEST_TMP_ROOT / f"missing-{uuid.uuid4().hex}.json")


def test_bridge_file_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_MEMORY_RAG_PATH", raising=False)
    bridge = _bridge_path("bridge")
    custom_memory = (TEST_TMP_ROOT / "custom_memory").resolve()
    try:
        bridge.write_text(
            json.dumps(
                {
                    "enable_memory": False,
                    "rag_path": str(custom_memory),
                    "max_window_tokens": 2048,
                    "retrieval_top_k": 7,
                }
            ),
            encoding="utf-8",
        )

        config = load_config(bridge)

        assert config.enable_memory is False
        assert config.rag_path == custom_memory
        assert config.max_window_tokens == 2048
        assert config.retrieval_top_k == 7
        assert isinstance(config, MemoryConfig)
        assert config.to_dict()["rag_path"] == str(config.rag_path)
    finally:
        bridge.unlink(missing_ok=True)


def test_ui_rag_path_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AGENT_MEMORY_RAG_PATH",
        str((PROJECT_ROOT / "from_env").resolve()),
    )
    bridge = _bridge_path("ui-rag")
    try:
        bridge.write_text(
            json.dumps({"rag_path": str((PROJECT_ROOT / "from_ui").resolve())}),
            encoding="utf-8",
        )

        config = load_config(bridge)

        assert config.rag_path == (PROJECT_ROOT / "from_ui").resolve()
    finally:
        bridge.unlink(missing_ok=True)


def test_environment_is_rag_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = (PROJECT_ROOT / "from_env_fallback").resolve()
    monkeypatch.setenv("AGENT_MEMORY_RAG_PATH", str(expected))

    config = load_config(TEST_TMP_ROOT / f"missing-{uuid.uuid4().hex}.json")

    assert config.rag_path == expected


def test_current_host_config_beats_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _bridge_path("current-host")
    legacy = _bridge_path("legacy-host")
    current_db = TEST_TMP_ROOT / f"current-db-{uuid.uuid4().hex}"
    legacy_db = TEST_TMP_ROOT / f"legacy-db-{uuid.uuid4().hex}"
    try:
        current.write_text(
            json.dumps(
                {
                    "enable_memory": False,
                    "rag_path": str(current_db),
                }
            ),
            encoding="utf-8",
        )
        legacy.write_text(
            json.dumps(
                {
                    "enable_memory": True,
                    "rag_path": str(legacy_db),
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("DSH_DREAM_CONFIG", str(current))
        monkeypatch.setenv("AGENT_MEMORY_CONFIG", str(legacy))

        config = load_config()

        assert config.enable_memory is False
        assert config.rag_path == current_db.resolve()
    finally:
        current.unlink(missing_ok=True)
        legacy.unlink(missing_ok=True)


def test_embedding_bridge_values_are_trimmed() -> None:
    bridge = _bridge_path("embedding")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "rag_path": str((TEST_TMP_ROOT / "embedding-db").resolve()),
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


def test_holo_indicator_is_tolerated_and_preserved() -> None:
    bridge = _bridge_path("holo-indicator")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "rag_path": str((TEST_TMP_ROOT / "holo-db").resolve()),
                    "holo_indicator": {
                        "enabled": False,
                        "style": "theme",
                        "unknown": "ignored",
                    },
                }
            ),
            encoding="utf-8",
        )
        config = load_config(bridge)
        assert config.holo_indicator.enabled is False
        assert config.to_dict()["holo_indicator"] == {"enabled": False}
    finally:
        bridge.unlink(missing_ok=True)


def test_legacy_config_defaults_holo_indicator() -> None:
    bridge = _bridge_path("holo-indicator-legacy")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "rag_path": str((TEST_TMP_ROOT / "holo-legacy-db").resolve()),
                    "holographic_mode": True,
                }
            ),
            encoding="utf-8",
        )
        config = load_config(bridge)
        assert config.holographic_mode is True
        assert config.holo_indicator.enabled is True
        assert config.to_dict()["holo_indicator"] == {"enabled": True}
    finally:
        bridge.unlink(missing_ok=True)


def test_invalid_holo_indicator_falls_back_to_defaults() -> None:
    bridge = _bridge_path("holo-indicator-invalid")
    try:
        bridge.write_text(
            json.dumps(
                {
                    "rag_path": str((TEST_TMP_ROOT / "holo-invalid-db").resolve()),
                    "holo_indicator": {"enabled": "not-a-bool", "style": "neon"},
                }
            ),
            encoding="utf-8",
        )
        config = load_config(bridge)
        assert config.holo_indicator.enabled is True
        assert config.to_dict()["holo_indicator"] == {"enabled": True}
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
            json.dumps(
                {
                    "rag_path": str((TEST_TMP_ROOT / "clamp-db").resolve()),
                    "max_window_tokens": 1,
                    "retrieval_top_k": 999,
                }
            ),
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
