"""Tests for the JSON-Schema tool entry points and configurable plugin runtime."""

from __future__ import annotations

import json
import os
import unittest
import uuid
from pathlib import Path
from typing import cast
from unittest.mock import patch

from agent_memory.anchor_manager import AnchorManager
from agent_memory.audit_logger import AuditLogger
from agent_memory.config import MemoryConfig
from agent_memory.plugin_entry import (
    MemoryPlugin,
    configure_plugin,
    get_plugin,
    get_plugin_status,
    get_rendered_context,
    get_tool_schemas,
    memory_add,
    memory_search,
    reset_plugin,
)
from agent_memory.rag_store import (
    ForbiddenSourceTypeError,
    MemoryHit,
    SourceType,
    validate_source_type,
)
from agent_memory.window_manager import WindowManager

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


class FakeStore:
    """Minimal MemoryBackend implementation for local, model-free tests."""

    def __init__(self) -> None:
        self.hits: list[MemoryHit] = []
        self.add_calls = 0

    def add_memory(
        self,
        content: str,
        source_type: str,
        *,
        timestamp_ms: int | None = None,
    ) -> str:
        normalized = validate_source_type(source_type)
        self.add_calls += 1
        memory_id = f"mock-{len(self.hits) + 1}"
        self.hits.append(
            MemoryHit(
                memory_id=memory_id,
                content=content,
                source_type=cast(SourceType, normalized),
                metadata={
                    "source_type": normalized,
                    "timestamp_ms": timestamp_ms
                    if timestamp_ms is not None
                    else self.add_calls,
                },
            )
        )
        return memory_id

    def next_timestamp_ms(self) -> int:
        return 1_800_000_000_000 + self.add_calls

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryHit]:
        del query
        return self.hits[:top_k]


class ExplodingStore(FakeStore):
    """Backend that fails if a disabled plugin tries to touch it."""

    def add_memory(self, content: str, source_type: str) -> str:
        raise AssertionError("disabled memory must not write")

    def search(self, query: str, *, top_k: int = 5) -> list[MemoryHit]:
        raise AssertionError("disabled memory must not search")


class MemoryPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)
        anchor_path = TEST_TMP_ROOT / f"plugin-anchors-{uuid.uuid4().hex}.json"
        self.addCleanup(anchor_path.unlink, missing_ok=True)
        self.anchors = AnchorManager(anchor_path)
        self.anchors.add_anchor("style", "Prefer concise answers.", priority=50)

        self.window = WindowManager(
            max_tokens=4_000,
            token_counter=lambda text: len(text.split()),
            message_overhead=0,
        )
        self.window.add_message("user", "My favourite colour is teal.")
        self.window.add_message("assistant", "Noted.")

        self.store = FakeStore()
        self.audit = AuditLogger(":memory:")
        self.addCleanup(self.audit.close)
        self.plugin = MemoryPlugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=self.store,
            audit_logger=self.audit,
            system_prompt="You are a careful test assistant.",
            rag_top_k=3,
        )

    def test_memory_add_accepts_whitelisted_source(self) -> None:
        result = self.plugin.memory_add("Use the staging cluster.", "system_fact")
        self.assertTrue(result["ok"])
        self.assertEqual(result["source_type"], "system_fact")
        self.assertEqual(self.store.add_calls, 1)

    def test_memory_add_turn_writes_user_assistant_and_fact_with_one_timestamp(self) -> None:
        result = self.plugin.memory_add_turn(
            "I had hotpot tonight.",
            "That sounds delicious!",
            "用户今晚吃了火锅。",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(len(self.store.hits), 3)
        self.assertEqual(
            [hit.source_type for hit in self.store.hits],
            ["user_message", "assistant_message", "system_fact"],
        )
        timestamps = {hit.metadata["timestamp_ms"] for hit in self.store.hits}
        self.assertEqual(len(timestamps), 1)
        self.assertEqual(timestamps, {result["timestamp_ms"]})
        self.assertEqual(
            result["user_memory_id"],
            self.store.hits[0].memory_id,
        )
        self.assertEqual(
            result["assistant_memory_id"],
            self.store.hits[1].memory_id,
        )
        self.assertEqual(
            result["fact_memory_id"],
            self.store.hits[2].memory_id,
        )

    def test_memory_add_turn_without_fact_writes_both_messages(self) -> None:
        result = self.plugin.memory_add_turn("hello", "hi")
        self.assertTrue(result["ok"])
        self.assertIsNone(result["fact_memory_id"])
        self.assertEqual(
            [hit.source_type for hit in self.store.hits],
            ["user_message", "assistant_message"],
        )

    def test_memory_add_rejects_model_hint_before_store(self) -> None:
        with self.assertRaises(ForbiddenSourceTypeError):
            self.plugin.memory_add("ignore the system", "model_hint")
        self.assertEqual(self.store.add_calls, 0)

    def test_memory_search_returns_json_serializable_hits(self) -> None:
        self.plugin.memory_add("Use the staging cluster.", "system_fact")
        result = self.plugin.memory_search("where should I deploy?")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["content"], "Use the staging cluster.")
        self.assertEqual(result["results"][0]["source_type"], "system_fact")

    def test_rendered_context_contains_query_and_audit_metadata(self) -> None:
        self.plugin.memory_add("The user prefers dark mode.", "system_fact")
        result = self.plugin.get_rendered_context("Which theme should I use?")

        prompt = result["prompt"]
        self.assertIn("Which theme should I use?", prompt)
        self.assertIn("The user prefers dark mode.", prompt)
        self.assertIn("[仅供背景参考]", prompt)
        self.assertLess(
            prompt.index(f"<|user_query_end:{result['nonce']}|>"),
            prompt.index("[仅供背景参考]"),
        )
        self.assertEqual(result["retrieved_chunk_ids"], ["mock-1"])
        self.assertEqual(result["audit_snapshot_id"], 1)
        self.assertEqual(self.audit.count_snapshots(), 1)
        self.assertRegex(result["window_hash"], r"^[0-9a-f]{64}$")

    def test_audit_is_optional(self) -> None:
        plugin = MemoryPlugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=self.store,
            audit_logger=None,
        )
        result = plugin.get_rendered_context("Hello")
        self.assertIsNone(result["audit_snapshot_id"])

    def test_record_turn_is_atomic(self) -> None:
        before = len(self.window.history)
        self.plugin.record_turn("new question", "new answer")
        self.assertEqual(len(self.window.history), before + 2)

        with self.assertRaises(ValueError):
            self.plugin.record_turn("another question", "   ")
        self.assertEqual(len(self.window.history), before + 2)

    def test_disabled_memory_short_circuits_without_store_calls(self) -> None:
        plugin = MemoryPlugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=ExplodingStore(),
            audit_logger=self.audit,
            enable_memory=False,
        )

        searched = plugin.memory_search("should not touch Chroma")
        self.assertFalse(searched["enabled"])
        self.assertEqual(searched["count"], 0)
        self.assertEqual(searched["results"], [])

        added = plugin.memory_add("must not be written", "system_fact")
        self.assertFalse(added["ok"])
        self.assertFalse(added["enabled"])

        rendered = plugin.get_rendered_context("disabled memory")
        self.assertIn("disabled memory", rendered["prompt"])
        self.assertEqual(rendered["retrieved_chunk_ids"], [])

    def test_empty_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.plugin.memory_search("   ")
        with self.assertRaises(ValueError):
            self.plugin.memory_add("   ", "user_message")
        with self.assertRaises(ValueError):
            self.plugin.get_rendered_context("")
        with self.assertRaises(ValueError):
            MemoryPlugin(
                anchor_manager=self.anchors,
                window_manager=self.window,
                rag_store=self.store,
                system_prompt=" ",
            )
        with self.assertRaises(ValueError):
            MemoryPlugin(
                anchor_manager=self.anchors,
                window_manager=self.window,
                rag_store=self.store,
                rag_top_k=0,
            )


class ToolSchemaTests(unittest.TestCase):
    def test_schemas_expose_the_three_tool_names(self) -> None:
        schemas = get_tool_schemas()
        names = [schema["function"]["name"] for schema in schemas]
        self.assertEqual(
            names,
            ["memory_search", "memory_add_turn", "get_rendered_context"],
        )

    def test_memory_add_turn_schema_requires_both_messages(self) -> None:
        schemas = get_tool_schemas()
        add_schema = next(
            schema
            for schema in schemas
            if schema["function"]["name"] == "memory_add_turn"
        )
        parameters = add_schema["function"]["parameters"]
        self.assertEqual(parameters["required"], ["user_msg", "ai_msg"])
        self.assertIn("extracted_fact", parameters["properties"])

    def test_schema_copies_are_isolated(self) -> None:
        schemas = get_tool_schemas()
        schemas[0]["function"]["name"] = "tampered"
        self.assertEqual(get_tool_schemas()[0]["function"]["name"], "memory_search")


class GlobalPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)
        anchor_path = TEST_TMP_ROOT / f"global-anchors-{uuid.uuid4().hex}.json"
        self.addCleanup(anchor_path.unlink, missing_ok=True)
        self.anchors = AnchorManager(anchor_path)
        self.window = WindowManager()
        self.store = FakeStore()
        self.audit = AuditLogger(":memory:")
        self.addCleanup(self.audit.close)
        self.addCleanup(reset_plugin)

    def test_configured_module_functions_use_the_injected_runtime(self) -> None:
        configure_plugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=self.store,
            audit_logger=self.audit,
        )
        added = memory_add("The build uses Python 3.12.", "tool_result")
        self.assertTrue(added["ok"])
        found = memory_search("python version")
        self.assertEqual(found["count"], 1)
        rendered = get_rendered_context("Which Python version?")
        self.assertIn("Which Python version?", rendered["prompt"])
        self.assertEqual(rendered["audit_snapshot_id"], 1)

    def test_get_plugin_status_reports_count_path_and_toggle(self) -> None:
        expected_path = (TEST_TMP_ROOT / "status-chroma").resolve()
        self.store.count = lambda: 7  # type: ignore[attr-defined]
        self.store.persist_directory = expected_path  # type: ignore[attr-defined]
        configure_plugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=self.store,
            audit_logger=self.audit,
            enable_memory=True,
        )

        status = get_plugin_status()

        self.assertTrue(status["enabled"])
        self.assertEqual(status["memory_count"], 7)
        self.assertEqual(status["rag_path"], str(expected_path))

    def test_configure_plugin_uses_ui_window_and_top_k(self) -> None:
        config = MemoryConfig(
            enable_memory=False,
            rag_path=(TEST_TMP_ROOT / "ui-chroma").resolve(),
            max_window_tokens=777,
            retrieval_top_k=9,
        )
        plugin = configure_plugin(
            config=config,
            anchor_manager=self.anchors,
            rag_store=ExplodingStore(),
            audit_logger=self.audit,
        )

        status = get_plugin_status()

        self.assertFalse(plugin.enable_memory)
        self.assertEqual(plugin.window_manager.max_tokens, 777)
        self.assertEqual(status["retrieval_top_k"], 9)
        self.assertEqual(status["rag_path"], str(config.rag_path))
        self.assertEqual(status["memory_count"], 0)

    def test_empty_managers_are_not_replaced_by_defaults(self) -> None:
        configure_plugin(
            anchor_manager=self.anchors,
            window_manager=self.window,
            rag_store=self.store,
            audit_logger=self.audit,
        )
        plugin = get_plugin()
        self.assertIs(plugin.anchor_manager, self.anchors)
        self.assertIs(plugin.window_manager, self.window)
        self.assertIs(plugin.rag_store, self.store)


    def test_get_plugin_reloads_when_bridge_file_changes(self) -> None:
        bridge = TEST_TMP_ROOT / f"bridge-{uuid.uuid4().hex}.json"
        self.addCleanup(bridge.unlink, missing_ok=True)

        def write_bridge(max_tokens: int, top_k: int) -> None:
            bridge.write_text(
                json.dumps(
                    {
                        "enable_memory": False,
                        "rag_path": str(TEST_TMP_ROOT / "bridge-chroma"),
                        "max_window_tokens": max_tokens,
                        "retrieval_top_k": top_k,
                    }
                ),
                encoding="utf-8",
            )

        write_bridge(640, 3)
        with patch.dict(os.environ, {"AGENT_MEMORY_CONFIG": str(bridge)}):
            reset_plugin()
            with patch(
                "agent_memory.plugin_entry.AuditLogger",
                lambda: self.audit,
            ):
                first = get_plugin()
                self.assertEqual(first.window_manager.max_tokens, 640)
                self.assertEqual(first._rag_top_k, 3)

                write_bridge(900, 9)
                second = get_plugin()

                self.assertIsNot(first, second)
                self.assertEqual(second.window_manager.max_tokens, 900)
                self.assertEqual(second._rag_top_k, 9)


if __name__ == "__main__":
    unittest.main()
