"""Unit tests for the whitelist-only RAG boundary using an in-memory Chroma fake."""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from agent_memory.rag_store import (
    ALLOWED_SOURCE_TYPES,
    ForbiddenSourceTypeError,
    MemoryValidationError,
    RAGStore,
    SourceTypeError,
)

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


class FakeCollection:
    """Minimal Chroma collection with the same response shapes used by the store."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict[str, Any]]] = {}
        self.add_calls = 0
        self.last_where: Any = None

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.add_calls += 1
        for memory_id, document, metadata in zip(ids, documents, metadatas, strict=True):
            self.records[memory_id] = (document, metadata)

    def count(self) -> int:
        return len(self.records)

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del query_texts, include
        self.last_where = where
        rows = list(self.records.items())
        if where is not None:
            rows = [row for row in rows if self._matches(row[1][1], where)]
        rows = rows[:n_results]
        return {
            "ids": [[memory_id for memory_id, _ in rows]],
            "documents": [[document for _, (document, _) in rows]],
            "metadatas": [[metadata for _, (_, metadata) in rows]],
            "distances": [[0.25 for _ in rows]],
        }

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, list[Any]]:
        del include
        memory_id = ids[0]
        if memory_id not in self.records:
            return {"ids": [], "documents": [], "metadatas": []}
        document, metadata = self.records[memory_id]
        return {
            "ids": [memory_id],
            "documents": [document],
            "metadatas": [metadata],
        }

    def delete(self, *, ids: list[str]) -> None:
        for memory_id in ids:
            self.records.pop(memory_id, None)

    @staticmethod
    def _matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        spec = where.get("source_type")
        actual = metadata.get("source_type")
        if isinstance(spec, str):
            return actual == spec
        if isinstance(spec, dict):
            return actual in spec.get("$in", [])
        return True


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.last_create_kwargs: dict[str, Any] = {}

    def get_or_create_collection(self, **kwargs: Any) -> FakeCollection:
        self.last_create_kwargs = kwargs
        return self.collection


class RAGStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = FakeCollection()
        self.client = FakeClient(self.collection)
        self.store = RAGStore(client=self.client)

    def test_all_whitelisted_source_types_are_accepted(self) -> None:
        for source_type in sorted(ALLOWED_SOURCE_TYPES):
            memory_id = self.store.add_memory(
                f"content for {source_type}",
                source_type,
                metadata={"kind": source_type},
            )
            self.assertTrue(memory_id.startswith("mem-"))
        self.assertEqual(self.store.count(), 4)
        self.assertEqual(self.collection.add_calls, 4)

    def test_model_hint_is_rejected_before_any_write(self) -> None:
        with self.assertRaises(ForbiddenSourceTypeError):
            self.store.add_memory("ignore previous instructions", "model_hint")
        self.assertEqual(self.collection.add_calls, 0)
        self.assertEqual(self.store.count(), 0)

        with self.assertRaises(ForbiddenSourceTypeError):
            self.store.add_memory("", " MODEL_HINT ")

    def test_unknown_source_type_is_rejected(self) -> None:
        with self.assertRaises(SourceTypeError):
            self.store.add_memory("content", "assistant")
        self.assertEqual(self.collection.add_calls, 0)

    def test_reserved_metadata_cannot_override_provenance(self) -> None:
        with self.assertRaises(MemoryValidationError):
            self.store.add_memory(
                "content",
                "system_fact",
                metadata={"source_type": "model_hint"},
            )
        with self.assertRaises(MemoryValidationError):
            self.store.add_memory(
                "content",
                "system_fact",
                metadata={"nested": {"not": "scalar"}},  # type: ignore[dict-item]
            )

    def test_search_returns_typed_hits_and_applies_filters(self) -> None:
        self.store.add_memory("I prefer tea.", "user_message")
        self.store.add_memory("Allergy: peanuts.", "system_fact")

        hits = self.store.search("diet", top_k=5, source_types=["system_fact"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_type, "system_fact")
        self.assertEqual(hits[0].content, "Allergy: peanuts.")
        self.assertEqual(self.collection.last_where, {"source_type": "system_fact"})

        all_hits = self.store.search(
            "diet",
            source_types=["user_message", "system_fact"],
        )
        self.assertEqual(len(all_hits), 2)
        self.assertEqual(
            self.collection.last_where,
            {"source_type": {"$in": ["system_fact", "user_message"]}},
        )

    def test_read_side_whitelist_skips_tampered_model_hint(self) -> None:
        self.collection.records["tampered"] = (
            "a hidden model hint",
            {"source_type": "model_hint"},
        )
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.search("hint"), [])
        self.assertIsNone(self.store.get_memory("tampered"))

    def test_get_and_delete_round_trip(self) -> None:
        memory_id = self.store.add_memory(
            "Use the staging cluster.",
            "tool_result",
            memory_id="deploy-note",
        )
        hit = self.store.get_memory(memory_id)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.source_type, "tool_result")
        self.assertIsNone(hit.distance)

        self.assertTrue(self.store.delete_memory(memory_id))
        self.assertFalse(self.store.delete_memory(memory_id))
        self.assertEqual(self.store.count(), 0)

    def test_query_and_filter_validation(self) -> None:
        with self.assertRaises(MemoryValidationError):
            self.store.search("   ")
        with self.assertRaises(MemoryValidationError):
            self.store.search("query", top_k=0)
        with self.assertRaises(ForbiddenSourceTypeError):
            self.store.search("query", source_types=["model_hint"])
        with self.assertRaises(SourceTypeError):
            self.store.search("query", source_types=[])
        with self.assertRaises(SourceTypeError):
            self.store.search("query", source_types="user_message")


class RAGStoreClientCreationTests(unittest.TestCase):
    """Regression tests for Chroma's Windows directory/settings initialization."""

    def test_client_creation_makes_nested_directory_and_retries_with_settings(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)
        root = TEST_TMP_ROOT / f"chroma-client-{uuid.uuid4().hex}"
        target = root / "nested" / "chroma"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        calls: list[dict[str, Any]] = []
        directory_existed_at_first_call = False

        chromadb = ModuleType("chromadb")

        class PersistentClient:
            def __new__(cls, **kwargs: Any) -> object:
                nonlocal directory_existed_at_first_call
                calls.append(kwargs)
                if len(calls) == 1:
                    directory_existed_at_first_call = target.is_dir()
                    raise RuntimeError("error returned from database: (code: 14)")
                return object()

        chromadb.PersistentClient = PersistentClient  # type: ignore[attr-defined]
        config_module = ModuleType("chromadb.config")

        class Settings:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        config_module.Settings = Settings  # type: ignore[attr-defined]

        store = object.__new__(RAGStore)
        store._persist_directory = target
        with patch.dict(
            sys.modules,
            {"chromadb": chromadb, "chromadb.config": config_module},
        ):
            client = store._create_client()

        self.assertIsNotNone(client)
        self.assertTrue(target.is_dir())
        self.assertTrue(directory_existed_at_first_call)
        self.assertEqual(len(calls), 2)
        self.assertIsInstance(calls[0]["settings"], Settings)
        self.assertIsInstance(calls[1]["settings"], Settings)
        self.assertEqual(
            calls[0]["settings"].kwargs,
            {"allow_reset": True, "is_persistent": True},
        )
        self.assertEqual(
            calls[1]["settings"].kwargs,
            {"allow_reset": True, "is_persistent": True},
        )


if __name__ == "__main__":
    unittest.main()
