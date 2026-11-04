"""Migration fallback must not break startup when Chroma is absent."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from agent_memory.storage.migrate import migrate_if_needed

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / ".test-tmp" / "migration"


def test_missing_chroma_is_graceful(monkeypatch) -> None:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    chroma = TEST_TMP / f"chroma-{uuid.uuid4().hex}"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"")
    sqlite_path = chroma / "agent_memory.sqlite3"
    monkeypatch.setitem(sys.modules, "chromadb", None)

    assert migrate_if_needed(chroma, sqlite_path) == 0
    assert not (chroma / ".agent_memory.chroma-migrated.json").exists()
