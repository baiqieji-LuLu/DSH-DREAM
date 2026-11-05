"""One-way Chroma -> sqlite-vec migration with an idempotent marker."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sqlite_backend import SqliteVecBackend

#: Legacy archive rows only carried ``role``; the current schema requires
#: ``source_type`` and rejects anything else at read time.
_ROLE_SOURCE_TYPES: dict[str, str] = {
    "user": "user_message",
    "assistant": "assistant_message",
    "system": "system_fact",
    "tool": "tool_result",
}


def _normalize_legacy_metadata(metadata: Any) -> dict[str, Any] | None:
    """Map one legacy archive row onto the current memory metadata schema.

    The old Chroma archive stored ``{role, turn, message_id, chunk_index,
    source, url}``.  ``RAGStore._build_hit`` drops every hit whose metadata has
    no allowed ``source_type``, so a migration that copies metadata verbatim
    reports success while recall stays empty.  ``timestamp_ms`` is derived from
    ``turn`` so timeline mode keeps the original ordering.
    """
    raw = dict(metadata) if isinstance(metadata, Mapping) else {}
    source_type = raw.get("source_type")
    if not isinstance(source_type, str) or not source_type.strip():
        role = str(raw.get("role") or "").strip().lower()
        source_type = _ROLE_SOURCE_TYPES.get(role)
    if source_type is None:
        return None
    clean: dict[str, Any] = {
        str(key): value
        for key, value in raw.items()
        if key != "source_type" and isinstance(value, (str, int, float, bool))
    }
    clean["source_type"] = source_type
    if "timestamp_ms" not in clean:
        turn = raw.get("turn")
        if isinstance(turn, int) and not isinstance(turn, bool) and turn > 0:
            clean["timestamp_ms"] = turn * 1000
    return clean


def migration_marker(sqlite_path: Path, collection_name: str) -> Path:
    return sqlite_path.parent / f".{collection_name}.chroma-migrated.json"


def migrate_chroma_to_sqlite(
    chroma_path: str | Path,
    sqlite_path: str | Path,
    *,
    collection_name: str = "agent_memory",
    batch_size: int = 1000,
) -> int:
    """Copy a Chroma collection into sqlite-vec; return migrated row count.

    The old Chroma directory is never modified or deleted. A marker file makes
    subsequent starts skip the migration.
    """
    chroma_dir = Path(chroma_path).expanduser()
    sqlite_path = Path(sqlite_path).expanduser()
    marker = migration_marker(sqlite_path, collection_name)
    if marker.exists():
        return 0
    if not (chroma_dir / "chroma.sqlite3").exists():
        return 0

    try:
        import chromadb
    except ImportError:
        # Core install intentionally omits Chroma. Never break plugin startup:
        # keep the old database untouched and let the user opt into migration.
        print(
            "[rag_store] 检测到旧版 Chroma 数据库；如需迁移，请安装 "
            "可选 legacy 引擎：uv pip install chromadb sentence-transformers",
            file=sys.stderr,
            flush=True,
        )
        return 0

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return 0
    total = int(collection.count())
    if total == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"source": str(chroma_dir), "count": 0}),
            encoding="utf-8",
        )
        return 0

    first = collection.get(limit=1, include=["embeddings"])
    raw_embeddings = first.get("embeddings")
    if raw_embeddings is None or len(raw_embeddings) == 0:
        raise RuntimeError("Chroma migration could not read embeddings")
    dimension = len(raw_embeddings[0])
    backend = SqliteVecBackend(
        sqlite_path,
        collection_name=collection_name,
        dimension=dimension,
    )
    migrated = 0
    skipped = 0
    try:
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            batch_ids = list(batch.get("ids") or [])
            if not batch_ids:
                break
            batch_documents = list(batch.get("documents") or [])
            batch_metadatas = list(batch.get("metadatas") or [])
            # Chroma returns embeddings as a 2-D ndarray; a bare ``or []`` would
            # raise "truth value of an array is ambiguous".
            raw = batch.get("embeddings")
            batch_embeddings = list(raw) if raw is not None and len(raw) else []
            if (
                len(batch_documents) != len(batch_ids)
                or len(batch_metadatas) != len(batch_ids)
            ):
                raise RuntimeError("Chroma returned misaligned migration batches")

            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embeddings: list[Any] = []
            for index, memory_id in enumerate(batch_ids):
                metadata = _normalize_legacy_metadata(batch_metadatas[index])
                if metadata is None:
                    skipped += 1
                    continue
                ids.append(memory_id)
                documents.append(batch_documents[index])
                metadatas.append(metadata)
                if batch_embeddings:
                    embeddings.append(batch_embeddings[index])
            if ids:
                backend.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings or None,
                )
            migrated += len(ids)
            offset += len(batch_ids)
    finally:
        backend.close()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "source": str(chroma_dir),
                "count": migrated,
                "skipped": skipped,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return migrated


def migrate_if_needed(
    chroma_path: str | Path,
    sqlite_path: str | Path,
    *,
    collection_name: str = "agent_memory",
) -> int:
    marker = migration_marker(Path(sqlite_path), collection_name)
    if marker.exists():
        return 0
    return migrate_chroma_to_sqlite(
        chroma_path,
        sqlite_path,
        collection_name=collection_name,
    )
