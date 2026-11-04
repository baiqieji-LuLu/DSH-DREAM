"""One-way Chroma -> sqlite-vec migration with an idempotent marker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sqlite_backend import SqliteVecBackend


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
    except ImportError as exc:
        raise RuntimeError(
            "Chroma migration requires the optional 'legacy' dependencies"
        ) from exc

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
    try:
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            ids = list(batch.get("ids") or [])
            if not ids:
                break
            documents = list(batch.get("documents") or [])
            metadatas = list(batch.get("metadatas") or [])
            embeddings = list(batch.get("embeddings") or [])
            if embeddings:
                backend.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            migrated += len(ids)
            offset += len(ids)
    finally:
        backend.close()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "source": str(chroma_dir),
                "count": migrated,
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
