"""Lightweight sqlite-vec storage backend.

This is the Phase 3 prototype. It exposes the same basic I/O surface as
ChromaVectorBackend and uses a single SQLite file for both metadata and vectors.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class SqliteVecBackend:
    """Vector storage backed by sqlite-vec (float32 embeddings)."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        collection_name: str = "agent_memory",
        dimension: int = 512,
        embedding_function: Any | None = None,
    ) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise RuntimeError(
                "SqliteVecBackend requires 'sqlite-vec'. "
                "Install it with: pip install sqlite-vec"
            ) from exc

        self.collection_name = collection_name
        self.dimension = int(dimension)
        self._embedding_function = embedding_function
        path = Path(db_path).expanduser()
        if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{collection_name}.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL UNIQUE,
                document TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                embedding float[{self.dimension}]
            )
            """
        )
        self._conn.commit()

    def _vector(self, values: Sequence[float]) -> bytes:
        return np.asarray(values, dtype=np.float32).tobytes()

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._embedding_function is None:
            raise RuntimeError("SqliteVecBackend requires an embedding_function")
        return np.asarray(self._embedding_function(list(texts)), dtype=np.float32)

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        vectors = (
            np.asarray(embeddings, dtype=np.float32)
            if embeddings is not None
            else self._embed(documents)
        )
        with self._conn:
            for memory_id, document, metadata, vector in zip(
                ids, documents, metadatas, vectors
            ):
                self._conn.execute(
                    """
                    INSERT INTO memories(memory_id, document, metadata_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        document=excluded.document,
                        metadata_json=excluded.metadata_json
                    """,
                    (memory_id, document, json.dumps(metadata, ensure_ascii=False)),
                )
                rowid = self._conn.execute(
                    "SELECT rowid FROM memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()[0]
                # vec0 does not implement INSERT OR REPLACE: re-adding an
                # existing row aborts with "UNIQUE constraint failed", so an
                # upsert must delete the old vector first.
                self._conn.execute(
                    "DELETE FROM vec_memories WHERE rowid = ?", (rowid,)
                )
                self._conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                    (rowid, self._vector(vector)),
                )

    def upsert(self, *args: Any, **kwargs: Any) -> None:
        self.add(*args, **kwargs)

    def query(
        self,
        *,
        query_texts: Sequence[str] | None = None,
        query_embeddings: Sequence[Sequence[float]] | None = None,
        n_results: int,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str],
    ) -> dict[str, list[list[Any]]]:
        if query_embeddings is not None:
            vector = np.asarray(query_embeddings[0], dtype=np.float32)
        elif query_texts is not None:
            vector = self._embed(query_texts)[0]
        else:
            raise ValueError("query_texts or query_embeddings is required")
        rows = self._conn.execute(
            """
            SELECT m.memory_id, m.document, m.metadata_json, v.distance
            FROM (
                SELECT rowid, distance
                FROM vec_memories
                WHERE embedding MATCH ? AND k = ?
            ) AS v
            JOIN memories AS m ON m.rowid = v.rowid
            ORDER BY v.distance
            """,
            (self._vector(vector), int(n_results)),
        ).fetchall()
        if where is not None:
            allowed = where.get("source_type")
            if isinstance(allowed, str):
                allowed = [allowed]
            if isinstance(allowed, list):
                rows = [
                    row
                    for row in rows
                    if json.loads(row[2]).get("source_type") in allowed
                ]
        ids = [row[0] for row in rows]
        docs = [row[1] for row in rows]
        metas = [json.loads(row[2]) for row in rows]
        distances = [float(row[3]) for row in rows]
        # vec0's default metric is L2. Embeddings are unit-normalized by the
        # BGE/ONNX backends, so cosine = 1 - d^2/2. Keep both values: distance
        # preserves the backend's native ordering, similarity is thresholdable.
        similarities = [
            max(-1.0, min(1.0, 1.0 - (distance * distance) / 2.0))
            for distance in distances
        ]
        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
            "similarities": [similarities],
        }

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, str, Mapping[str, Any], float]]:
        result = self.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (memory_id, doc, meta, distance)
            for memory_id, doc, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str],
    ) -> dict[str, list[Any]]:
        wanted = {str(item) for item in (include or [])}
        want_embeddings = "embeddings" in wanted
        if want_embeddings:
            sql = (
                "SELECT m.memory_id, m.document, m.metadata_json, v.embedding "
                "FROM memories AS m JOIN vec_memories AS v ON v.rowid = m.rowid"
            )
        else:
            sql = "SELECT m.memory_id, m.document, m.metadata_json FROM memories AS m"
        params: list[Any] = []
        if ids is not None:
            placeholders = ",".join("?" for _ in ids)
            sql += f" WHERE m.memory_id IN ({placeholders})"
            params.extend(ids)
        sql += " ORDER BY m.rowid"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        rows = self._conn.execute(sql, params).fetchall()
        # vec0 stores the vector as a float32 blob; the RAG store's corpus cache
        # requires one vector per id, so an empty list here would silently
        # disable Flat search and BM25 for the whole corpus.
        embeddings = (
            [np.frombuffer(row[3], dtype=np.float32).copy() for row in rows]
            if want_embeddings
            else []
        )
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [json.loads(row[2]) for row in rows],
            "embeddings": embeddings,
        }

    def delete(self, ids: Sequence[str]) -> None:
        with self._conn:
            for memory_id in ids:
                row = self._conn.execute(
                    "SELECT rowid FROM memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    continue
                self._conn.execute(
                    "DELETE FROM vec_memories WHERE rowid = ?", (row[0],)
                )
                self._conn.execute(
                    "DELETE FROM memories WHERE rowid = ?", (row[0],)
                )

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS backend_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO backend_metadata(key, value) VALUES ('meta', ?)",
                (json.dumps(dict(metadata), ensure_ascii=False),),
            )

    def close(self) -> None:
        self._conn.close()
