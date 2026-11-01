"""Append-only SQLite audit log for rendered request context.

Every prompt assembly should produce one snapshot containing exactly what the
agent could have seen: when the request happened, which retrieved chunks were
injected, a hash of the conversation window, and the anchor-store version. The
schema is intentionally small so audit rows stay cheap and easy to export.

Snapshots are append-only. SQLite triggers reject ``UPDATE`` and ``DELETE`` at
the database layer, so application bugs cannot rewrite history through the
public API or raw SQL on this connection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

DEFAULT_AUDIT_PATH: Final[str] = "audit.db"
_MAX_HASH_LENGTH: Final[int] = 256
_MAX_CHUNK_ID_LENGTH: Final[int] = 256

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS request_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    retrieved_chunk_ids TEXT NOT NULL,
    window_hash TEXT NOT NULL,
    anchor_version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_request_snapshots_timestamp
    ON request_snapshots(timestamp);

CREATE TRIGGER IF NOT EXISTS request_snapshots_no_update
BEFORE UPDATE ON request_snapshots
BEGIN
    SELECT RAISE(ABORT, 'audit snapshots are immutable');
END;

CREATE TRIGGER IF NOT EXISTS request_snapshots_no_delete
BEFORE DELETE ON request_snapshots
BEGIN
    SELECT RAISE(ABORT, 'audit snapshots are immutable');
END;
"""


class AuditError(Exception):
    """Base class for audit-log failures."""


class AuditValidationError(AuditError, ValueError):
    """Raised when a snapshot field is invalid."""


class AuditDataError(AuditError, RuntimeError):
    """Raised when stored audit data cannot be read or decoded."""


class AuditClosedError(AuditError, RuntimeError):
    """Raised when an operation is attempted after :meth:`AuditLogger.close`."""


@dataclass(frozen=True, slots=True)
class AuditSnapshot:
    """One immutable request-context audit record.

    Attributes:
        snapshot_id: SQLite-assigned identifier.
        timestamp: Timezone-aware UTC request time.
        retrieved_chunk_ids: Retrieval IDs in the order they were considered.
        window_hash: Stable hash of the rendered conversation window.
        anchor_version: Anchor-store version active for the request.
    """

    snapshot_id: int
    timestamp: datetime
    retrieved_chunk_ids: tuple[str, ...]
    window_hash: str
    anchor_version: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this snapshot."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "window_hash": self.window_hash,
            "anchor_version": self.anchor_version,
        }


class AuditLogger:
    """Write and inspect append-only request snapshots in SQLite.

    Args:
        db_path: SQLite database path. Use ``":memory:"`` for an ephemeral log.
            Parent directories are created for file-backed databases.

    Raises:
        AuditDataError: If the database cannot be opened or initialized.
    """

    def __init__(self, db_path: str | Path = DEFAULT_AUDIT_PATH) -> None:
        if not isinstance(db_path, (str, Path)):
            raise AuditValidationError("db_path must be a string or Path")
        raw_path = str(db_path).strip()
        if not raw_path:
            raise AuditValidationError("db_path must be a non-empty path")
        if raw_path == ":memory:":
            self._database = ":memory:"
        else:
            path = Path(raw_path).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AuditDataError(f"cannot create audit directory for {path}") from exc
            self._database = str(path)

        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._connection = self._open()

    @property
    def database(self) -> str:
        """SQLite filename, or ``:memory:``."""
        return self._database

    def __len__(self) -> int:
        return self.count_snapshots()

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def record_snapshot(
        self,
        *,
        retrieved_chunk_ids: Sequence[str],
        window_hash: str,
        anchor_version: int,
        timestamp: datetime | None = None,
    ) -> AuditSnapshot:
        """Append one request-context snapshot.

        Args:
            retrieved_chunk_ids: Ordered retrieval IDs. An empty sequence means
                no memory was injected.
            window_hash: Non-empty stable hash of the conversation window.
            anchor_version: Non-negative anchor-store version.
            timestamp: Optional timezone-aware time. Naive values are treated
                as UTC; when omitted, the current UTC time is used.

        Returns:
            The stored, immutable snapshot.

        Raises:
            AuditValidationError: If a field is invalid.
            AuditClosedError: If the logger has been closed.
            AuditDataError: If SQLite rejects the write.
        """
        chunk_ids = self._validate_chunk_ids(retrieved_chunk_ids)
        normalized_hash = self._validate_hash(window_hash)
        normalized_version = self._validate_anchor_version(anchor_version)
        normalized_timestamp = self._normalize_timestamp(timestamp)
        encoded_ids = json.dumps(list(chunk_ids), ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            connection = self._require_connection()
            try:
                with connection:
                    cursor = connection.execute(
                        """
                        INSERT INTO request_snapshots
                            (timestamp, retrieved_chunk_ids, window_hash, anchor_version)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            normalized_timestamp.isoformat(timespec="microseconds"),
                            encoded_ids,
                            normalized_hash,
                            normalized_version,
                        ),
                    )
                snapshot_id = cursor.lastrowid
            except sqlite3.Error as exc:
                raise AuditDataError("cannot append audit snapshot") from exc

        if isinstance(snapshot_id, bool) or not isinstance(snapshot_id, int):
            raise AuditDataError("SQLite did not return a valid snapshot ID")
        return AuditSnapshot(
            snapshot_id=snapshot_id,
            timestamp=normalized_timestamp,
            retrieved_chunk_ids=chunk_ids,
            window_hash=normalized_hash,
            anchor_version=normalized_version,
        )

    def get_snapshot(self, snapshot_id: int) -> AuditSnapshot | None:
        """Return one snapshot by ID, or ``None`` when absent.

        Raises:
            AuditValidationError: If ``snapshot_id`` is not a positive integer.
            AuditClosedError: If the logger has been closed.
        """
        if (
            isinstance(snapshot_id, bool)
            or not isinstance(snapshot_id, int)
            or snapshot_id <= 0
        ):
            raise AuditValidationError("snapshot_id must be a positive integer")
        with self._lock:
            connection = self._require_connection()
            try:
                row = connection.execute(
                    "SELECT * FROM request_snapshots WHERE snapshot_id = ?",
                    (snapshot_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise AuditDataError("cannot read audit snapshot") from exc
        return None if row is None else self._row_to_snapshot(row)

    def list_snapshots(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditSnapshot]:
        """Return snapshots newest first.

        Args:
            limit: Page size in ``[1, 1000]``.
            offset: Non-negative number of rows to skip.

        Raises:
            AuditValidationError: If pagination values are invalid.
            AuditClosedError: If the logger has been closed.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000:
            raise AuditValidationError("limit must be an integer between 1 and 1000")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AuditValidationError("offset must be a non-negative integer")
        with self._lock:
            connection = self._require_connection()
            try:
                rows = connection.execute(
                    """
                    SELECT * FROM request_snapshots
                    ORDER BY snapshot_id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            except sqlite3.Error as exc:
                raise AuditDataError("cannot list audit snapshots") from exc
        return [self._row_to_snapshot(row) for row in rows]

    def count_snapshots(self) -> int:
        """Return the total number of audit rows."""
        with self._lock:
            connection = self._require_connection()
            try:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM request_snapshots"
                ).fetchone()
            except sqlite3.Error as exc:
                raise AuditDataError("cannot count audit snapshots") from exc
        if row is None:
            raise AuditDataError("SQLite did not return an audit count")
        return int(row["total"])

    def close(self) -> None:
        """Close the SQLite connection. Safe to call more than once."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self._database,
                timeout=5.0,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            if self._database != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.commit()
            return connection
        except sqlite3.Error as exc:
            raise AuditDataError(f"cannot open audit database: {self._database}") from exc

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise AuditClosedError("audit logger is closed")
        return self._connection

    @staticmethod
    def _normalize_timestamp(timestamp: datetime | None) -> datetime:
        if timestamp is None:
            return datetime.now(timezone.utc)
        if not isinstance(timestamp, datetime):
            raise AuditValidationError("timestamp must be a datetime")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _validate_chunk_ids(retrieved_chunk_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(retrieved_chunk_ids, (str, bytes)):
            raise AuditValidationError("retrieved_chunk_ids must be a sequence, not a string")
        try:
            values = tuple(retrieved_chunk_ids)
        except TypeError as exc:
            raise AuditValidationError("retrieved_chunk_ids must be an iterable") from exc
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise AuditValidationError("every retrieved chunk ID must be a non-empty string")
            item = value.strip()
            if len(item) > _MAX_CHUNK_ID_LENGTH:
                raise AuditValidationError(
                    f"retrieved chunk IDs must be at most {_MAX_CHUNK_ID_LENGTH} characters"
                )
            normalized.append(item)
        return tuple(normalized)

    @staticmethod
    def _validate_hash(window_hash: str) -> str:
        if not isinstance(window_hash, str):
            raise AuditValidationError("window_hash must be a string")
        value = window_hash.strip()
        if not value:
            raise AuditValidationError("window_hash must be non-empty")
        if len(value) > _MAX_HASH_LENGTH:
            raise AuditValidationError(
                f"window_hash must be at most {_MAX_HASH_LENGTH} characters"
            )
        return value

    @staticmethod
    def _validate_anchor_version(anchor_version: int) -> int:
        if (
            isinstance(anchor_version, bool)
            or not isinstance(anchor_version, int)
            or anchor_version < 0
        ):
            raise AuditValidationError("anchor_version must be a non-negative integer")
        return anchor_version

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> AuditSnapshot:
        try:
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
            chunk_ids_value = json.loads(str(row["retrieved_chunk_ids"]))
        except (TypeError, ValueError) as exc:
            raise AuditDataError("stored audit snapshot is corrupted") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        if not isinstance(chunk_ids_value, list) or not all(
            isinstance(item, str) for item in chunk_ids_value
        ):
            raise AuditDataError("stored audit chunk IDs are corrupted")
        return AuditSnapshot(
            snapshot_id=int(row["snapshot_id"]),
            timestamp=timestamp,
            retrieved_chunk_ids=tuple(chunk_ids_value),
            window_hash=str(row["window_hash"]),
            anchor_version=int(row["anchor_version"]),
        )


__all__ = [
    "DEFAULT_AUDIT_PATH",
    "AuditClosedError",
    "AuditDataError",
    "AuditError",
    "AuditLogger",
    "AuditSnapshot",
    "AuditValidationError",
]
