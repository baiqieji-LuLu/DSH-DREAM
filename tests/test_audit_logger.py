"""Unit tests for append-only SQLite request snapshots."""

from __future__ import annotations

import sqlite3
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_memory.audit_logger import (
    AuditClosedError,
    AuditLogger,
    AuditValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".test-tmp"


class AuditLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = AuditLogger(":memory:")
        self.addCleanup(self.logger.close)
        self.timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    def test_record_and_read_snapshot(self) -> None:
        snapshot = self.logger.record_snapshot(
            retrieved_chunk_ids=["chunk-b", "chunk-a"],
            window_hash="window-hash-1",
            anchor_version=7,
            timestamp=self.timestamp,
        )
        self.assertEqual(snapshot.snapshot_id, 1)
        self.assertEqual(snapshot.timestamp, self.timestamp)
        self.assertEqual(snapshot.retrieved_chunk_ids, ("chunk-b", "chunk-a"))
        self.assertEqual(snapshot.anchor_version, 7)

        loaded = self.logger.get_snapshot(snapshot.snapshot_id)
        self.assertEqual(loaded, snapshot)
        self.assertIsNone(self.logger.get_snapshot(999))
        self.assertEqual(self.logger.count_snapshots(), 1)

    def test_listing_is_newest_first(self) -> None:
        first = self.logger.record_snapshot(
            retrieved_chunk_ids=[],
            window_hash="hash-1",
            anchor_version=1,
        )
        second = self.logger.record_snapshot(
            retrieved_chunk_ids=["chunk"],
            window_hash="hash-2",
            anchor_version=2,
        )
        listed = self.logger.list_snapshots()
        self.assertEqual(
            [item.snapshot_id for item in listed],
            [second.snapshot_id, first.snapshot_id],
        )
        self.assertEqual(len(self.logger.list_snapshots(limit=1, offset=1)), 1)

    def test_naive_timestamp_is_treated_as_utc(self) -> None:
        snapshot = self.logger.record_snapshot(
            retrieved_chunk_ids=[],
            window_hash="hash",
            anchor_version=0,
            timestamp=datetime(2026, 1, 2, 3, 4, 5),
        )
        self.assertEqual(snapshot.timestamp.tzinfo, timezone.utc)

    def test_invalid_fields_are_rejected(self) -> None:
        with self.assertRaises(AuditValidationError):
            self.logger.record_snapshot(
                retrieved_chunk_ids="not-a-list",  # type: ignore[arg-type]
                window_hash="hash",
                anchor_version=1,
            )
        with self.assertRaises(AuditValidationError):
            self.logger.record_snapshot(
                retrieved_chunk_ids=[],
                window_hash="  ",
                anchor_version=1,
            )
        with self.assertRaises(AuditValidationError):
            self.logger.record_snapshot(
                retrieved_chunk_ids=[],
                window_hash="hash",
                anchor_version=-1,
            )
        with self.assertRaises(AuditValidationError):
            self.logger.record_snapshot(
                retrieved_chunk_ids=[],
                window_hash="hash",
                anchor_version=True,  # type: ignore[arg-type]
            )
        self.assertEqual(self.logger.count_snapshots(), 0)

    def test_snapshots_are_append_only(self) -> None:
        self.logger.record_snapshot(
            retrieved_chunk_ids=[],
            window_hash="hash",
            anchor_version=1,
        )
        connection = self.logger._connection
        assert connection is not None
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute("UPDATE request_snapshots SET anchor_version = 99")
        with self.assertRaises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM request_snapshots")

    def test_file_backed_log_survives_reopen(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)
        path = TEST_TMP_ROOT / f"audit-{uuid.uuid4().hex}.db"

        def remove_database_files() -> None:
            for suffix in ("", "-wal", "-shm"):
                path.with_name(path.name + suffix).unlink(missing_ok=True)

        self.addCleanup(remove_database_files)
        logger = AuditLogger(path)
        logger.record_snapshot(
            retrieved_chunk_ids=["chunk"],
            window_hash="hash",
            anchor_version=3,
        )
        logger.close()

        reopened = AuditLogger(path)
        try:
            self.assertEqual(reopened.count_snapshots(), 1)
            snapshot = reopened.get_snapshot(1)
            assert snapshot is not None
            self.assertEqual(snapshot.anchor_version, 3)
        finally:
            reopened.close()

    def test_closed_logger_rejects_operations(self) -> None:
        self.logger.close()
        with self.assertRaises(AuditClosedError):
            self.logger.count_snapshots()

    def test_database_is_reported(self) -> None:
        self.assertEqual(self.logger.database, ":memory:")


if __name__ == "__main__":
    unittest.main()
