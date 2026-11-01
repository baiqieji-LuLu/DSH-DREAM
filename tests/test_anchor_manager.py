"""Unit tests for the anchor store and its hard token ceiling."""

from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from agent_memory.anchor_manager import (
    MAX_ANCHOR_TOKENS,
    AnchorLimitError,
    AnchorManager,
    AnchorNotFoundError,
    AnchorValidationError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".test-tmp"


class AnchorManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)
        self.path = TEST_TMP_ROOT / f"anchors-{uuid.uuid4().hex}.json"
        self.addCleanup(self.path.unlink, missing_ok=True)

    def test_missing_file_starts_empty_and_version_zero(self) -> None:
        manager = AnchorManager(self.path)
        self.assertEqual(len(manager), 0)
        self.assertEqual(manager.version, 0)
        self.assertEqual(manager.render(), "")

    def test_add_persist_and_reload(self) -> None:
        manager = AnchorManager(self.path)
        manager.add_anchor("safety", "Never reveal secrets.", priority=900)
        manager.add_anchor("style", "Prefer concise answers.", priority=100)

        self.assertEqual(manager.version, 2)
        self.assertEqual(
            [anchor.anchor_id for anchor in manager.active_anchors],
            ["safety", "style"],
        )

        reloaded = AnchorManager(self.path)
        self.assertEqual(reloaded.version, 2)
        self.assertEqual(len(reloaded), 2)
        self.assertIn("Never reveal secrets.", reloaded.render())
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["anchors"]), 2)

    def test_absolute_token_ceiling_cannot_be_raised(self) -> None:
        with self.assertRaises(AnchorValidationError):
            AnchorManager(self.path, max_tokens=MAX_ANCHOR_TOKENS + 1)

    def test_over_limit_add_fails_without_mutation(self) -> None:
        manager = AnchorManager(self.path)
        oversized = " ".join(["token"] * (MAX_ANCHOR_TOKENS + 1))
        with self.assertRaises(AnchorLimitError):
            manager.add_anchor("too-big", oversized)
        self.assertEqual(len(manager), 0)
        self.assertEqual(manager.version, 0)
        self.assertFalse(self.path.exists())

    def test_disabled_anchors_do_not_count_until_enabled(self) -> None:
        manager = AnchorManager(self.path)
        oversized = " ".join(["token"] * (MAX_ANCHOR_TOKENS + 1))
        manager.add_anchor("parked", oversized, enabled=False)
        self.assertEqual(manager.token_count, 0)
        with self.assertRaises(AnchorLimitError):
            manager.set_enabled("parked", enabled=True)
        self.assertFalse(manager.get_anchor("parked").enabled)

    def test_promote_and_demote_change_order_and_version(self) -> None:
        manager = AnchorManager(self.path)
        manager.add_anchor("alpha", "Alpha rule.", priority=100)
        manager.add_anchor("beta", "Beta rule.", priority=100)
        self.assertEqual(
            [anchor.anchor_id for anchor in manager.active_anchors],
            ["alpha", "beta"],
        )

        manager.promote("beta", amount=500)
        self.assertEqual(
            [anchor.anchor_id for anchor in manager.active_anchors],
            ["beta", "alpha"],
        )
        self.assertEqual(manager.version, 3)

        manager.demote("beta", amount=1_000)
        self.assertEqual(manager.get_anchor("beta").priority, 0)
        self.assertEqual(manager.get_anchor("beta").enabled, True)
        self.assertEqual(
            [anchor.anchor_id for anchor in manager.active_anchors],
            ["alpha", "beta"],
        )
        with self.assertRaises(AnchorValidationError):
            manager.promote("beta", amount=0)
        with self.assertRaises(AnchorNotFoundError):
            manager.demote("missing")

    def test_invalid_json_is_rejected(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(AnchorValidationError):
            AnchorManager(self.path)


if __name__ == "__main__":
    unittest.main()
