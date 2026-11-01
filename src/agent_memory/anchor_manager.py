"""Versioned storage for the hard constraints that anchor an agent's behaviour.

Anchors are not chat history. They are short, durable rules that must survive
window trimming and retrieval noise. This module provides a small JSON-backed
store with a non-negotiable token ceiling, ordered promotion/demotion, atomic
writes, and a monotonically increasing version for audit records.

The on-disk format is intentionally human-editable::

    {
      "schema_version": 1,
      "version": 3,
      "anchors": [
        {
          "id": "safety",
          "content": "Never reveal secrets.",
          "priority": 900,
          "enabled": true
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ._tokens import TokenCounter, checked_count, normalize_counter

MAX_ANCHOR_TOKENS: Final[int] = 2_000
"""Absolute upper bound on the token footprint of all enabled anchors."""

MAX_ANCHOR_PRIORITY: Final[int] = 1_000
DEFAULT_ANCHOR_PRIORITY: Final[int] = 100
DEFAULT_ANCHOR_PATH: Final[str] = "anchors.json"
_SCHEMA_VERSION: Final[int] = 1


class AnchorError(Exception):
    """Base class for anchor-store failures."""


class AnchorValidationError(AnchorError, ValueError):
    """Raised when anchor data or configuration is invalid."""


class AnchorNotFoundError(AnchorError, LookupError):
    """Raised when an anchor ID does not exist."""


class AnchorAlreadyExistsError(AnchorError, ValueError):
    """Raised when adding an anchor whose ID is already in use."""


class AnchorLimitError(AnchorValidationError):
    """Raised when enabled anchors would exceed the hard token ceiling."""


class AnchorPersistenceError(AnchorError, OSError):
    """Raised when anchors cannot be read from or written to disk."""


@dataclass(frozen=True, slots=True)
class Anchor:
    """A single durable hard constraint.

    Attributes:
        anchor_id: Stable identifier used by :meth:`AnchorManager.promote`
            and :meth:`AnchorManager.demote`.
        content: The constraint text included in the rendered prompt.
        priority: Ordering weight in ``[0, 1000]``. Higher values are considered
            more important and are rendered first.
        enabled: Whether the anchor participates in token accounting and
            rendering.
    """

    anchor_id: str
    content: str
    priority: int = DEFAULT_ANCHOR_PRIORITY
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id.strip():
            raise AnchorValidationError("anchor_id must be a non-empty string")
        if len(self.anchor_id) > 128:
            raise AnchorValidationError("anchor_id must be at most 128 characters")
        if any(character.isspace() for character in self.anchor_id):
            raise AnchorValidationError("anchor_id must not contain whitespace")
        if not isinstance(self.content, str) or not self.content.strip():
            raise AnchorValidationError("anchor content must be a non-empty string")
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or not 0 <= self.priority <= MAX_ANCHOR_PRIORITY
        ):
            raise AnchorValidationError(
                f"priority must be an integer between 0 and {MAX_ANCHOR_PRIORITY}"
            )
        if not isinstance(self.enabled, bool):
            raise AnchorValidationError("enabled must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable representation of this anchor."""
        return {
            "id": self.anchor_id,
            "content": self.content,
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Anchor:
        """Create an anchor from a decoded JSON object.

        Args:
            data: Mapping containing ``id``, ``content``, and optional
                ``priority`` and ``enabled`` keys.

        Returns:
            A validated anchor.

        Raises:
            AnchorValidationError: If required fields are missing or invalid.
        """
        try:
            anchor_id = data["id"]
            content = data["content"]
        except KeyError as exc:
            raise AnchorValidationError(
                f"anchor is missing required field {exc.args[0]!r}"
            ) from exc
        priority = data.get("priority", DEFAULT_ANCHOR_PRIORITY)
        enabled = data.get("enabled", True)
        return cls(
            anchor_id=anchor_id,
            content=content,
            priority=priority,
            enabled=enabled,
        )


class AnchorManager:
    """Load, rank, persist, and render a set of hard constraints.

    The manager enforces :data:`MAX_ANCHOR_TOKENS` on every load and mutation.
    The ceiling is not a target to trim toward: an operation that would exceed
    it fails with :class:`AnchorLimitError` and leaves both memory and disk
    unchanged. Mutations are written atomically and bump :attr:`version`.

    Args:
        path: Path to the JSON anchor file. A missing file is treated as an
            empty store until the first mutation creates it.
        max_tokens: Local ceiling in ``[1, 2000]``. It can be lowered below the
            absolute maximum but never raised above it.
        token_counter: Optional exact tokenizer. Defaults to the deterministic
            estimator in :mod:`agent_memory._tokens`.

    Raises:
        AnchorValidationError: If ``max_tokens`` is outside the allowed range.
        AnchorPersistenceError: If an existing file cannot be read.
        AnchorLimitError: If the file contains too many enabled anchor tokens.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_ANCHOR_PATH,
        *,
        max_tokens: int = MAX_ANCHOR_TOKENS,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise AnchorValidationError("max_tokens must be an integer")
        if not 1 <= max_tokens <= MAX_ANCHOR_TOKENS:
            raise AnchorValidationError(
                f"max_tokens must be between 1 and {MAX_ANCHOR_TOKENS}"
            )

        self._path = Path(path).expanduser()
        self._max_tokens = max_tokens
        self._count = normalize_counter(token_counter)
        self._lock = threading.RLock()
        self._anchors: dict[str, Anchor] = {}
        self._version = 0
        self._reload_unlocked()

    @property
    def path(self) -> Path:
        """Path of the backing JSON file."""
        return self._path

    @property
    def max_tokens(self) -> int:
        """Configured anchor token ceiling."""
        return self._max_tokens

    @property
    def version(self) -> int:
        """Monotonic anchor-store version, suitable for audit snapshots."""
        with self._lock:
            return self._version

    @property
    def all_anchors(self) -> tuple[Anchor, ...]:
        """All anchors, sorted by ID for deterministic inspection."""
        with self._lock:
            return tuple(sorted(self._anchors.values(), key=lambda anchor: anchor.anchor_id))

    @property
    def active_anchors(self) -> tuple[Anchor, ...]:
        """Enabled anchors in render order (highest priority first)."""
        with self._lock:
            return self._sorted_active(self._anchors)

    @property
    def token_count(self) -> int:
        """Token footprint of the currently rendered anchor block."""
        with self._lock:
            return self._rendered_token_count(self._anchors)

    def __len__(self) -> int:
        with self._lock:
            return len(self._anchors)

    def __contains__(self, anchor_id: object) -> bool:
        with self._lock:
            return isinstance(anchor_id, str) and anchor_id in self._anchors

    def get_anchor(self, anchor_id: str) -> Anchor:
        """Return one anchor by ID.

        Raises:
            AnchorNotFoundError: If no anchor has the given ID.
        """
        with self._lock:
            try:
                return self._anchors[anchor_id]
            except KeyError:
                raise AnchorNotFoundError(f"unknown anchor: {anchor_id!r}") from None

    def add_anchor(
        self,
        anchor_id: str,
        content: str,
        *,
        priority: int = DEFAULT_ANCHOR_PRIORITY,
        enabled: bool = True,
    ) -> Anchor:
        """Add a new anchor and persist the store atomically.

        Args:
            anchor_id: Unique, whitespace-free identifier.
            content: Constraint text.
            priority: Initial priority in ``[0, 1000]``.
            enabled: Whether the anchor counts toward the token ceiling.

        Returns:
            The newly stored anchor.

        Raises:
            AnchorAlreadyExistsError: If ``anchor_id`` already exists.
            AnchorLimitError: If the enabled set would exceed the ceiling.
            AnchorPersistenceError: If writing the file fails.
        """
        anchor = Anchor(
            anchor_id=anchor_id,
            content=content,
            priority=priority,
            enabled=enabled,
        )
        with self._lock:
            if anchor.anchor_id in self._anchors:
                raise AnchorAlreadyExistsError(f"anchor already exists: {anchor.anchor_id!r}")
            candidate = dict(self._anchors)
            candidate[anchor.anchor_id] = anchor
            self._commit(candidate)
            return anchor

    def remove_anchor(self, anchor_id: str) -> Anchor:
        """Remove an anchor and persist the store.

        Returns:
            The removed anchor.

        Raises:
            AnchorNotFoundError: If the anchor does not exist.
        """
        with self._lock:
            try:
                removed = self._anchors[anchor_id]
            except KeyError:
                raise AnchorNotFoundError(f"unknown anchor: {anchor_id!r}") from None
            candidate = dict(self._anchors)
            del candidate[anchor_id]
            self._commit(candidate)
            return removed

    def promote(self, anchor_id: str, *, amount: int = 100) -> Anchor:
        """Increase an anchor's priority and enable it.

        Promotion is clamped at :data:`MAX_ANCHOR_PRIORITY`. If enabling the
        anchor would exceed the token ceiling, the operation fails without
        changing the store.

        Args:
            anchor_id: Anchor to promote.
            amount: Positive priority increment.

        Returns:
            The updated anchor.

        Raises:
            AnchorNotFoundError: If the anchor does not exist.
            AnchorValidationError: If ``amount`` is not a positive integer.
            AnchorLimitError: If enabling the anchor exceeds the ceiling.
        """
        self._validate_amount(amount)
        with self._lock:
            try:
                current = self._anchors[anchor_id]
            except KeyError:
                raise AnchorNotFoundError(f"unknown anchor: {anchor_id!r}") from None
            updated = Anchor(
                anchor_id=current.anchor_id,
                content=current.content,
                priority=min(MAX_ANCHOR_PRIORITY, current.priority + amount),
                enabled=True,
            )
            if updated == current:
                return current
            candidate = dict(self._anchors)
            candidate[anchor_id] = updated
            self._commit(candidate)
            return updated

    def demote(self, anchor_id: str, *, amount: int = 100) -> Anchor:
        """Decrease an anchor's priority without deleting it.

        Demotion is clamped at zero. A demoted anchor remains enabled; lower
        priority only changes render order and makes it the first candidate for
        future capacity decisions. Use :meth:`set_enabled` to exclude it from
        token accounting entirely.

        Args:
            anchor_id: Anchor to demote.
            amount: Positive priority decrement.

        Returns:
            The updated anchor.

        Raises:
            AnchorNotFoundError: If the anchor does not exist.
            AnchorValidationError: If ``amount`` is not a positive integer.
        """
        self._validate_amount(amount)
        with self._lock:
            try:
                current = self._anchors[anchor_id]
            except KeyError:
                raise AnchorNotFoundError(f"unknown anchor: {anchor_id!r}") from None
            updated = Anchor(
                anchor_id=current.anchor_id,
                content=current.content,
                priority=max(0, current.priority - amount),
                enabled=current.enabled,
            )
            if updated == current:
                return current
            candidate = dict(self._anchors)
            candidate[anchor_id] = updated
            self._commit(candidate)
            return updated

    def set_enabled(self, anchor_id: str, *, enabled: bool) -> Anchor:
        """Enable or disable an anchor.

        Enabling is rejected if it would push the enabled set over the token
        ceiling; disabling always succeeds.

        Raises:
            AnchorNotFoundError: If the anchor does not exist.
            AnchorValidationError: If ``enabled`` is not a boolean.
            AnchorLimitError: If enabling would exceed the ceiling.
        """
        if not isinstance(enabled, bool):
            raise AnchorValidationError("enabled must be a boolean")
        with self._lock:
            try:
                current = self._anchors[anchor_id]
            except KeyError:
                raise AnchorNotFoundError(f"unknown anchor: {anchor_id!r}") from None
            if current.enabled is enabled:
                return current
            updated = Anchor(
                anchor_id=current.anchor_id,
                content=current.content,
                priority=current.priority,
                enabled=enabled,
            )
            candidate = dict(self._anchors)
            candidate[anchor_id] = updated
            self._commit(candidate)
            return updated

    def render(self, *, separator: str = "\n") -> str:
        """Render enabled anchors in priority order.

        Args:
            separator: Text inserted between anchors.

        Returns:
            The anchor block, or an empty string when no anchors are enabled.
        """
        with self._lock:
            return separator.join(anchor.content for anchor in self._sorted_active(self._anchors))

    def save(self) -> None:
        """Persist the current state without changing its version.

        Raises:
            AnchorPersistenceError: If the file cannot be written.
        """
        with self._lock:
            self._write(self._anchors, self._version)

    def reload(self) -> None:
        """Discard in-memory state and reload the backing file.

        Raises:
            AnchorPersistenceError: If the file cannot be read.
            AnchorLimitError: If the file exceeds the token ceiling.
        """
        with self._lock:
            self._reload_unlocked()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the store."""
        with self._lock:
            return self._payload(self._anchors, self._version)

    def _reload_unlocked(self) -> None:
        if not self._path.exists():
            self._anchors = {}
            self._version = 0
            return
        if not self._path.is_file():
            raise AnchorPersistenceError(f"anchor path is not a file: {self._path}")

        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnchorPersistenceError(f"cannot read anchor file: {self._path}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnchorValidationError(f"invalid anchor JSON in {self._path}") from exc

        if not isinstance(payload, dict):
            raise AnchorValidationError("anchor file root must be a JSON object")
        if payload.get("schema_version", _SCHEMA_VERSION) != _SCHEMA_VERSION:
            raise AnchorValidationError(
                f"unsupported anchor schema version: {payload.get('schema_version')!r}"
            )

        version = payload.get("version", 0)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise AnchorValidationError("anchor file version must be a non-negative integer")

        anchors_payload = payload.get("anchors")
        if not isinstance(anchors_payload, list):
            raise AnchorValidationError("anchor file must contain an 'anchors' array")

        anchors: dict[str, Anchor] = {}
        for index, item in enumerate(anchors_payload):
            if not isinstance(item, dict):
                raise AnchorValidationError(f"anchor at index {index} must be an object")
            anchor = Anchor.from_dict(item)
            if anchor.anchor_id in anchors:
                raise AnchorValidationError(f"duplicate anchor ID: {anchor.anchor_id!r}")
            anchors[anchor.anchor_id] = anchor

        self._enforce_limit(anchors)
        self._anchors = anchors
        self._version = version

    def _commit(self, candidate: Mapping[str, Anchor]) -> None:
        """Validate, atomically persist, then publish a candidate state."""
        self._enforce_limit(candidate)
        next_version = self._version + 1
        self._write(candidate, next_version)
        self._anchors = dict(candidate)
        self._version = next_version

    def _enforce_limit(self, anchors: Mapping[str, Anchor]) -> None:
        token_count = self._rendered_token_count(anchors)
        if token_count > self._max_tokens:
            raise AnchorLimitError(
                f"enabled anchors use {token_count} tokens, exceeding the limit of "
                f"{self._max_tokens}"
            )

    def _rendered_token_count(self, anchors: Mapping[str, Anchor]) -> int:
        ordered = self._sorted_active(anchors)
        if not ordered:
            return 0
        return checked_count("\n".join(anchor.content for anchor in ordered), self._count)

    @staticmethod
    def _sorted_active(anchors: Mapping[str, Anchor]) -> tuple[Anchor, ...]:
        active = (anchor for anchor in anchors.values() if anchor.enabled)
        return tuple(sorted(active, key=lambda anchor: (-anchor.priority, anchor.anchor_id)))

    def _write(self, anchors: Mapping[str, Anchor], version: int) -> None:
        payload = self._payload(anchors, version)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        parent = self._path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary_name).replace(self._path)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise AnchorPersistenceError(f"cannot write anchor file: {self._path}") from exc

    @staticmethod
    def _payload(anchors: Mapping[str, Anchor], version: int) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "version": version,
            "anchors": [
                anchor.to_dict()
                for anchor in sorted(anchors.values(), key=lambda item: item.anchor_id)
            ],
        }

    @staticmethod
    def _validate_amount(amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise AnchorValidationError("amount must be a positive integer")


__all__ = [
    "DEFAULT_ANCHOR_PATH",
    "DEFAULT_ANCHOR_PRIORITY",
    "MAX_ANCHOR_PRIORITY",
    "MAX_ANCHOR_TOKENS",
    "Anchor",
    "AnchorAlreadyExistsError",
    "AnchorError",
    "AnchorLimitError",
    "AnchorManager",
    "AnchorNotFoundError",
    "AnchorPersistenceError",
    "AnchorValidationError",
]
