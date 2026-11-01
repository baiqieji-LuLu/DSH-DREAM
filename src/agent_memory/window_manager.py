"""Token-budgeted sliding window over recent conversation turns.

The window manager never mutates or truncates individual message content. It
keeps a contiguous suffix of the conversation and protects the newest turn:
everything from the last ``user`` message onward is mandatory. If that
mandatory suffix cannot fit the budget, the manager fails closed instead of
silently cutting the user's latest message or the assistant's latest reply.

Only the window is disposable. The complete history remains available until
the caller explicitly clears it, which allows a later request to use a larger
budget without losing data.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, cast

from ._tokens import TokenCounter, checked_count, normalize_counter

DEFAULT_WINDOW_TOKENS: Final[int] = 4_000
DEFAULT_MESSAGE_OVERHEAD: Final[int] = 4
MessageRole: TypeAlias = Literal["system", "user", "assistant", "tool"]
_ALLOWED_ROLES: Final[frozenset[str]] = frozenset(
    {"system", "user", "assistant", "tool"}
)


class WindowError(Exception):
    """Base class for sliding-window failures."""


class WindowValidationError(WindowError, ValueError):
    """Raised when a message, role, or configuration value is invalid."""


class WindowBudgetError(WindowError, ValueError):
    """Raised when the protected latest turn cannot fit the token budget.

    Attributes:
        required_tokens: Tokens needed by the mandatory suffix.
        max_tokens: Configured window budget.
    """

    def __init__(self, required_tokens: int, max_tokens: int) -> None:
        self.required_tokens = required_tokens
        self.max_tokens = max_tokens
        super().__init__(
            "the latest user/assistant turn requires "
            f"{required_tokens} tokens, but the window budget is {max_tokens}"
        )


@dataclass(frozen=True, slots=True)
class Message:
    """One immutable conversation message.

    Attributes:
        role: One of ``system``, ``user``, ``assistant``, or ``tool``.
        content: Full message text. It is never partially truncated.
        name: Optional participant or tool name.
    """

    role: MessageRole
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _ALLOWED_ROLES:
            allowed = ", ".join(sorted(_ALLOWED_ROLES))
            raise WindowValidationError(f"role must be one of: {allowed}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise WindowValidationError("message content must be a non-empty string")
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name.strip()
        ):
            raise WindowValidationError("message name must be a non-empty string when provided")

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation of this message."""
        return {"role": self.role, "content": self.content, "name": self.name}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Message:
        """Create a message from an OpenAI-style mapping.

        Only ``role``, ``content``, and ``name`` are preserved. Provider-specific
        fields such as ``tool_calls`` should be rendered before entering the
        window or kept in a separate tool-result message.

        Args:
            data: Mapping with at least ``role`` and ``content``.

        Returns:
            A validated immutable message.

        Raises:
            WindowValidationError: If required fields are missing or invalid.
        """
        try:
            role = data["role"]
            content = data["content"]
        except KeyError as exc:
            raise WindowValidationError(
                f"message is missing required field {exc.args[0]!r}"
            ) from exc
        return cls(role=cast(MessageRole, role), content=content, name=data.get("name"))


MessageLike: TypeAlias = Message | Mapping[str, Any]


def hash_messages(messages: Sequence[Message]) -> str:
    """Return a stable SHA-256 hash for an ordered sequence of messages.

    The hash covers role, content, and optional name. It is independent of
    tokenizer choice and is suitable for audit snapshots.

    Args:
        messages: Messages in chronological order.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(
        [message.to_dict() for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WindowManager:
    """Trim conversation history to a token budget.

    The manager stores full history. :meth:`build_window` returns a contiguous
    suffix that fits ``max_tokens`` while always retaining the latest turn,
    defined as the messages from the last ``user`` message through the end. If
    no user message exists, the final message is protected instead.

    Each message costs ``message_overhead`` tokens plus the tokens in
    ``"<role>: <content>"``. This accounts for chat-format framing and makes
    custom counters receive both role and content.

    Args:
        max_tokens: Positive token budget. Defaults to 4000.
        token_counter: Optional exact tokenizer. Defaults to the deterministic
            estimator in :mod:`agent_memory._tokens`.
        message_overhead: Non-negative per-message framing overhead.

    Raises:
        WindowValidationError: If configuration values are invalid.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_WINDOW_TOKENS,
        *,
        token_counter: TokenCounter | None = None,
        message_overhead: int = DEFAULT_MESSAGE_OVERHEAD,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise WindowValidationError("max_tokens must be a positive integer")
        if (
            isinstance(message_overhead, bool)
            or not isinstance(message_overhead, int)
            or message_overhead < 0
        ):
            raise WindowValidationError("message_overhead must be a non-negative integer")

        self._max_tokens = max_tokens
        self._message_overhead = message_overhead
        self._count = normalize_counter(token_counter)
        self._messages: list[Message] = []
        self._lock = threading.RLock()

    @property
    def max_tokens(self) -> int:
        """Configured token budget."""
        return self._max_tokens

    @property
    def history(self) -> tuple[Message, ...]:
        """Complete stored history, untrimmed."""
        with self._lock:
            return tuple(self._messages)

    @property
    def window(self) -> tuple[Message, ...]:
        """Current trimmed window for the stored history."""
        return tuple(self.build_window())

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def add_message(
        self,
        role: MessageRole,
        content: str,
        *,
        name: str | None = None,
    ) -> Message:
        """Append one message to the stored history.

        Args:
            role: Message role.
            content: Full message text.
            name: Optional participant or tool name.

        Returns:
            The appended immutable message.

        Raises:
            WindowValidationError: If the message is invalid.
        """
        message = Message(role=role, content=content, name=name)
        with self._lock:
            self._messages.append(message)
        return message

    def append(self, message: MessageLike) -> Message:
        """Append a :class:`Message` or an OpenAI-style mapping.

        Returns:
            The normalized message that was stored.

        Raises:
            WindowValidationError: If the input is not a valid message.
        """
        normalized = self._coerce_message(message)
        with self._lock:
            self._messages.append(normalized)
        return normalized

    def extend(self, messages: Sequence[MessageLike]) -> None:
        """Append several messages atomically.

        Args:
            messages: Messages or mappings to append.

        Raises:
            WindowValidationError: If any message is invalid. Validation happens
                before the stored history changes, so a failed call is atomic.
        """
        normalized = self._coerce_history(messages)
        with self._lock:
            self._messages.extend(normalized)

    def clear(self) -> None:
        """Remove all stored history."""
        with self._lock:
            self._messages.clear()

    def build_window(self, messages: Sequence[MessageLike] | None = None) -> list[Message]:
        """Return a token-budgeted, order-preserving suffix.

        The latest turn is mandatory. Older messages are prepended one at a
        time while they fit; trimming stops at the first older message that
        does not fit, so the result never has gaps.

        Args:
            messages: Explicit history to trim. When omitted, the manager's
                stored history is used.

        Returns:
            The selected messages in chronological order.

        Raises:
            WindowValidationError: If an input message is invalid.
            WindowBudgetError: If the mandatory latest turn alone exceeds the
                budget.
        """
        with self._lock:
            history = (
                self._coerce_history(messages)
                if messages is not None
                else list(self._messages)
            )

        if not history:
            return []

        mandatory_start = self._latest_turn_start(history)
        mandatory = history[mandatory_start:]
        required_tokens = self._count_messages(mandatory)
        if required_tokens > self._max_tokens:
            raise WindowBudgetError(required_tokens, self._max_tokens)

        selected = list(mandatory)
        used = required_tokens
        for message in reversed(history[:mandatory_start]):
            cost = self._count_message(message)
            if used + cost > self._max_tokens:
                break
            selected.insert(0, message)
            used += cost
        return selected

    def trim(self, messages: Sequence[MessageLike] | None = None) -> list[Message]:
        """Alias for :meth:`build_window` with the same semantics."""
        return self.build_window(messages)

    def window_tokens(self, messages: Sequence[MessageLike] | None = None) -> int:
        """Return the token cost of the trimmed window."""
        return self._count_messages(self.build_window(messages))

    def window_hash(self, messages: Sequence[MessageLike] | None = None) -> str:
        """Return a stable SHA-256 hash of the trimmed window.

        The hash covers role, content, and optional name in chronological
        order. It is suitable for the ``window_hash`` field of an audit record.
        """
        return hash_messages(self.build_window(messages))

    def _count_message(self, message: Message) -> int:
        payload = f"{message.role}: {message.content}"
        if message.name is not None:
            payload = f"{message.name} {payload}"
        return self._message_overhead + checked_count(payload, self._count)

    def _count_messages(self, messages: Sequence[Message]) -> int:
        return sum(self._count_message(message) for message in messages)

    @staticmethod
    def _latest_turn_start(history: Sequence[Message]) -> int:
        for index in range(len(history) - 1, -1, -1):
            if history[index].role == "user":
                return index
        return len(history) - 1

    @classmethod
    def _coerce_message(cls, value: MessageLike) -> Message:
        if isinstance(value, Message):
            return value
        if isinstance(value, Mapping):
            return Message.from_mapping(value)
        raise WindowValidationError("each message must be a Message or a mapping")

    @classmethod
    def _coerce_history(
        cls,
        messages: Sequence[MessageLike] | str | bytes,
    ) -> list[Message]:
        if isinstance(messages, (str, bytes)):
            raise WindowValidationError("messages must be a sequence of messages, not text")
        return [cls._coerce_message(message) for message in messages]


__all__ = [
    "DEFAULT_MESSAGE_OVERHEAD",
    "DEFAULT_WINDOW_TOKENS",
    "Message",
    "MessageLike",
    "MessageRole",
    "WindowBudgetError",
    "WindowError",
    "WindowManager",
    "WindowValidationError",
    "hash_messages",
]
