"""Deterministic token estimation shared by the memory layers.

The project deliberately avoids importing a model-specific tokenizer at import
time. The default estimator is a conservative, language-aware approximation:
each CJK ideograph, Latin word, number, or punctuation mark counts as one
token. Callers that need exact model accounting can inject a ``TokenCounter``
(for example, one backed by ``tiktoken`` or a Hugging Face tokenizer).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final, TypeAlias

TokenCounter: TypeAlias = Callable[[str], int]
"""A callable that maps text to a non-negative token count."""

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"  # CJK ideographs
    r"|[A-Za-z0-9_]+"  # words and numbers
    r"|[^\sA-Za-z0-9_\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"  # punctuation
)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``.

    The estimator is conservative and deterministic. It is intended for
    enforcing budgets without requiring a tokenizer download, not for billing
    or reproducing a provider's exact count.

    Args:
        text: Text to measure.

    Returns:
        A non-negative token estimate.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sum(1 for _ in _TOKEN_PATTERN.finditer(text))


def normalize_counter(counter: TokenCounter | None) -> TokenCounter:
    """Return ``counter`` or the default estimator.

    Args:
        counter: Optional custom token counter.

    Returns:
        A callable suitable for token accounting.
    """
    return estimate_tokens if counter is None else counter


def checked_count(text: str, counter: TokenCounter) -> int:
    """Count tokens and reject invalid counter results.

    Args:
        text: Text to count.
        counter: Token-counting callable.

    Returns:
        The validated token count.

    Raises:
        TypeError: If the counter returns a non-integer.
        ValueError: If the counter returns a negative value.
    """
    value = counter(text)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("token counter must return an int")
    if value < 0:
        raise ValueError("token counter must return a non-negative int")
    return value
