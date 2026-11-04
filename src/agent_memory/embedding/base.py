"""Provider-independent embedding interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Minimal contract used by :class:`agent_memory.rag_store.RAGStore`."""

    dimension: int
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents in input order; shape (n, dimension)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query with any provider-specific instruction."""
        ...
