"""Provider-independent vector storage interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    """Small I/O contract used by :class:`agent_memory.rag_store.RAGStore`.

    Implementations must not expose provider-specific objects to callers.
    Chroma is the current default; sqlite-vec/LanceDB can implement the same
    surface in later phases.
    """

    collection_name: str

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        ...

    def query(
        self,
        *,
        query_texts: Sequence[str] | None = None,
        query_embeddings: Sequence[Sequence[float]] | None = None,
        n_results: int,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str],
    ) -> Any:
        ...

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str],
    ) -> Any:
        ...

    def delete(self, ids: Sequence[str]) -> None:
        ...

    def count(self) -> int:
        ...

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...
