"""Chroma implementation of the VectorBackend contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ChromaVectorBackend:
    """Thin adapter that keeps Chroma-specific calls out of RAGStore I/O."""

    def __init__(self, collection: Any, collection_name: str) -> None:
        self._collection = collection
        self.collection_name = collection_name

    @property
    def collection(self) -> Any:
        """Compatibility escape hatch used by legacy tests/tools."""
        return self._collection

    def add(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "ids": list(ids),
            "documents": list(documents),
            "metadatas": [dict(item) for item in metadatas],
        }
        if embeddings is not None:
            kwargs["embeddings"] = list(embeddings)
        self._collection.add(**kwargs)

    def query(
        self,
        *,
        query_texts: Sequence[str] | None = None,
        query_embeddings: Sequence[Sequence[float]] | None = None,
        n_results: int,
        where: Mapping[str, Any] | None = None,
        include: Sequence[str],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "n_results": n_results,
            "include": list(include),
        }
        if query_texts is not None:
            kwargs["query_texts"] = list(query_texts)
        if query_embeddings is not None:
            kwargs["query_embeddings"] = list(query_embeddings)
        if where is not None:
            kwargs["where"] = dict(where)
        return self._collection.query(**kwargs)

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str],
    ) -> Any:
        kwargs: dict[str, Any] = {"include": list(include)}
        if ids is not None:
            kwargs["ids"] = list(ids)
        if limit is not None:
            kwargs["limit"] = limit
            kwargs["offset"] = offset
        return self._collection.get(**kwargs)

    def delete(self, ids: Sequence[str]) -> None:
        self._collection.delete(ids=list(ids))

    def count(self) -> int:
        return int(self._collection.count())

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._collection.modify(metadata=dict(metadata))

    def close(self) -> None:
        # The Chroma client is owned by RAGStore's shared-client cache.
        return None
