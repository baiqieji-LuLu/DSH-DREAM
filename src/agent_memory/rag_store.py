"""Whitelist-only local vector memory backed by Chroma.

The store is a security boundary, not a general-purpose vector database
wrapper. Every write must declare a trusted ``source_type``, and only three
provenance classes are accepted:

* ``user_message`` - words authored by the user.
* ``system_fact`` - facts explicitly promoted by the host system.
* ``tool_result`` - output returned by an approved tool.

``model_hint`` is categorically rejected before embedding or persistence. The
same whitelist is enforced again on reads so a manually corrupted collection
cannot smuggle a model-authored hint into a prompt.

Chroma and sentence-transformers are imported lazily. Tests and embedded hosts
can inject a compatible client and embedding function without installing the
heavy runtime dependencies.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy ships with chromadb/sentence-transformers
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    BM25Okapi = None

ALLOWED_SOURCE_TYPES: Final[frozenset[str]] = frozenset(
    {"user_message", "system_fact", "tool_result", "assistant_message"}
)
"""Source types that may enter the vector store."""

FORBIDDEN_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"model_hint"})
"""Source types that are always rejected, even on reads."""

SourceType: TypeAlias = Literal[
    "user_message", "system_fact", "tool_result", "assistant_message"
]
MetadataValue: TypeAlias = str | int | float | bool
_RESERVED_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {"source_type", "created_at", "timestamp_ms", "content_sha256"}
)
_DEFAULT_MODEL: Final[str] = "BAAI/bge-small-zh-v1.5"
_FALLBACK_MODEL: Final[str] = "all-MiniLM-L6-v2"
_DEFAULT_CLOUD_MODEL: Final[str] = "text-embedding-3-small"
_DEFAULT_CLOUD_BASE_URL: Final[str] = ""
_DIMENSION_KEY: Final[str] = "embedding_dimension"
_MAX_ADD_BATCH_SIZE: Final[int] = 5_000
MIN_VECTOR_SCORE: Final[float] = 0.35
RRF_K: Final[int] = 60
DEFAULT_FLAT_MAX_RECORDS: Final[int] = 200_000

# State predicates, not entity words.  A conflict requires the same predicate
# family with *different* values (e.g. "key is at A" vs "key is at B").  A
# sequence of actions ("bought", "broke", "bought a replacement") has no
# state-slot pair and therefore must not be collapsed by latest-wins.
STATE_SLOT_FAMILIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "location",
        ("放在了", "放到了", "放在", "放到", "塞在", "藏在", "收在", "搁在", "存在", "位于"),
    ),
    (
        "copula",
        ("设置为", "改成", "换成", "是"),
    ),
)


class RAGStoreError(Exception):
    """Base class for vector-store failures."""


class SourceTypeError(RAGStoreError, ValueError):
    """Raised when a source type is unknown or not permitted."""


class ForbiddenSourceTypeError(SourceTypeError):
    """Raised when a caller attempts to store or query ``model_hint``."""


class MemoryValidationError(RAGStoreError, ValueError):
    """Raised when memory content, ID, metadata, or query parameters are invalid."""


class RAGStoreUnavailableError(RAGStoreError, ImportError):
    """Raised when Chroma or the default embedding stack is unavailable."""


def validate_source_type(source_type: str, *, for_filter: bool = False) -> SourceType:
    """Validate and normalize a provenance label.

    ``model_hint`` is rejected explicitly, with a more specific exception than
    other unknown labels. This function is shared by the store and the plugin
    entry points so the whitelist has exactly one implementation.

    Args:
        source_type: Caller-provided provenance label.
        for_filter: Adjust the error wording for read filters.

    Returns:
        The normalized lower-case source type.

    Raises:
        ForbiddenSourceTypeError: If the label is ``model_hint``.
        SourceTypeError: If the label is not in the write whitelist.
    """
    action = "filter memory" if for_filter else "store memory"
    if not isinstance(source_type, str):
        raise SourceTypeError(f"source_type must be a string when used to {action}")
    normalized = source_type.strip().lower()
    if normalized in FORBIDDEN_SOURCE_TYPES:
        raise ForbiddenSourceTypeError(
            f"{source_type!r} is forbidden: model-authored hints can never be used "
            f"to {action}"
        )
    if normalized not in ALLOWED_SOURCE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_SOURCE_TYPES))
        raise SourceTypeError(
            f"unsupported source_type {source_type!r}; expected one of: {allowed}"
        )
    return cast(SourceType, normalized)


class OpenAICompatibleEmbeddingFunction:
    """Minimal OpenAI-compatible ``/embeddings`` HTTP embedding function.

    The API key is kept only in memory and is deliberately omitted from
    Chroma's persisted ``get_config()`` payload.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
    ) -> None:
        if not model.strip():
            raise RAGStoreUnavailableError("cloud embedding model must not be empty")
        if not api_key.strip():
            raise RAGStoreUnavailableError(
                "cloud embedding_api_key is required for openai_compatible"
            )
        if not base_url.strip():
            raise RAGStoreUnavailableError(
                "cloud embedding_base_url is required for openai_compatible"
            )
        if not base_url.strip().lower().startswith(("http://", "https://")):
            raise RAGStoreUnavailableError(
                "embedding_base_url must start with http:// or https://"
            )
        self.model = model.strip()
        self._api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = timeout

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        texts = [str(text) for text in input]
        if not texts:
            return []
        payload = json.dumps(
            {"model": self.model, "input": texts},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RAGStoreUnavailableError(
                f"embedding API returned HTTP {exc.code}: {body or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RAGStoreUnavailableError(
                f"cannot reach embedding API at {self.base_url}: {exc.reason}"
            ) from exc
        try:
            data = json.loads(raw)
            rows = sorted(data["data"], key=lambda row: int(row.get("index", 0)))
            vectors = [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RAGStoreUnavailableError(
                "embedding API returned an invalid /embeddings payload"
            ) from exc
        if len(vectors) != len(texts):
            raise RAGStoreUnavailableError(
                f"embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) <= 0:
            raise RAGStoreUnavailableError(
                "embedding API returned inconsistent or empty vector dimensions"
            )
        return vectors

    def embed_query(self, input: Sequence[str]) -> list[list[float]]:
        """Embed query texts for Chroma's query path.

        Chroma 1.x routes ``Collection.query`` through ``embed_query`` while
        document inserts call ``__call__``; both share the same HTTP request.
        中文说明：Chroma 1.x 查询时会调用 ``embed_query``，写入时调用 ``__call__``，
        两者复用同一个 HTTP 请求。
        """
        return self(input)

    @staticmethod
    def name() -> str:
        return "agent-memory-openai-compatible"

    def get_config(self) -> dict[str, str]:
        # Never persist the bearer token in Chroma's collection configuration.
        return {"model": self.model, "base_url": self.base_url}

    @staticmethod
    def build_from_config(config: Mapping[str, Any]) -> Any:
        return OpenAICompatibleEmbeddingFunction(
            model=str(config.get("model", _DEFAULT_CLOUD_MODEL)),
            api_key="",
            base_url=str(config.get("base_url", _DEFAULT_CLOUD_BASE_URL)),
        )


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One ranked retrieval result.

    Attributes:
        memory_id: Stable Chroma record ID.
        content: Stored memory text.
        source_type: Validated provenance class.
        metadata: Stored scalar metadata, including ``created_at`` and
            ``content_sha256``.
        distance: Chroma distance when supplied. Lower is more similar; the
            exact scale depends on the configured collection metric.
        retrieval_source: ``"vector"`` for HNSW hits, ``"keyword"`` for the
            SQLite/``$contains`` fallback.
    """

    memory_id: str
    content: str
    source_type: SourceType
    metadata: Mapping[str, MetadataValue]
    distance: float | None = None
    retrieval_source: str = "vector"


class RAGStore:
    """A local, persistent, whitelist-only memory store.

    Args:
        persist_directory: Directory used by ``chromadb.PersistentClient``.
            Defaults to ``.chroma``.
        collection_name: Chroma collection name.
        embedding_function: Optional compatible embedding callable. When
            omitted and ``client`` is also omitted, Chroma's
            ``SentenceTransformerEmbeddingFunction`` is used.
        client: Optional pre-built Chroma-compatible client, primarily for
            tests or hosts that manage the client lifecycle.
        model_name: Sentence-transformers model used by the default embedding
            function.
        device: Optional sentence-transformers device (for example ``"cpu"``).
        index_metadata: Optional advanced HNSW settings merged into the
            collection metadata, for example ``{"hnsw:search_ef": 2000}``.
        flat_max_records: When the collection has fewer than this many
            records, use an exact NumPy cosine scan instead of HNSW.  Set to
            zero to disable the Flat path.  Defaults to 200,000.

    Raises:
        RAGStoreUnavailableError: If Chroma or sentence-transformers is not
            installed when the default client/embedding stack is requested.
        MemoryValidationError: If configuration values are invalid.
        RAGStoreError: If Chroma cannot create the persistent collection.
    """

    def __init__(
        self,
        persist_directory: str | Path = ".chroma",
        collection_name: str = "agent_memory",
        *,
        embedding_function: Any | None = None,
        client: Any | None = None,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        index_metadata: Mapping[str, Any] | None = None,
        embedding_provider: str = "local",
        embedding_model: str = "",
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        auto_rebuild_collection: bool = True,
        flat_max_records: int = DEFAULT_FLAT_MAX_RECORDS,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise MemoryValidationError("collection_name must be a non-empty string")
        if not isinstance(model_name, str) or not model_name.strip():
            raise MemoryValidationError("model_name must be a non-empty string")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise MemoryValidationError("device must be a non-empty string when provided")
        if not isinstance(embedding_provider, str):
            raise MemoryValidationError("embedding_provider must be a string")
        provider = embedding_provider.strip().lower()
        if provider not in {"local", "openai_compatible"}:
            raise MemoryValidationError(
                "embedding_provider must be 'local' or 'openai_compatible'"
            )
        if (
            isinstance(flat_max_records, bool)
            or not isinstance(flat_max_records, int)
            or flat_max_records < 0
        ):
            raise MemoryValidationError(
                "flat_max_records must be a non-negative integer"
            )

        self._persist_directory = Path(persist_directory).expanduser()
        self._collection_name = collection_name
        self._lock = threading.RLock()
        self._last_timestamp_ms = 0
        self._device = device
        self._model_name = model_name
        self._embedding_provider = provider
        self._auto_rebuild_collection = bool(auto_rebuild_collection)
        self._flat_max_records = flat_max_records
        self._collection: Any = None
        self._custom_embedding = embedding_function is not None or client is not None
        self._query_instruction = ""
        # Lazy caches for the exact Flat scan, full-corpus BM25, and RRF.
        self._corpus_ids: list[str] | None = None
        self._corpus_documents: list[str] | None = None
        self._corpus_metadatas: list[Mapping[str, Any]] | None = None
        self._flat_embeddings: Any | None = None
        self._flat_unavailable = False
        self._bm25: Any | None = None
        self._bm25_unavailable = False

        active_client = client if client is not None else self._create_client()
        self._client = active_client
        active_embedding = embedding_function
        if active_embedding is None and client is None:
            if provider == "openai_compatible":
                if not embedding_base_url.strip():
                    raise RAGStoreError(
                        "embedding_provider=openai_compatible 时必须填写 embedding_base_url"
                        "（例如 https://api.deepseek.com/v1，请确认与 API Key 匹配）"
                    )
                cloud_model = embedding_model.strip() or _DEFAULT_CLOUD_MODEL
                active_embedding = OpenAICompatibleEmbeddingFunction(
                    model=cloud_model,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                )
                self._model_name = cloud_model
            else:
                local_model = embedding_model.strip() or model_name
                active_embedding, self._model_name = (
                    self._create_embedding_with_fallback(local_model, device)
                )
        self._embedding_function = active_embedding
        self._set_query_instruction()
        if provider == "local" and active_embedding is not None:
            self.ensure_embedding_ready()

        new_dimension = self._probe_embedding_dimension()
        collection_metadata: dict[str, Any] = {
            "hnsw:space": "cosine",
            # 100k-scale recall needs a wider search beam than the old 200.
            # construction_ef is also raised so the graph has enough
            # long-range edges for isolated but semantically close memories.
            "hnsw:search_ef": 1000,
            "hnsw:construction_ef": 400,
        }
        if new_dimension is not None:
            collection_metadata[_DIMENSION_KEY] = new_dimension
        if index_metadata is not None:
            collection_metadata.update(index_metadata)
        self._collection_metadata = dict(collection_metadata)

        self._open_collection(
            active_client,
            active_embedding,
            collection_metadata,
            new_dimension,
        )

    @property
    def persist_directory(self) -> Path:
        """Directory backing the persistent client."""
        return self._persist_directory

    @property
    def embedding_model_name(self) -> str:
        """Return the embedding model actually in use.

        中文说明：默认 BAAI 中文模型不可用时会回退到 MiniLM，此属性返回
        实际生效的模型名，便于状态展示与诊断。
        """
        return self._model_name

    @property
    def embedding_provider(self) -> str:
        """Return ``local`` or ``openai_compatible``."""
        return self._embedding_provider

    @property
    def auto_rebuild_collection(self) -> bool:
        """Return whether dimension-mismatched collections are rebuilt."""
        return self._auto_rebuild_collection

    @property
    def flat_max_records(self) -> int:
        """Return the exact-Flat threshold (0 disables it)."""
        return self._flat_max_records

    def _probe_embedding_dimension(self) -> int | None:
        """Measure the active model with the short ``dimension_check`` text."""
        if self._embedding_function is None:
            return None
        try:
            vectors = self._embedding_function(["dimension_check"])
            if not hasattr(vectors, "__len__") or len(vectors) == 0:
                raise RAGStoreError("embedding function returned no vectors")
            first = vectors[0]
            if not hasattr(first, "__len__") or len(first) == 0:
                raise RAGStoreError("embedding function returned an empty vector")
            dimension = len(first)
            if dimension <= 0:
                raise RAGStoreError("embedding function returned a zero dimension")
            return dimension
        except RAGStoreError:
            if self._custom_embedding:
                return None
            raise
        except Exception as exc:
            if self._custom_embedding:
                return None
            raise RAGStoreUnavailableError(
                f"cannot determine embedding dimension for {self._model_name!r}: {exc}"
            ) from exc

    def _existing_collection_dimension(self) -> int | None:
        """Read the stored dimension, falling back to the first embedding."""
        metadata = getattr(self._collection, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get(_DIMENSION_KEY)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        try:
            raw = self._collection.get(limit=1, include=["embeddings"])
            embeddings = raw.get("embeddings") if isinstance(raw, Mapping) else None
            if embeddings is not None and len(embeddings) > 0:
                first = embeddings[0]
                if first is not None and len(first) > 0:
                    return len(first)
        except Exception:
            return None
        return None

    def _set_collection_dimension(
        self,
        metadata: Mapping[str, Any],
        dimension: int,
    ) -> None:
        merged = dict(metadata)
        merged[_DIMENSION_KEY] = dimension
        try:
            self._collection.modify(metadata=merged)
        except Exception:
            # The collection may be freshly created with metadata already set,
            # or a compatible test double may not implement modify().
            pass

    def migrate_embeddings(
        self,
        new_embedding_function: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Re-embed every stored document with a new embedding function.

        Documents and metadata are cached in memory first, then the old
        collection is deleted and recreated and the cached text is embedded
        with the new model. IDs, source types, timestamps, and metadata are
        preserved.

        中文说明：先把旧 documents/metadatas 缓存在内存，再删除旧集合、
        用新模型重新计算向量并批量写回；保留原 ID、来源、时间戳和 metadata。
        使用云端 Embedding 时会产生 Token 消耗。
        """
        old_collection = self._collection
        raw = old_collection.get(include=["documents", "metadatas"])
        ids = self._flat_row(raw.get("ids")) if isinstance(raw, Mapping) else []
        documents = self._flat_row(raw.get("documents")) if isinstance(raw, Mapping) else []
        metadatas = self._flat_row(raw.get("metadatas")) if isinstance(raw, Mapping) else []

        cached: list[tuple[str, str, dict[str, Any]]] = []
        for index, memory_id in enumerate(ids):
            document = documents[index] if index < len(documents) else None
            if not isinstance(memory_id, str) or not isinstance(document, str):
                continue
            raw_metadata = metadatas[index] if index < len(metadatas) else {}
            record_metadata = (
                dict(raw_metadata)
                if isinstance(raw_metadata, Mapping)
                else {}
            )
            cached.append((memory_id, document, record_metadata))

        self._embedding_function = new_embedding_function
        self._set_query_instruction()
        new_dimension = self._probe_embedding_dimension()
        new_metadata = dict(metadata or self._collection_metadata)
        if new_dimension is not None:
            new_metadata[_DIMENSION_KEY] = new_dimension

        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:
            raise RAGStoreError(f"无法删除旧集合，重嵌入中止: {exc}") from exc
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=new_embedding_function,
                metadata=new_metadata,
            )
        except Exception as exc:
            raise RAGStoreError(f"无法创建重嵌入集合: {exc}") from exc
        self._collection_metadata = new_metadata

        for start in range(0, len(cached), _MAX_ADD_BATCH_SIZE):
            batch = cached[start : start + _MAX_ADD_BATCH_SIZE]
            self._collection.add(
                ids=[item[0] for item in batch],
                documents=[item[1] for item in batch],
                metadatas=[item[2] for item in batch],
            )
        self._invalidate_search_caches()
        return len(cached)

    def _open_collection(
        self,
        client: Any,
        embedding_function: Any,
        metadata: Mapping[str, Any],
        new_dimension: int | None,
    ) -> None:
        """Open an existing collection and validate its dimension first.

        中文说明：必须先读取旧集合的维度，不能先用新 metadata 调
        ``get_or_create_collection``，否则 Chroma 会把旧维度覆盖成新维度，
        导致维度防呆失效。
        """
        existing = None
        get_collection = getattr(client, "get_collection", None)
        if callable(get_collection):
            # Read the existing collection first WITHOUT forcing the new
            # embedding function. Chroma may reject get_collection when the
            # provider class changed, which would otherwise bypass the
            # dimension guard and silently overwrite metadata.
            try:
                existing = get_collection(name=self._collection_name)
            except Exception:
                try:
                    existing = get_collection(
                        name=self._collection_name,
                        embedding_function=embedding_function,
                    )
                except Exception:
                    existing = None

        if existing is not None:
            self._collection = existing
            count = self._collection.count()
            old_dimension = self._existing_collection_dimension()
            if (
                count > 0
                and new_dimension is not None
                and old_dimension is not None
                and old_dimension != new_dimension
            ):
                if not self._auto_rebuild_collection:
                    raise RAGStoreError(
                        f"向量维度从 {old_dimension} 变成了 {new_dimension}！"
                        "请手动清空数据库，或勾选自动重嵌入！"
                    )
                print(
                    "[WARNING] 维度不匹配，正在无损重嵌入旧数据...",
                    file=sys.stderr,
                    flush=True,
                )
                self.migrate_embeddings(embedding_function, metadata)
                return
            elif count > 0 and old_dimension is None:
                raise RAGStoreError(
                    "无法读取旧集合的向量维度。请手动清空 "
                    f"{self._persist_directory} 数据库后重试。"
                )
            else:
                if new_dimension is not None:
                    self._set_collection_dimension(metadata, new_dimension)
                try:
                    existing._embedding_function = embedding_function
                except Exception:
                    pass
                return

        try:
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=embedding_function,
                metadata=dict(metadata),
            )
        except Exception as exc:
            raise RAGStoreError(
                f"cannot open Chroma collection {self._collection_name!r} at "
                f"{self._persist_directory}"
            ) from exc
        if new_dimension is not None:
            self._set_collection_dimension(metadata, new_dimension)

    def ensure_embedding_ready(self) -> str:
        """Verify the default embedding model is locally loadable.

        If the preferred model is unavailable, switch to the MiniLM fallback
        instead of failing the agent/store. Custom injected embedding functions
        are never replaced.

        中文说明：在 Agent 初始化时检查默认模型；本地缺失时自动回退，
        不直接抛错。显式注入的 embedding 不会被替换。
        """
        if self._embedding_provider != "local":
            return self._model_name
        if self._custom_embedding or self._embedding_function is None:
            return self._model_name
        if self._model_name == _FALLBACK_MODEL:
            return self._model_name
        try:
            from sentence_transformers import SentenceTransformer

            SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        except Exception as exc:
            print(
                f"[rag_store] embedding model {self._model_name!r} unavailable; "
                f"falling back to {_FALLBACK_MODEL!r}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._switch_embedding(_FALLBACK_MODEL)
        return self._model_name

    def _set_query_instruction(self) -> None:
        """Configure the BGE query instruction when applicable."""
        if "bge" in self._model_name.casefold():
            self._query_instruction = (
                "为这个句子生成表示以用于检索相关文章："
            )
        else:
            self._query_instruction = ""

    def _switch_embedding(self, model_name: str) -> None:
        """Replace the active embedding function with a fallback model."""
        fallback_embedding = self._create_embedding_function(model_name, self._device)
        self._embedding_function = fallback_embedding
        self._model_name = model_name
        self._set_query_instruction()
        collection = getattr(self, "_collection", None)
        if collection is None:
            return
        try:
            collection._embedding_function = fallback_embedding
        except Exception as exc:  # pragma: no cover - Chroma internal shape
            print(
                f"[rag_store] could not attach fallback embedding to collection: {exc}",
                file=sys.stderr,
                flush=True,
            )

    @property
    def collection_name(self) -> str:
        """Name of the active Chroma collection."""
        return self._collection_name

    def __len__(self) -> int:
        return self.count()

    def add_memory(
        self,
        content: str,
        source_type: str,
        *,
        metadata: Mapping[str, MetadataValue] | None = None,
        memory_id: str | None = None,
        timestamp_ms: int | None = None,
    ) -> str:
        """Embed and persist one trusted memory.

        The ``source_type`` argument is mandatory. Validation occurs before the
        embedding function or Chroma collection is touched, so a rejected
        ``model_hint`` never produces an embedding or a stored record.

        Args:
            content: Non-empty memory text.
            source_type: One of ``user_message``, ``system_fact``, or
                ``tool_result``.
            metadata: Optional scalar metadata. Reserved keys are rejected so
                provenance cannot be overwritten.
            memory_id: Optional caller-supplied ID. A UUID-based ID is generated
                when omitted.
            timestamp_ms: Optional existing millisecond timestamp. Used by
                dual-write callers so the original user message and extracted
                fact share exactly the same timestamp. When omitted, a fresh
                monotonic timestamp is generated.

        Returns:
            The stored memory ID.

        Raises:
            ForbiddenSourceTypeError: If ``source_type`` is ``model_hint``.
            SourceTypeError: For any other non-whitelisted source type.
            MemoryValidationError: If content, ID, metadata, or timestamp is
                invalid.
        """
        normalized_source = validate_source_type(source_type)
        text = self._validate_content(content)
        record_id = self._validate_memory_id(memory_id)
        custom_metadata = self._validate_metadata(metadata)

        timestamp_ms = self._resolve_timestamp_ms(timestamp_ms)
        stored_metadata: dict[str, MetadataValue] = dict(custom_metadata)
        stored_metadata["source_type"] = normalized_source
        stored_metadata["created_at"] = self._utc_now()
        stored_metadata["timestamp_ms"] = timestamp_ms
        stored_metadata["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

        with self._lock:
            self._collection.add(
                ids=[record_id],
                documents=[text],
                metadatas=[stored_metadata],
            )
            self._invalidate_search_caches()
        return record_id

    def add_memories(self, memories: Sequence[tuple[str, str]]) -> list[str]:
        """Embed and persist several trusted memories with chunked batch writes.

        This is the batch path for imports and stress tests. Every record is
        validated before any write. Chroma rejects batches above its internal
        limit (currently 5,461), so the prepared records are submitted in
        chunks of 5,000 rather than one record at a time.

        Args:
            memories: ``(content, source_type)`` pairs.

        Returns:
            Stored memory IDs in input order.

        Raises:
            ForbiddenSourceTypeError: If any pair uses ``model_hint``.
            SourceTypeError: For any other non-whitelisted source type.
            MemoryValidationError: If any content is invalid.
        """
        if not memories:
            return []

        record_ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, MetadataValue]] = []
        created_at = self._utc_now()
        timestamp_ms = self._next_timestamp_ms()
        for content, source_type in memories:
            normalized_source = validate_source_type(source_type)
            text = self._validate_content(content)
            record_ids.append(self._validate_memory_id(None))
            texts.append(text)
            metadatas.append(
                {
                    "source_type": normalized_source,
                    "created_at": created_at,
                    "timestamp_ms": timestamp_ms,
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )

        with self._lock:
            for start in range(0, len(record_ids), _MAX_ADD_BATCH_SIZE):
                stop = start + _MAX_ADD_BATCH_SIZE
                self._collection.add(
                    ids=record_ids[start:stop],
                    documents=texts[start:stop],
                    metadatas=metadatas[start:stop],
                )
            self._invalidate_search_caches()
        return record_ids

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_types: Iterable[str] | None = None,
        min_vector_score: float = MIN_VECTOR_SCORE,
        timeline_mode: bool = False,
    ) -> list[MemoryHit]:
        """Backward-compatible alias for :meth:`search_memory`."""
        return self.search_memory(
            query,
            top_k=top_k,
            source_types=source_types,
            min_vector_score=min_vector_score,
            timeline_mode=timeline_mode,
        )

    def search_memory(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_types: Iterable[str] | None = None,
        min_vector_score: float = MIN_VECTOR_SCORE,
        timeline_mode: bool = False,
    ) -> list[MemoryHit]:
        """Hybrid retrieval with reciprocal-rank fusion (RRF).

        Vector search (exact Flat when the collection is below
        ``flat_max_records``, otherwise HNSW) and full-corpus BM25 are run for
        every query.  Their ranks are fused with ``1/(RRF_K + rank)``; no
        query-specific keyword list is consulted.

        By default the fused candidates pass through latest-wins conflict
        resolution.  With ``timeline_mode=True`` that step is skipped: the
        Top-K relevant raw memories are returned in ascending
        ``timestamp_ms`` order, preserving a sequence of events.
        """
        query_text = self._validate_query(query)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise MemoryValidationError("top_k must be a positive integer")
        if not isinstance(min_vector_score, (int, float)) or isinstance(
            min_vector_score, bool
        ):
            raise MemoryValidationError("min_vector_score must be a number")
        if not isinstance(timeline_mode, bool):
            raise MemoryValidationError("timeline_mode must be a boolean")
        selected_types = self._normalize_source_filter(source_types)

        # RRF needs a reasonable candidate pool from both retrievers.
        candidate_k = max(top_k * 10, 60)
        vector_hits = self._search_vector(
            query_text,
            top_k=candidate_k,
            selected_types=selected_types,
        )
        bm25_hits = self._bm25_search(
            query_text,
            top_k=candidate_k,
            selected_types=selected_types,
        )
        fused = self._rrf_fuse(vector_hits, bm25_hits)
        if timeline_mode:
            timeline = fused[:top_k]
            timeline.sort(
                key=lambda hit: (
                    self._hit_timestamp_ms(hit),
                    hit.memory_id,
                )
            )
            return timeline
        return self._rerank_recency(fused, query_text)[:top_k]

    def search_vector(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_types: Iterable[str] | None = None,
    ) -> list[MemoryHit]:
        """Run only the vector path (Flat below the threshold, else HNSW)."""
        query_text = self._validate_query(query)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise MemoryValidationError("top_k must be a positive integer")
        selected_types = self._normalize_source_filter(source_types)
        return self._search_vector(
            query_text,
            top_k=top_k,
            selected_types=selected_types,
        )

    def _search_vector(
        self,
        query_text: str,
        *,
        top_k: int,
        selected_types: set[str] | None,
    ) -> list[MemoryHit]:
        total = self.count()
        if total == 0:
            return []
        if (
            self._flat_max_records > 0
            and not self._flat_unavailable
            and total < self._flat_max_records
        ):
            flat_hits = self._search_flat(
                query_text,
                top_k=top_k,
                selected_types=selected_types,
            )
            if flat_hits is not None:
                return flat_hits

        # HNSW fallback for large collections, injected test clients, or
        # environments without numpy.
        with self._lock:
            encoded_query = (
                f"{self._query_instruction}{query_text}"
                if self._query_instruction
                else query_text
            )
            query_args: dict[str, Any] = {
                "query_texts": [encoded_query],
                "n_results": min(top_k, total),
                "include": ["documents", "metadatas", "distances"],
            }
            if selected_types is not None:
                query_args["where"] = self._source_filter(selected_types)
            result = self._collection.query(**query_args)
        return self._parse_hits(result)

    def _ensure_corpus_cache(self) -> bool:
        """Load ids/docs/metadatas (and embeddings for Flat) exactly once."""
        with self._lock:
            if self._corpus_ids is not None:
                return True
            if not _NUMPY_AVAILABLE:
                self._flat_unavailable = True
                self._bm25_unavailable = True
                return False
            total = self.count()
            if total == 0:
                self._corpus_ids = []
                self._corpus_documents = []
                self._corpus_metadatas = []
                self._flat_embeddings = np.empty((0, 0), dtype=np.float32)
                return True
            include_embeddings = (
                self._flat_max_records > 0 and total < self._flat_max_records
            )
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[Mapping[str, Any]] = []
            embeddings: list[Any] = []
            try:
                offset = 0
                while True:
                    include = ["documents", "metadatas"]
                    if include_embeddings:
                        include.append("embeddings")
                    raw = self._collection.get(
                        limit=5000,
                        offset=offset,
                        include=include,
                    )
                    if not isinstance(raw, Mapping):
                        raise RAGStoreError("Chroma returned an invalid get response")
                    # Chroma may return numpy arrays when embeddings are
                    # requested; list() avoids ambiguous truth-value checks.
                    raw_ids = raw.get("ids")
                    raw_docs = raw.get("documents")
                    raw_metas = raw.get("metadatas")
                    batch_ids = [] if raw_ids is None else list(raw_ids)
                    if not batch_ids:
                        break
                    batch_docs = [] if raw_docs is None else list(raw_docs)
                    batch_metas = [] if raw_metas is None else list(raw_metas)
                    if len(batch_docs) != len(batch_ids) or len(batch_metas) != len(batch_ids):
                        raise RAGStoreError("Chroma returned misaligned corpus batches")
                    ids.extend(str(value) for value in batch_ids)
                    documents.extend(str(value) for value in batch_docs)
                    metadatas.extend(
                        value if isinstance(value, Mapping) else {}
                        for value in batch_metas
                    )
                    if include_embeddings:
                        raw_embeddings = raw.get("embeddings")
                        if raw_embeddings is None:
                            raise RAGStoreError(
                                "Chroma did not return an embedding list"
                            )
                        # Collection.get returns a flat list/array of vectors;
                        # do NOT collapse it with _flat_row.
                        batch_embeddings = list(raw_embeddings)
                        if len(batch_embeddings) != len(batch_ids):
                            raise RAGStoreError(
                                "Chroma returned misaligned embedding batches"
                            )
                        embeddings.extend(batch_embeddings)
                    offset += len(batch_ids)
                    if offset >= total:
                        break
                if len(ids) != total:
                    raise RAGStoreError(
                        f"corpus cache expected {total} rows, got {len(ids)}"
                    )
                self._corpus_ids = ids
                self._corpus_documents = documents
                self._corpus_metadatas = metadatas
                if include_embeddings:
                    matrix = np.asarray(embeddings, dtype=np.float32)
                    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
                        raise RAGStoreError("invalid embedding matrix shape")
                    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    self._flat_embeddings = matrix / norms
                return True
            except Exception:
                # Injected test doubles or unusual clients may not support the
                # batch get shape.  Fall back to the legacy Chroma path.
                self._corpus_ids = None
                self._corpus_documents = None
                self._corpus_metadatas = None
                self._flat_embeddings = None
                self._flat_unavailable = True
                self._bm25_unavailable = True
                return False

    def _search_flat(
        self,
        query_text: str,
        *,
        top_k: int,
        selected_types: set[str] | None,
    ) -> list[MemoryHit] | None:
        if not _NUMPY_AVAILABLE or not self._ensure_corpus_cache():
            return None
        if self._flat_embeddings is None or self._corpus_ids is None:
            return None
        if not self._corpus_ids:
            return []
        if self._embedding_function is None:
            return None
        encoded_query = (
            f"{self._query_instruction}{query_text}"
            if self._query_instruction
            else query_text
        )
        query_vector = np.asarray(
            self._embedding_function([encoded_query])[0],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            return []
        query_vector = query_vector / norm
        sims = self._flat_embeddings @ query_vector
        if selected_types is not None:
            mask = np.asarray(
                [
                    str(metadata.get("source_type")) in selected_types
                    for metadata in (self._corpus_metadatas or [])
                ],
                dtype=bool,
            )
            sims = np.where(mask, sims, -np.inf)
        limit = min(top_k, len(sims))
        if limit <= 0:
            return []
        order = np.argsort(-sims)[:limit]
        hits: list[MemoryHit] = []
        for index in order.tolist():
            similarity = float(sims[index])
            if not np.isfinite(similarity):
                continue
            hit = self._build_hit(
                self._corpus_ids[index],
                (self._corpus_documents or [])[index],
                (self._corpus_metadatas or [])[index],
                1.0 - similarity,
                retrieval_source="vector",
            )
            if hit is not None:
                hits.append(hit)
        return hits

    def _rerank_recency(
        self,
        hits: list[MemoryHit],
        query: str,
    ) -> list[MemoryHit]:
        """Resolve event versions under a generic latest-wins policy.

        Records are grouped by lexical/semantic overlap, never by a hardcoded
        entity or slot list.  Within each group, only the newest timestamp is
        retained.  Conflict groups are ordered by their newest timestamp
        descending; singleton hits keep their original RRF order.  Same-turn
        fact/user/assistant triples share a timestamp and are all kept, with
        ``system_fact`` first.
        """
        del query  # grouping is query-independent by design
        if len(hits) < 2:
            return hits
        groups: list[list[MemoryHit]] = []
        for hit in hits:
            for group in groups:
                if self._same_conflict_group(hit, group[0]):
                    group.append(hit)
                    break
            else:
                groups.append([hit])

        ordered: list[MemoryHit] = []
        for group in groups:
            latest_timestamp = max(
                (self._hit_timestamp_ms(hit) for hit in group),
                default=0,
            )
            if len(group) == 1:
                latest = list(group)
            else:
                # Keep every record from the newest turn: a fact is a
                # highlight, not a replacement for the original
                # user/assistant messages.  The group's position in the
                # fused ranking is preserved; only its members are reduced
                # to the newest timestamp.
                latest = [
                    hit
                    for hit in group
                    if self._hit_timestamp_ms(hit) == latest_timestamp
                ]
            latest.sort(
                key=lambda hit: (
                    0 if hit.source_type == "system_fact" else 1,
                    -self._hit_timestamp_ms(hit),
                )
            )
            ordered.extend(latest)
        return self._fact_first_within_timestamp(ordered)

    @staticmethod
    def _fact_first_within_timestamp(hits: list[MemoryHit]) -> list[MemoryHit]:
        """Keep timestamp order but place facts before raw text in each turn."""
        buckets: dict[int, list[MemoryHit]] = {}
        order: list[int] = []
        for hit in hits:
            timestamp = RAGStore._hit_timestamp_ms(hit)
            if timestamp not in buckets:
                buckets[timestamp] = []
                order.append(timestamp)
            buckets[timestamp].append(hit)
        result: list[MemoryHit] = []
        for timestamp in order:
            bucket = buckets[timestamp]
            bucket.sort(
                key=lambda hit: 0 if hit.source_type == "system_fact" else 1
            )
            result.extend(bucket)
        return result

    @classmethod
    def _same_conflict_group(
        cls,
        left: MemoryHit,
        right: MemoryHit,
    ) -> bool:
        """Group only direct state-slot conflicts.

        A conflict exists only when both records contain the same state
        predicate family (location/copula) with *different* values, and they
        share an entity phrase.  Timeline actions such as "bought", "broke",
        "bought a replacement" have no state-slot pair and are not grouped.
        """
        left_slots = cls._state_slots(left.content)
        right_slots = cls._state_slots(right.content)
        if not left_slots or not right_slots:
            return False
        shared_phrase = cls._longest_common_substring_length(
            left.content, right.content
        )
        for left_family, left_value in left_slots:
            for right_family, right_value in right_slots:
                if left_family != right_family:
                    continue
                if cls._normalize_slot_value(left_value) == cls._normalize_slot_value(
                    right_value
                ):
                    continue
                # A copula often has a short anchor ("密码"), while location
                # phrases carry a longer entity/location phrase.
                minimum_shared = 2 if left_family == "copula" else 3
                if shared_phrase >= minimum_shared:
                    return True
        return False

    @classmethod
    def _state_slots(cls, text: str) -> list[tuple[str, str]]:
        """Extract (predicate family, value) state slots without an entity lexicon."""
        slots: list[tuple[str, str]] = []
        for family, cues in STATE_SLOT_FAMILIES:
            for cue in sorted(cues, key=len, reverse=True):
                prefix = r"(?<![但只还就总可])" if cue == "是" else ""
                pattern = prefix + re.escape(cue) + r"(?P<value>[^，。；！？\n]{1,40})"
                for match in re.finditer(pattern, text):
                    value = match.group("value").strip()
                    if value and (family, value) not in slots:
                        slots.append((family, value))
        return slots

    @staticmethod
    def _normalize_slot_value(value: str) -> str:
        return re.sub(r"[\s，。；！？、,.!?;:：]+", "", value)

    @staticmethod
    def _longest_common_substring_length(left: str, right: str) -> int:
        """Return the longest shared Chinese content substring length."""
        left_text = "".join(re.findall(r"[\u4e00-\u9fff]+", left))
        right_text = "".join(re.findall(r"[\u4e00-\u9fff]+", right))
        if not left_text or not right_text:
            return 0
        if len(left_text) > len(right_text):
            left_text, right_text = right_text, left_text
        previous = [0] * (len(right_text) + 1)
        best = 0
        for left_char in left_text:
            current = [0] * (len(right_text) + 1)
            for index, right_char in enumerate(right_text, start=1):
                if left_char == right_char:
                    current[index] = previous[index - 1] + 1
                    if current[index] > best:
                        best = current[index]
            previous = current
        return best

    @staticmethod
    def _lexical_tokens(text: str) -> list[str]:
        """Tokenize ASCII words and Chinese bigrams for BM25.

        中文按连续片段切二元词，避免“冰”把“冰箱”误当成“冰美式”。
        """
        folded = text.casefold()
        tokens = re.findall(r"[a-z0-9_]+", folded)
        for segment in re.findall(r"[\u4e00-\u9fff]+", folded):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(len(segment) - 1)
                )
        return tokens

    def _ensure_bm25(self) -> bool:
        """Build the full-corpus BM25 index once, lazily."""
        if self._bm25 is not None:
            return True
        if BM25Okapi is None or self._bm25_unavailable:
            return False
        if not self._ensure_corpus_cache():
            return False
        documents = self._corpus_documents or []
        if not documents:
            return False
        try:
            self._bm25 = BM25Okapi(
                [self._lexical_tokens(document) for document in documents]
            )
        except Exception:
            self._bm25_unavailable = True
            self._bm25 = None
            return False
        return True

    def _bm25_search(
        self,
        query_text: str,
        *,
        top_k: int,
        selected_types: set[str] | None,
    ) -> list[MemoryHit]:
        """Rank the entire corpus with BM25 (no candidate-window shortcut)."""
        if not _NUMPY_AVAILABLE or not self._ensure_bm25():
            return []
        query_tokens = self._lexical_tokens(query_text)
        if not query_tokens:
            return []
        bm25 = self._bm25
        if bm25 is None:
            return []
        try:
            scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
        except Exception:
            return []
        if scores.size == 0:
            return []
        if selected_types is not None:
            mask = np.asarray(
                [
                    str(metadata.get("source_type")) in selected_types
                    for metadata in (self._corpus_metadatas or [])
                ],
                dtype=bool,
            )
            scores = np.where(mask, scores, -np.inf)
        limit = min(top_k, len(scores))
        if limit <= 0:
            return []
        order = np.argsort(-scores)[:limit]
        hits: list[MemoryHit] = []
        for index in order.tolist():
            score = float(scores[index])
            if not np.isfinite(score) or score <= 0.0:
                continue
            hit = self._build_hit(
                (self._corpus_ids or [])[index],
                (self._corpus_documents or [])[index],
                (self._corpus_metadatas or [])[index],
                None,
                retrieval_source="bm25",
            )
            if hit is not None:
                hits.append(hit)
        return hits

    @staticmethod
    def _rrf_fuse(
        vector_hits: list[MemoryHit],
        bm25_hits: list[MemoryHit],
    ) -> list[MemoryHit]:
        """Reciprocal-rank fusion: score = 1 / (RRF_K + rank)."""
        entries: dict[str, dict[str, Any]] = {}
        for source, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                entry = entries.setdefault(
                    hit.memory_id,
                    {
                        "hit": hit,
                        "score": 0.0,
                        "sources": set(),
                        "best_rank": rank,
                    },
                )
                entry["score"] += 1.0 / (RRF_K + rank)
                entry["sources"].add(source)
                entry["best_rank"] = min(entry["best_rank"], rank)

        fused: list[MemoryHit] = []
        for entry in entries.values():
            hit = entry["hit"]
            sources = entry["sources"]
            if len(sources) > 1:
                hit = MemoryHit(
                    memory_id=hit.memory_id,
                    content=hit.content,
                    source_type=hit.source_type,
                    metadata=hit.metadata,
                    distance=hit.distance,
                    retrieval_source="hybrid",
                )
            elif "bm25" in sources:
                hit = MemoryHit(
                    memory_id=hit.memory_id,
                    content=hit.content,
                    source_type=hit.source_type,
                    metadata=hit.metadata,
                    distance=None,
                    retrieval_source="bm25",
                )
            else:
                hit = MemoryHit(
                    memory_id=hit.memory_id,
                    content=hit.content,
                    source_type=hit.source_type,
                    metadata=hit.metadata,
                    distance=hit.distance,
                    retrieval_source="vector",
                )
            fused.append(hit)
        fused.sort(
            key=lambda hit: (
                -entries[hit.memory_id]["score"],
                entries[hit.memory_id]["best_rank"],
                hit.memory_id,
            )
        )
        return fused

    @staticmethod
    def _hit_timestamp_ms(hit: MemoryHit) -> int:
        value = hit.metadata.get("timestamp_ms", 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return int(value)

    def get_memory(self, memory_id: str) -> MemoryHit | None:
        """Return a stored memory by ID, or ``None`` when absent.

        Non-whitelisted records are treated as absent.

        Raises:
            MemoryValidationError: If ``memory_id`` is invalid.
        """
        record_id = self._validate_memory_id(memory_id, allow_none=False)
        with self._lock:
            result = self._collection.get(
                ids=[record_id],
                include=["documents", "metadatas"],
            )
            return self._parse_single(result)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Returns:
            ``True`` if a whitelisted record existed and was deleted.
        """
        existing = self.get_memory(memory_id)
        if existing is None:
            return False
        with self._lock:
            self._collection.delete(ids=[existing.memory_id])
            self._invalidate_search_caches()
        return True

    def _invalidate_search_caches(self) -> None:
        """Drop corpus/Flat/BM25 caches after any write."""
        self._corpus_ids = None
        self._corpus_documents = None
        self._corpus_metadatas = None
        self._flat_embeddings = None
        self._bm25 = None
        self._flat_unavailable = False
        self._bm25_unavailable = False

    def count(self) -> int:
        """Return the number of stored records."""
        with self._lock:
            value = self._collection.count()
        if isinstance(value, bool) or not isinstance(value, int):
            raise RAGStoreError("Chroma returned a non-integer collection count")
        if value < 0:
            raise RAGStoreError("Chroma returned a negative collection count")
        return value

    def close(self) -> None:
        """Release the underlying Chroma client and its file handles.

        This matters on Windows, where an open HNSW/SQLite handle prevents a
        temporary persistence directory from being removed.
        """
        with self._lock:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def _parse_hits(self, result: Any) -> list[MemoryHit]:
        if not isinstance(result, Mapping):
            raise RAGStoreError("Chroma returned an invalid query response")
        ids_row = self._first_row(result.get("ids"))
        documents_row = self._first_row(result.get("documents"))
        metadatas_row = self._first_row(result.get("metadatas"))
        distances_row = self._first_row(result.get("distances"))

        hits: list[MemoryHit] = []
        for index, raw_id in enumerate(ids_row):
            document = documents_row[index] if index < len(documents_row) else None
            metadata = metadatas_row[index] if index < len(metadatas_row) else None
            distance = distances_row[index] if index < len(distances_row) else None
            hit = self._build_hit(
                raw_id,
                document,
                metadata,
                distance,
                retrieval_source="vector",
            )
            if hit is not None:
                hits.append(hit)
        return hits

    def _parse_get_hits(
        self,
        result: Any,
        *,
        retrieval_source: str,
    ) -> list[MemoryHit]:
        if not isinstance(result, Mapping):
            raise RAGStoreError("Chroma returned an invalid get response")
        ids_row = self._flat_row(result.get("ids"))
        documents_row = self._flat_row(result.get("documents"))
        metadatas_row = self._flat_row(result.get("metadatas"))
        hits: list[MemoryHit] = []
        for index, raw_id in enumerate(ids_row):
            document = documents_row[index] if index < len(documents_row) else None
            metadata = metadatas_row[index] if index < len(metadatas_row) else None
            hit = self._build_hit(
                raw_id,
                document,
                metadata,
                None,
                retrieval_source=retrieval_source,
            )
            if hit is not None:
                hits.append(hit)
        return hits

    def _parse_single(self, result: Any) -> MemoryHit | None:
        if not isinstance(result, Mapping):
            raise RAGStoreError("Chroma returned an invalid get response")
        # ``Collection.get`` returns flat lists, while ``Collection.query``
        # returns one nested list per query. Normalize both shapes.
        ids_row = self._flat_row(result.get("ids"))
        documents_row = self._flat_row(result.get("documents"))
        metadatas_row = self._flat_row(result.get("metadatas"))
        distances_row = self._flat_row(result.get("distances"))
        if not ids_row:
            return None
        document = documents_row[0] if documents_row else None
        metadata = metadatas_row[0] if metadatas_row else None
        distance = distances_row[0] if distances_row else None
        return self._build_hit(
            ids_row[0],
            document,
            metadata,
            distance,
            retrieval_source="vector",
        )

    @staticmethod
    def _build_hit(
        raw_id: Any,
        document: Any,
        metadata: Any,
        distance: Any,
        *,
        retrieval_source: str,
    ) -> MemoryHit | None:
        if not isinstance(raw_id, str) or not isinstance(document, str):
            return None
        if not isinstance(metadata, Mapping):
            return None
        raw_source = metadata.get("source_type")
        if not isinstance(raw_source, str) or raw_source not in ALLOWED_SOURCE_TYPES:
            return None
        clean_metadata: dict[str, MetadataValue] = {
            str(key): value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        numeric_distance: float | None = None
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            numeric_distance = float(distance)
        return MemoryHit(
            memory_id=raw_id,
            content=document,
            source_type=cast(SourceType, raw_source),
            metadata=clean_metadata,
            distance=numeric_distance,
            retrieval_source=retrieval_source,
        )

    @staticmethod
    def _first_row(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)) or not value:
            return []
        first = value[0]
        if not isinstance(first, (list, tuple)):
            return []
        return list(first)

    @staticmethod
    def _flat_row(value: Any) -> list[Any]:
        if not isinstance(value, (list, tuple)) or not value:
            return []
        if isinstance(value[0], (list, tuple)):
            return list(value[0])
        return list(value)

    @staticmethod
    def _normalize_source_filter(source_types: Iterable[str] | None) -> set[str] | None:
        if source_types is None:
            return None
        if isinstance(source_types, (str, bytes)):
            raise SourceTypeError("source_types must be an iterable of source types, not a string")
        normalized: set[str] = set()
        for source_type in source_types:
            normalized.add(validate_source_type(source_type, for_filter=True))
        if not normalized:
            raise SourceTypeError("source_types must contain at least one source type")
        return normalized

    @staticmethod
    def _source_filter(source_types: set[str]) -> dict[str, Any]:
        if len(source_types) == 1:
            return {"source_type": next(iter(source_types))}
        return {"source_type": {"$in": sorted(source_types)}}

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str):
            raise MemoryValidationError("content must be a string")
        text = content.strip()
        if not text:
            raise MemoryValidationError("content must be a non-empty string")
        return text

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise MemoryValidationError("query must be a string")
        text = query.strip()
        if not text:
            raise MemoryValidationError("query must be a non-empty string")
        return text

    @staticmethod
    def _validate_memory_id(memory_id: str | None, *, allow_none: bool = True) -> str:
        if memory_id is None:
            if not allow_none:
                raise MemoryValidationError("memory_id must be a non-empty string")
            return f"mem-{uuid.uuid4().hex}"
        if not isinstance(memory_id, str):
            raise MemoryValidationError("memory_id must be a string")
        normalized = memory_id.strip()
        if not normalized:
            raise MemoryValidationError("memory_id must be non-empty")
        if len(normalized) > 128:
            raise MemoryValidationError("memory_id must be at most 128 characters")
        if any(character.isspace() for character in normalized):
            raise MemoryValidationError("memory_id must not contain whitespace")
        return normalized

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, MetadataValue] | None,
    ) -> Mapping[str, MetadataValue]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise MemoryValidationError("metadata must be a mapping")
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise MemoryValidationError("metadata keys must be non-empty strings")
            if key in _RESERVED_METADATA_KEYS:
                raise MemoryValidationError(f"metadata key {key!r} is reserved")
            if isinstance(value, float) and not math.isfinite(value):
                raise MemoryValidationError(
                    f"metadata value for {key!r} must be a finite number"
                )
            if not isinstance(value, (str, int, float, bool)):
                raise MemoryValidationError(
                    f"metadata value for {key!r} must be a string, int, float, or bool"
                )
        return metadata

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def next_timestamp_ms(self) -> int:
        """Return a fresh, strictly increasing dual-write timestamp.

        中文说明：供 MemoryAgent 在未抽取到 system_fact 时先申请一个时间戳，
        再用它写入 user_message 与 assistant_message，确保同一回合完全绑定。
        """
        return self._next_timestamp_ms()

    def _resolve_timestamp_ms(self, timestamp_ms: int | None) -> int:
        """Use an explicit dual-write timestamp or allocate a fresh one.

        中文说明：双写时允许调用方显式传入同一 timestamp_ms；否则生成严格的
        单调毫秒时间戳。显式值必须是正整数。
        """
        if timestamp_ms is None:
            return self._next_timestamp_ms()
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms <= 0
        ):
            raise MemoryValidationError(
                "timestamp_ms must be a positive integer"
            )
        with self._lock:
            if timestamp_ms > self._last_timestamp_ms:
                self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _next_timestamp_ms(self) -> int:
        """Return a strictly increasing millisecond timestamp.

        The counter is monotonic even when several writes land in the same
        millisecond, which makes conflict ordering deterministic.
        """
        with self._lock:
            now_ms = time.time_ns() // 1_000_000
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = now_ms
            return now_ms

    def _create_client(self) -> Any:
        try:
            chromadb = importlib.import_module("chromadb")
        except ImportError as exc:
            raise RAGStoreUnavailableError(
                "RAGStore requires 'chromadb'. Install it with: pip install chromadb"
            ) from exc
        factory = getattr(chromadb, "PersistentClient", None)
        if not callable(factory):
            raise RAGStoreUnavailableError(
                "the installed 'chromadb' package does not expose PersistentClient"
            )

        # Chroma's Rust binding does not reliably create nested parent
        # directories on Windows and may otherwise fail with SQLite code 14
        # ("unable to open database file").
        try:
            self._persist_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RAGStoreError(
                f"cannot create the Chroma persistence directory: "
                f"{self._persist_directory}"
            ) from exc
        if not self._persist_directory.is_dir():
            raise RAGStoreError(
                f"Chroma persistence path is not a directory: {self._persist_directory}"
            )

        try:
            return factory(path=str(self._persist_directory))
        except Exception as first_error:
            # Ensure the failed attempt does not stay registered in Chroma's
            # process-wide system cache; otherwise the settings retry fails
            # with "An instance of Chroma already exists ... with different
            # settings" instead of exercising the intended fallback.
            self._clear_chroma_system_cache()
            # Some Chroma/Rust builds need explicit persistent settings even
            # after the directory exists. Retry once with an explicit Settings
            # object; if that also fails, report both causes.
            settings = self._create_chroma_settings()
            if settings is None:
                raise RAGStoreError(
                    f"cannot create a persistent Chroma client at "
                    f"{self._persist_directory}: {first_error}"
                ) from first_error
            try:
                return factory(path=str(self._persist_directory), settings=settings)
            except Exception as second_error:
                raise RAGStoreError(
                    f"cannot create a persistent Chroma client at "
                    f"{self._persist_directory}; first attempt: {first_error}; "
                    f"settings retry: {second_error}"
                ) from second_error

    @staticmethod
    def _create_chroma_settings() -> Any | None:
        """Create explicit persistent Chroma settings when available."""
        try:
            config_module = importlib.import_module("chromadb.config")
            settings_factory = getattr(config_module, "Settings", None)
            if not callable(settings_factory):
                return None
            return settings_factory(allow_reset=True, is_persistent=True)
        except Exception:
            return None

    @staticmethod
    def _clear_chroma_system_cache() -> None:
        """Best-effort removal of a failed Chroma system from its global cache."""
        try:
            module = importlib.import_module("chromadb.api.shared_system_client")
            client = getattr(module, "SharedSystemClient", None)
            clear = getattr(client, "clear_system_cache", None)
            if callable(clear):
                clear()
        except Exception:
            return

    @classmethod
    def _create_embedding_with_fallback(
        cls,
        model_name: str,
        device: str | None,
    ) -> tuple[Any, str]:
        """Create the preferred embedding function, falling back if needed.

        中文说明：优先加载默认中文模型；本地缺失或初始化失败时自动回退到
        MiniLM，并返回实际生效的模型名。
        """
        candidates: list[str] = [model_name]
        if model_name != _DEFAULT_MODEL:
            candidates.append(_DEFAULT_MODEL)
        if _FALLBACK_MODEL not in candidates:
            candidates.append(_FALLBACK_MODEL)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                embedding = cls._create_embedding_function(candidate, device)
                if candidate != model_name:
                    print(
                        f"[rag_store] embedding model {model_name!r} unavailable; "
                        f"falling back to {candidate!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                return embedding, candidate
            except Exception as exc:
                last_error = exc
                continue
        raise RAGStoreUnavailableError(
            f"cannot initialize any local embedding model from {candidates!r}"
        ) from last_error

    @staticmethod
    def _create_embedding_function(model_name: str, device: str | None) -> Any:
        try:
            module = importlib.import_module("chromadb.utils.embedding_functions")
        except ImportError as exc:
            raise RAGStoreUnavailableError(
                "the default embedding function requires 'sentence-transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        factory = getattr(module, "SentenceTransformerEmbeddingFunction", None)
        if not callable(factory):
            raise RAGStoreUnavailableError(
                "the installed 'chromadb' package does not expose "
                "SentenceTransformerEmbeddingFunction"
            )
        kwargs: dict[str, Any] = {"model_name": model_name}
        if device is not None:
            kwargs["device"] = device
        try:
            return factory(**kwargs)
        except Exception as exc:
            raise RAGStoreUnavailableError(
                f"cannot initialize embedding model {model_name!r}"
            ) from exc


__all__ = [
    "ALLOWED_SOURCE_TYPES",
    "FORBIDDEN_SOURCE_TYPES",
    "ForbiddenSourceTypeError",
    "MemoryHit",
    "MemoryValidationError",
    "MetadataValue",
    "RAGStore",
    "RAGStoreError",
    "RAGStoreUnavailableError",
    "SourceType",
    "SourceTypeError",
    "validate_source_type",
]
