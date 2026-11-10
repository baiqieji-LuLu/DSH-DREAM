"""Local ``bge-head`` router backend.

The backend reuses the embedding encoder that the RAG store already loads
(BGE ONNX by default) and adds a classifier head on top:

* zero-config mode: cosine prototypes / kNN over bundled seed sentences
  (:data:`SEEDS_PATH`), no training weights required;
* trained mode: a 512-dim logistic head loaded from ``bge_head.npz`` and
  validated against ``bge_head.json``.

Either way the work per query is one ``embed_query`` plus one dot product.
The encoder is only ever loaded from :meth:`BgeHead.warmup`; the request path
never triggers provisioning or a model download.

中文说明：本地 ``bge-head`` 后端复用 RAG 已加载的 BGE 向量，再叠加一层分类头。
第一版用随包分发的种子句做余弦原型（零配置、无权重）；有 ``bge_head.npz`` 时
升级为 logistic 头。模型只在 warmup 加载，请求路径绝不做下载或预热。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from .base import (
    DEFAULT_THRESHOLD,
    DEFAULT_UNCERTAINTY_BAND,
    LOCAL_MODEL_MISMATCH,
    LOCAL_NOT_READY,
    LocalRouteResult,
    RouterBackendError,
    clamp_probability,
    sigmoid,
)

try:  # numpy is a hard dependency of the package, but keep imports defensive.
    import numpy as np
except Exception:  # pragma: no cover - only on a broken install
    np = None  # type: ignore[assignment]

ROUTER_PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
SEEDS_PATH: Final[Path] = ROUTER_PACKAGE_DIR / "seeds_zh.json"
DEFAULT_BACKEND_ID: Final[str] = "bge-head"
DEFAULT_EMBEDDING_MODEL: Final[str] = "bge-small-zh-v1.5"
DEFAULT_MODEL_ID: Final[str] = f"{DEFAULT_EMBEDDING_MODEL}:head:v1"
_WEIGHTS_FILE: Final[str] = "bge_head.npz"
_METADATA_FILE: Final[str] = "bge_head.json"


def _l2_normalize(vector: Any) -> Any:
    """Return an L2-normalized float32 vector (zero vectors stay zero)."""
    assert np is not None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 0.0:
        return array
    return array / norm


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BgeHead:
    """Cosine-prototype / logistic head over the RAG embedding encoder.

    Args:
        embedder: Callable mapping one text to one embedding vector.  The plugin
            passes ``RAGStore.embed_query_text`` so the router and retrieval use
            exactly the same instruction and model.
        model_dir: Optional directory holding ``bge_head.npz`` and
            ``bge_head.json``.  Missing files fall back to seed prototypes.
        seeds_path: Optional override for the bundled seed sentence file.
        threshold: Fallback decision threshold when metadata provides none.
        uncertainty_band: Fallback abstention band.

    中文说明：复用 RAG 的 embedder，默认走种子句原型，有训练头时走 logistic。
    """

    def __init__(
        self,
        *,
        embedder: Callable[[str], Sequence[float]],
        model_dir: str | Path | None = None,
        seeds_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        uncertainty_band: float = DEFAULT_UNCERTAINTY_BAND,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        backend_id: str = DEFAULT_BACKEND_ID,
    ) -> None:
        self._embedder = embedder
        self._model_dir = Path(model_dir).expanduser() if model_dir else None
        self._seeds_path = Path(seeds_path) if seeds_path else SEEDS_PATH
        self._embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.backend_id = backend_id
        self._threshold = float(threshold)
        self._uncertainty_band = float(uncertainty_band)
        self._model_id = f"{self._embedding_model}:head:v1"
        self._kind = "prototype"
        self._dimension = 0
        self._ready = False
        self._error: str | None = None
        self._warmup_ms = 0.0
        self._seed_version = ""
        self._positive: Any = None
        self._negative: Any = None
        self._weights: Any = None
        self._bias = 0.0
        self._margin = 0.0
        self._temperature = 0.05
        self._threshold_source = "default"

    # ------------------------------------------------------------------ state

    @property
    def model_id(self) -> str:
        """Model identity used in audit records and the route cache key."""
        return self._model_id

    @property
    def ready(self) -> bool:
        """Whether warmup completed and inference can run immediately."""
        return self._ready

    @property
    def error(self) -> str | None:
        """Last warmup failure, if any."""
        return self._error

    @property
    def dimension(self) -> int:
        """Embedding dimension discovered during warmup."""
        return self._dimension

    @property
    def threshold(self) -> float:
        """Decision threshold resolved from metadata or configuration."""
        return self._threshold

    @property
    def uncertainty_band(self) -> float:
        """Abstention band resolved from metadata or configuration."""
        return self._uncertainty_band

    @property
    def kind(self) -> str:
        """``prototype`` or ``logistic``."""
        return self._kind

    def describe(self) -> dict[str, object]:
        """Return JSON-safe backend metadata for ``router.status``."""
        return {
            "backend": self.backend_id,
            "model_id": self._model_id,
            "kind": self._kind,
            "dimension": self._dimension,
            "threshold": self._threshold,
            "uncertainty_band": self._uncertainty_band,
            "embedding_model": self._embedding_model,
            "seed_version": self._seed_version,
            "warmup_ms": self._warmup_ms,
            "ready": self._ready,
            "error": self._error,
        }

    # ----------------------------------------------------------------- warmup

    def warmup(self) -> None:
        """Load seeds, optionally weights, and verify the embedding dimension.

        Raises:
            RouterBackendError: ``LOCAL_NOT_READY`` when seeds/NumPy are
                unavailable, ``LOCAL_MODEL_MISMATCH`` when a trained head does
                not match the active encoder or fails its SHA check.
        """
        if self._ready:
            return
        started = time.perf_counter()
        try:
            if np is None:
                raise RouterBackendError(
                    LOCAL_NOT_READY, "numpy is not available for the router head"
                )
            seeds = self._load_seeds()
            positive = list(seeds["positive"])
            negative = list(seeds["negative"])
            self._margin = float(seeds.get("margin", 0.0) or 0.0)
            temperature = float(seeds.get("temperature", 0.05) or 0.05)
            self._temperature = temperature if temperature > 0 else 0.05
            self._seed_version = str(seeds.get("version", ""))

            positive_vectors = [self._encode(text) for text in positive]
            negative_vectors = [self._encode(text) for text in negative]
            self._positive = np.stack(positive_vectors)
            self._negative = np.stack(negative_vectors)
            self._dimension = int(self._positive.shape[1])

            self._load_trained_head()
            self._ready = True
            self._error = None
        except RouterBackendError as exc:
            self._ready = False
            self._error = exc.message
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._ready = False
            self._error = f"{type(exc).__name__}: {exc}"
            raise RouterBackendError(LOCAL_NOT_READY, self._error) from exc
        finally:
            self._warmup_ms = (time.perf_counter() - started) * 1000.0

    def _load_seeds(self) -> dict[str, Any]:
        path = self._seeds_path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RouterBackendError(
                LOCAL_NOT_READY, f"router seeds unavailable: {path}"
            ) from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - broken asset
            raise RouterBackendError(
                LOCAL_NOT_READY, f"router seeds are not valid JSON: {path}"
            ) from exc
        if not isinstance(raw, dict):
            raise RouterBackendError(LOCAL_NOT_READY, "router seeds must be an object")
        positive = raw.get("positive")
        negative = raw.get("negative")
        if not isinstance(positive, list) or not positive:
            raise RouterBackendError(LOCAL_NOT_READY, "router seeds need a positive set")
        if not isinstance(negative, list) or not negative:
            raise RouterBackendError(LOCAL_NOT_READY, "router seeds need a negative set")
        return raw

    def _load_trained_head(self) -> None:
        """Load and validate a trained logistic head when one is installed."""
        if self._model_dir is None:
            return
        weights_path = self._model_dir / _WEIGHTS_FILE
        metadata_path = self._model_dir / _METADATA_FILE
        if not weights_path.exists() and not metadata_path.exists():
            return
        if not weights_path.exists() or not metadata_path.exists():
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH,
                "bge_head.npz and bge_head.json must be installed together",
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH, f"cannot read bge_head.json: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise RouterBackendError(LOCAL_MODEL_MISMATCH, "bge_head.json must be an object")

        declared_dimension = int(metadata.get("dimension") or 0)
        if declared_dimension and declared_dimension != self._dimension:
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH,
                f"head dimension {declared_dimension} != encoder dimension {self._dimension}",
            )
        declared_model = str(metadata.get("embedding_model") or "")
        if declared_model and declared_model != self._embedding_model:
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH,
                f"head was trained for {declared_model!r}, not {self._embedding_model!r}",
            )
        expected_sha = str(metadata.get("sha256") or "")
        if expected_sha and expected_sha != _sha256_of(weights_path):
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH, "bge_head.npz SHA256 does not match bge_head.json"
            )

        assert np is not None
        try:
            with np.load(weights_path) as payload:
                weights = np.asarray(payload["w"], dtype=np.float32).reshape(-1)
                bias = float(np.asarray(payload["b"], dtype=np.float32).reshape(-1)[0])
        except Exception as exc:
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH, f"cannot read bge_head.npz: {exc}"
            ) from exc
        if weights.shape[0] != self._dimension:
            raise RouterBackendError(
                LOCAL_MODEL_MISMATCH,
                f"head weights have {weights.shape[0]} values, expected {self._dimension}",
            )
        self._weights = _l2_normalize(weights)
        self._bias = bias
        self._kind = "logistic"
        self._model_id = str(metadata.get("model_id") or DEFAULT_MODEL_ID)
        if metadata.get("threshold") is not None:
            self._threshold = clamp_probability(float(metadata["threshold"]))
            self._threshold_source = "metadata"
        if metadata.get("uncertainty_band") is not None:
            band = float(metadata["uncertainty_band"])
            self._uncertainty_band = max(0.0, min(0.5, band))

    # -------------------------------------------------------------- inference

    def _encode(self, text: str) -> Any:
        if np is None:  # pragma: no cover - guarded by warmup
            raise RouterBackendError(LOCAL_NOT_READY, "numpy is not available")
        vector = self._embedder(str(text or "").strip())
        return _l2_normalize(vector)

    def probability(self, vector: Any) -> float:
        """Return ``P(need_rag = 1)`` for one normalized embedding."""
        assert np is not None
        normalized = _l2_normalize(vector)
        if self._kind == "logistic" and self._weights is not None:
            return clamp_probability(sigmoid(float(normalized @ self._weights) + self._bias))
        if self._positive is None or self._negative is None:
            raise RouterBackendError(LOCAL_NOT_READY, "router head is not warmed up")
        positive = float(np.max(self._positive @ normalized))
        negative = float(np.max(self._negative @ normalized))
        return clamp_probability(
            sigmoid((positive - negative - self._margin) / self._temperature)
        )

    def route(self, text: str, deadline_ms: int = 200) -> LocalRouteResult:
        """Classify ``text`` and return a calibrated local result.

        The head itself does not enforce ``deadline_ms``; the service owns the
        budget check.  Raises ``LOCAL_NOT_READY`` before warmup so callers can
        degrade instead of blocking.
        """
        del deadline_ms  # enforced by RouterService
        if not self._ready:
            raise RouterBackendError(LOCAL_NOT_READY, self._error or "router head is not ready")
        started = time.perf_counter()
        probability = self.probability(self._encode(text))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LocalRouteResult.from_probability(
            probability=probability,
            backend=self.backend_id,
            model_id=self._model_id,
            latency_ms=latency_ms,
            threshold=self._threshold,
            uncertainty_band=self._uncertainty_band,
        )


__all__ = [
    "BgeHead",
    "DEFAULT_BACKEND_ID",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MODEL_ID",
    "ROUTER_PACKAGE_DIR",
    "SEEDS_PATH",
]
