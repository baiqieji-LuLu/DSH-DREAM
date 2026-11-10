"""Contracts for the DSH-DREAM local RAG router (v0.4.0).

The router answers exactly one question: **does this turn need long-term
memory retrieval?**  Its output is a decision (``retrieve`` / ``skip``) plus a
probability; it must never generate dialogue text or explanations.

This module is intentionally dependency-free so that both the Python service
and its tests can import the contracts without loading NumPy or an embedding
model.

中文说明：RouterEngine 的本地契约层。只回答「这一轮是否需要检索长期记忆」，
输出布尔决策与概率，不生成任何对话内容。这里同时固定「模糊即查」的默认态度：
概率落在不确定带内、超时或异常时一律返回 ``retrieve``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

DECISION_RETRIEVE: Literal["retrieve"] = "retrieve"
DECISION_SKIP: Literal["skip"] = "skip"
RouteDecision = Literal["retrieve", "skip"]

#: Probability band around the threshold where the model abstains.
DEFAULT_THRESHOLD = 0.55
DEFAULT_UNCERTAINTY_BAND = 0.10
#: Hard router budget, measured from the JS ``route()`` entry point.
DEFAULT_BUDGET_MS = 200

# Error codes shared with the Node host (see DESIGN_ROUTER.md 5.7).
LOCAL_NOT_READY = "LOCAL_NOT_READY"
LOCAL_BUSY = "LOCAL_BUSY"
LOCAL_TIMEOUT = "LOCAL_TIMEOUT"
LOCAL_BUDGET_EXCEEDED = "LOCAL_BUDGET_EXCEEDED"
LOCAL_MODEL_MISMATCH = "LOCAL_MODEL_MISMATCH"
LOCAL_BACKEND_DISABLED = "LOCAL_BACKEND_DISABLED"
LOCAL_INTERNAL_ERROR = "INTERNAL"


class RouterBackendError(RuntimeError):
    """A local-router failure carrying a stable error code.

    中文说明：本地路由错误携带稳定错误码，Node 端据此降级，不抛出到主流程。
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def clamp_probability(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``; non-finite input becomes ``0.5``.

    中文说明：把概率限制在 0..1；非法值回落到 0.5（即「不确定」）。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(number):
        return 0.5
    return min(1.0, max(0.0, number))


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def decide(
    probability: float,
    threshold: float = DEFAULT_THRESHOLD,
    uncertainty_band: float = DEFAULT_UNCERTAINTY_BAND,
) -> tuple[RouteDecision, bool, float]:
    """Map a probability to ``(decision, uncertain, confidence)``.

    Neutral default: when the probability sits inside the uncertainty band the
    decision is ``retrieve`` with ``uncertain=True``.  The router would rather
    over-retrieve than silently lose memory.

    中文说明：概率落在阈值带内时返回 ``retrieve`` 且 ``uncertain=True``：
    宁可多查，不可失忆。带宽两端（如阈值 0.55、带宽 0.10 时的 0.45）同样
    视为模糊地带，因此下边界用严格小于判断。
    """
    p = clamp_probability(probability)
    band = max(0.0, min(0.5, float(uncertainty_band)))
    mid = min(1.0, max(0.0, float(threshold)))
    low = max(0.0, mid - band)
    high = min(1.0, mid + band)
    confidence = max(p, 1.0 - p)
    # Float tolerance: 0.55 - 0.10 must treat exactly 0.45 as in-band.
    if p >= high - 1e-9:
        return DECISION_RETRIEVE, False, confidence
    if p < low - 1e-9:
        return DECISION_SKIP, False, confidence
    return DECISION_RETRIEVE, True, confidence


@dataclass(frozen=True, slots=True)
class LocalRouteResult:
    """One local routing conclusion.

    Attributes mirror the ``op=route`` response documented in
    ``DESIGN_ROUTER.md`` 5.4.  ``threshold`` and ``uncertainty_band`` travel with
    the result because the local head owns its own calibration; the Node
    arbiter must not reuse the cloud threshold for it.
    """

    backend: str
    model_id: str
    probability: float
    decision: RouteDecision
    confidence: float
    uncertain: bool
    threshold: float
    uncertainty_band: float
    latency_ms: float

    @classmethod
    def from_probability(
        cls,
        *,
        probability: float,
        backend: str,
        model_id: str,
        latency_ms: float,
        threshold: float = DEFAULT_THRESHOLD,
        uncertainty_band: float = DEFAULT_UNCERTAINTY_BAND,
    ) -> LocalRouteResult:
        """Build a result, applying the neutral ``uncertain -> retrieve`` rule."""
        decision, uncertain, confidence = decide(
            probability, threshold, uncertainty_band
        )
        return cls(
            backend=backend,
            model_id=model_id,
            probability=clamp_probability(probability),
            decision=decision,
            confidence=confidence,
            uncertain=uncertain,
            threshold=threshold,
            uncertainty_band=uncertainty_band,
            latency_ms=max(0.0, float(latency_ms)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON payload emitted by ``op=route``."""
        return {
            "ok": True,
            "backend": self.backend,
            "model_id": self.model_id,
            "probability": self.probability,
            "decision": self.decision,
            "confidence": self.confidence,
            "uncertain": self.uncertain,
            "threshold": self.threshold,
            "uncertainty_band": self.uncertainty_band,
            "latency_ms": self.latency_ms,
        }


@runtime_checkable
class LocalRouter(Protocol):
    """Structural contract implemented by :class:`BgeHead` backends.

    中文说明：本地路由后端协议。实现方必须提供就绪状态、预热与一次短推理。
    """

    backend_id: str
    model_id: str

    @property
    def ready(self) -> bool:
        """Whether warmup finished and inference can start immediately."""
        ...

    def warmup(self) -> None:
        """Load encoder/head assets.  Never called from the request path."""
        ...

    def route(self, text: str, deadline_ms: int = DEFAULT_BUDGET_MS) -> LocalRouteResult:
        """Classify one query within ``deadline_ms``."""
        ...


__all__ = [
    "DECISION_RETRIEVE",
    "DECISION_SKIP",
    "DEFAULT_BUDGET_MS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_UNCERTAINTY_BAND",
    "LOCAL_BACKEND_DISABLED",
    "LOCAL_BUDGET_EXCEEDED",
    "LOCAL_BUSY",
    "LOCAL_INTERNAL_ERROR",
    "LOCAL_MODEL_MISMATCH",
    "LOCAL_NOT_READY",
    "LOCAL_TIMEOUT",
    "LocalRouteResult",
    "LocalRouter",
    "RouteDecision",
    "RouterBackendError",
    "clamp_probability",
    "decide",
    "sigmoid",
]
