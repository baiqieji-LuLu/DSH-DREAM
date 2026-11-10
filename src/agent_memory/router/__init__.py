"""Local RAG routing layer for DSH-DREAM v0.4.0.

The package answers one binary question before retrieval: does this turn need
long-term memory?  It never produces dialogue text, explanations, or prompt
fragments.  The Node ``RouterEngine`` owns arbitration, budget, cache, circuit
breaking, and audit; this package owns the local model and its lifecycle.

中文说明：本地路由层。只回答「本轮是否需要检索长期记忆」这一个布尔问题，
不生成任何对话内容；仲裁、预算、缓存、熔断与审计都在 Node 侧完成。
"""

from .base import (
    DEFAULT_BUDGET_MS,
    DEFAULT_THRESHOLD,
    DEFAULT_UNCERTAINTY_BAND,
    LocalRouteResult,
    LocalRouter,
    RouterBackendError,
    decide,
)
from .bge_head import BgeHead, SEEDS_PATH
from .service import RouterService

__all__ = [
    "DEFAULT_BUDGET_MS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_UNCERTAINTY_BAND",
    "BgeHead",
    "LocalRouteResult",
    "LocalRouter",
    "RouterBackendError",
    "RouterService",
    "SEEDS_PATH",
    "decide",
]
