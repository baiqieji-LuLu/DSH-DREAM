"""Thread-safe local router service used by ``turn_bridge.py``.

Design constraints (DESIGN_ROUTER.md 3.5 / 4.5):

* ``route`` is a short task: one embed plus one dot product, no IO, no model
  download, no provisioning;
* the lock is **non-blocking**: a concurrent ``memory_search`` owned by the same
  single-process bridge yields ``LOCAL_BUSY`` instead of queueing;
* warmup happens in a background thread and is never triggered from the request
  path;
* every failure is reported as a stable error code, never raised into the
  bridge loop.

中文说明：本地路由服务。请求路径只做一次 embedding + 一次点积；锁是非阻塞的，
忙时立刻返回 ``LOCAL_BUSY``；预热在后台线程完成，绝不在请求路径加载模型。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Final, Literal

from .base import (
    DEFAULT_BUDGET_MS,
    LOCAL_BUDGET_EXCEEDED,
    LOCAL_BUSY,
    LOCAL_INTERNAL_ERROR,
    LOCAL_NOT_READY,
    LocalRouter,
    RouterBackendError,
)

RouterState = Literal["unavailable", "warming", "ready", "failed"]
DEFAULT_WARMUP_TIMEOUT_MS: Final[int] = 30_000


class RouterService:
    """Own one local router backend, its lifecycle, and its budget.

    Args:
        router: Backend implementing :class:`~agent_memory.router.base.LocalRouter`.
        warmup_timeout_ms: Upper bound for a warmup attempt; informational only,
            the background thread cannot be interrupted safely.
        clock: Monotonic clock, injectable for tests.
        unavailable_reason: When the local backend cannot exist at all (for
            example a cloud embedding provider), pass a reason to keep
            ``route`` returning ``LOCAL_NOT_READY`` without a head.

    中文说明：管理本地后端的生命周期与预算。后端不可用时保持 ``LOCAL_NOT_READY``，
    由 Node 端自动降级到正则或云端。
    """

    def __init__(
        self,
        router: LocalRouter | None = None,
        *,
        warmup_timeout_ms: int = DEFAULT_WARMUP_TIMEOUT_MS,
        clock: Callable[[], float] = time.perf_counter,
        unavailable_reason: str | None = None,
    ) -> None:
        self._router = router
        self._clock = clock
        self._warmup_timeout_ms = int(warmup_timeout_ms)
        self._state: RouterState = "unavailable" if router is None else "warming"
        self._error = unavailable_reason if router is None else None
        self._lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._warmup_ms: float | None = None
        self._last_latency_ms: float | None = None
        self._last_code: str | None = None
        self._stats = {"routes": 0, "errors": 0, "busy": 0, "over_budget": 0}

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> RouterState:
        """Lifecycle state exposed to ``router.status``."""
        return self._state

    @property
    def ready(self) -> bool:
        """Whether ``route`` can return a real probability right now."""
        return self._state == "ready" and self._router is not None

    @property
    def error(self) -> str | None:
        """Last warmup or backend error message."""
        return self._error

    def status(self) -> dict[str, object]:
        """Return the ``router.status`` payload (never raises)."""
        router = self._router
        described: dict[str, object] = {}
        describe = getattr(router, "describe", None) if router is not None else None
        if callable(describe):
            try:
                value = describe()
                if isinstance(value, dict):
                    described = value
            except Exception as exc:  # pragma: no cover - defensive
                described = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "ready": self.ready,
            "state": self._state,
            "backend": getattr(router, "backend_id", "off"),
            "model_id": getattr(router, "model_id", None),
            "dimension": int(described.get("dimension") or 0),
            "kind": described.get("kind"),
            "warmup_ms": self._warmup_ms,
            "last_latency_ms": self._last_latency_ms,
            "threshold": described.get("threshold"),
            "uncertainty_band": described.get("uncertainty_band"),
            "seed_version": described.get("seed_version"),
            "error": self._error,
            "last_code": self._last_code,
            "stats": dict(self._stats),
        }

    # ----------------------------------------------------------------- warmup

    def warmup(self, *, wait: bool = False) -> dict[str, object]:
        """Start (or join) warmup without blocking the caller by default.

        Args:
            wait: Run warmup synchronously.  Only tests and CLI tools should
                use this; the bridge always warms up in the background.

        Returns:
            ``{"ok": True, "accepted": bool, "state": ...}``.
        """
        if self._router is None:
            return {
                "ok": True,
                "accepted": False,
                "state": self._state,
                "error": self._error,
            }
        if self._state == "ready":
            return {"ok": True, "accepted": True, "state": self._state}
        if wait:
            self._run_warmup()
            return {
                "ok": self._state == "ready",
                "accepted": True,
                "state": self._state,
                "error": self._error,
            }
        with self._warmup_lock:
            if self._state == "warming" and self._thread and self._thread.is_alive():
                return {"ok": True, "accepted": True, "state": self._state}
            self._state = "warming"
            self._error = None
            thread = threading.Thread(
                target=self._run_warmup,
                name="dsh-dream-router-warmup",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return {"ok": True, "accepted": True, "state": "warming"}

    def _run_warmup(self) -> None:
        router = self._router
        if router is None:
            return
        with self._warmup_lock:
            if self._state == "ready":
                return
            self._state = "warming"
        started = self._clock()
        try:
            router.warmup()
        except RouterBackendError as exc:
            self._state = "failed"
            self._error = f"{exc.code}: {exc.message}"
            return
        except Exception as exc:  # pragma: no cover - defensive
            self._state = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
            return
        self._warmup_ms = (self._clock() - started) * 1000.0
        self._state = "ready"
        self._error = None

    # ------------------------------------------------------------------ route

    def route(self, text: str, deadline_ms: int = DEFAULT_BUDGET_MS) -> dict[str, object]:
        """Classify ``text`` within ``deadline_ms``; never raises.

        Returns either a ``LocalRouteResult.to_dict()`` payload plus
        ``latency_ms`` for the whole call, or ``{"ok": False, "code": ...}``.
        """
        started = self._clock()
        query = str(text or "").strip()
        if not query:
            return self._failure("INVALID_INPUT", started)
        if self._router is None:
            return self._failure(LOCAL_NOT_READY, started, message=self._error)
        if self._state != "ready":
            code = LOCAL_NOT_READY
            self._last_code = code
            return self._failure(code, started, state=self._state)
        budget = max(1, int(deadline_ms or DEFAULT_BUDGET_MS))
        if not self._lock.acquire(blocking=False):
            # The task requires one fail-fast code for "busy or not ready" so the
            # JS layer degrades without blocking. Keep the design's LOCAL_BUSY
            # only as a detail field for observability.
            self._stats["busy"] += 1
            payload = self._failure(
                LOCAL_NOT_READY, started, message="router is busy"
            )
            payload["detail_code"] = LOCAL_BUSY
            payload["busy"] = True
            return payload
        try:
            remaining_ms = budget - (self._clock() - started) * 1000.0
            if remaining_ms <= 0:
                return self._failure(LOCAL_BUDGET_EXCEEDED, started)
            result = self._router.route(query, deadline_ms=int(remaining_ms))
            latency_ms = (self._clock() - started) * 1000.0
            self._last_latency_ms = latency_ms
            self._stats["routes"] += 1
            payload = result.to_dict()
            payload["latency_ms"] = latency_ms
            payload["inference_latency_ms"] = result.latency_ms
            payload["deadline_ms"] = budget
            if latency_ms > budget:
                self._stats["over_budget"] += 1
                payload["over_budget"] = True
            self._last_code = None
            return payload
        except RouterBackendError as exc:
            return self._failure(exc.code, started, message=exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            return self._failure(
                LOCAL_INTERNAL_ERROR, started, message=f"{type(exc).__name__}: {exc}"
            )
        finally:
            self._lock.release()

    def _failure(
        self,
        code: str,
        started: float,
        *,
        message: str | None = None,
        state: str | None = None,
    ) -> dict[str, object]:
        self._stats["errors"] += 1
        self._last_code = code
        payload: dict[str, object] = {
            "ok": False,
            "code": code,
            "latency_ms": (self._clock() - started) * 1000.0,
        }
        if state is not None:
            payload["state"] = state
        if message:
            payload["error"] = message
        return payload


__all__ = [
    "DEFAULT_WARMUP_TIMEOUT_MS",
    "RouterService",
    "RouterState",
]
