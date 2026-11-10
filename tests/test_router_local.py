"""Unit tests for the v0.4.0 local router (P1).

Covers the neutral default (ambiguous -> ``retrieve``), the warmup/ready
lifecycle, the non-blocking lock, the deadline, dimension/SHA validation, and
the ``op=route`` protocol ordering in the Python bridge.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from agent_memory.anchor_manager import AnchorManager
from agent_memory.config import RouterConfig
from agent_memory.plugin_entry import MemoryPlugin
from agent_memory.router.base import (
    LOCAL_BUSY,
    LOCAL_BUDGET_EXCEEDED,
    LOCAL_MODEL_MISMATCH,
    LOCAL_NOT_READY,
    LocalRouteResult,
    RouterBackendError,
    decide,
)
from agent_memory.router.bge_head import BgeHead
from agent_memory.router.service import RouterService
from agent_memory.window_manager import WindowManager

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "dsh_plugin" / "turn_bridge.py"
TEST_TMP = ROOT / ".test-tmp"


@pytest.fixture()
def router_tmp() -> Iterator[Path]:
    """Workspace-local temp directory (the system temp root is read-only here)."""
    TEST_TMP.mkdir(exist_ok=True)
    path = TEST_TMP / f"router-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------
# helpers


def _fake_embedder(dimension: int = 8) -> object:
    """Deterministic embedder: buckets a text by a keyword prefix."""

    def embed(text: str) -> list[float]:
        lowered = str(text)
        vector = np.zeros(dimension, dtype=np.float32)
        if any(cue in lowered for cue in ("上次", "记得", "以前", "previously")):
            vector[0] = 1.0
        elif any(cue in lowered for cue in ("写", "解释", "什么是", "how to")):
            vector[1] = 1.0
        else:
            digest = hashlib.sha256(lowered.encode("utf-8")).digest()
            vector[2:] = [(byte - 127.5) / 127.5 for byte in digest[: dimension - 2]]
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).tolist()
    return embed


def _seeds_file(router_tmp: Path) -> Path:
    payload = {
        "schema_version": 1,
        "version": "test-1",
        "margin": 0.0,
        "temperature": 0.05,
        "positive": ["上次那个蓝色马克杯", "你还记得我的偏好", "以前说过的计划"],
        "negative": ["帮我写一个函数", "解释一下什么是向量", "今天天气不错"],
    }
    path = router_tmp / "seeds.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class _FixedRouter:
    """Deterministic LocalRouter double for service tests."""

    backend_id = "fake"
    model_id = "fake:head:v1"

    def __init__(self, probability: float = 0.9, *, ready: bool = True) -> None:
        self.probability = probability
        self._ready = ready
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def warmup(self) -> None:
        self._ready = True

    def route(self, text: str, deadline_ms: int = 200) -> LocalRouteResult:
        self.calls += 1
        return LocalRouteResult.from_probability(
            probability=self.probability,
            backend=self.backend_id,
            model_id=self.model_id,
            latency_ms=0.1,
        )


class _FakeStore:
    """Minimal memory backend exposing the router embedder surface."""

    embedding_provider = "local"
    embedding_model_name = "fake-bge"
    persist_directory = Path(".test-tmp/router-fake")

    def __init__(self, *, provider: str = "local") -> None:
        self.embedding_provider = provider
        self._embed = _fake_embedder()

    def add_memory(self, content: str, source_type: str, **_: object) -> str:
        return f"mem-{uuid.uuid4().hex}"

    def next_timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def search(self, query: str, **_: object) -> list[object]:
        return []

    def count(self) -> int:
        return 0

    def embed_query_text(self, query: str) -> list[float]:
        return self._embed(query)


def _plugin(store: object | None = None, *, router: RouterConfig | None = None) -> MemoryPlugin:
    return MemoryPlugin(
        anchor_manager=AnchorManager(),
        window_manager=WindowManager(),
        rag_store=store or _FakeStore(),
        rag_top_k=5,
        enable_memory=True,
        router=router or RouterConfig(),
    )


# --------------------------------------------------------------------------
# contracts


def test_neutral_decision_returns_retrieve() -> None:
    # The exact 0.55/0.45 case from the requirement is ambiguous -> retrieve.
    decision, uncertain, _ = decide(0.55)
    assert decision == "retrieve"
    assert uncertain is True
    decision, uncertain, _ = decide(0.45)
    assert decision == "retrieve"
    assert uncertain is True
    # Confident answers still split normally.
    assert decide(0.95)[0] == "retrieve"
    assert decide(0.05)[0] == "skip"
    assert decide(0.95)[1] is False


def test_local_route_result_from_probability_is_neutral() -> None:
    result = LocalRouteResult.from_probability(
        probability=0.5,
        backend="bge-head",
        model_id="fake:head:v1",
        latency_ms=1.0,
    )
    assert result.decision == "retrieve"
    assert result.uncertain is True
    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["decision"] == "retrieve"
    assert payload["threshold"] == 0.55


# --------------------------------------------------------------------------
# bge head


def test_bge_head_prototype_prefers_memory_cues(router_tmp: Path) -> None:
    head = BgeHead(
        embedder=_fake_embedder(),
        seeds_path=_seeds_file(router_tmp),
        embedding_model="fake-bge",
    )
    head.warmup()
    assert head.ready is True
    assert head.kind == "prototype"
    assert head.dimension == 8
    retrieve = head.route("上次那个蓝色马克杯在哪里")
    skip = head.route("帮我写一个二分查找")
    assert retrieve.probability > skip.probability
    assert retrieve.decision == "retrieve"
    assert skip.decision == "skip"
    assert 0.0 <= retrieve.probability <= 1.0


def test_bge_head_requires_seeds(router_tmp: Path) -> None:
    head = BgeHead(
        embedder=_fake_embedder(),
        seeds_path=router_tmp / "missing.json",
    )
    with pytest.raises(RouterBackendError) as error:
        head.warmup()
    assert error.value.code == LOCAL_NOT_READY
    assert head.ready is False
    with pytest.raises(RouterBackendError):
        head.route("上次说的那个")


def test_bge_head_rejects_dimension_mismatch(router_tmp: Path) -> None:
    model_dir = router_tmp / "head"
    model_dir.mkdir()
    np.savez(model_dir / "bge_head.npz", w=np.zeros(3, dtype=np.float32), b=np.float32(0))
    metadata = {
        "schema_version": 1,
        "backend": "bge-head",
        "model_id": "fake:head:v2",
        "embedding_model": "fake-bge",
        "dimension": 3,
        "threshold": 0.55,
        "uncertainty_band": 0.1,
        "sha256": hashlib.sha256((model_dir / "bge_head.npz").read_bytes()).hexdigest(),
    }
    (model_dir / "bge_head.json").write_text(json.dumps(metadata), encoding="utf-8")
    head = BgeHead(
        embedder=_fake_embedder(),
        seeds_path=_seeds_file(router_tmp),
        model_dir=model_dir,
        embedding_model="fake-bge",
    )
    with pytest.raises(RouterBackendError) as error:
        head.warmup()
    assert error.value.code == LOCAL_MODEL_MISMATCH


def test_bge_head_loads_trained_head_with_threshold(router_tmp: Path) -> None:
    model_dir = router_tmp / "head"
    model_dir.mkdir()
    weights = np.zeros(8, dtype=np.float32)
    weights[0] = 1.0
    np.savez(model_dir / "bge_head.npz", w=weights, b=np.float32(0.0))
    metadata = {
        "schema_version": 1,
        "backend": "bge-head",
        "model_id": "fake-bge:head:v9",
        "embedding_model": "fake-bge",
        "dimension": 8,
        "threshold": 0.7,
        "uncertainty_band": 0.05,
        "sha256": hashlib.sha256((model_dir / "bge_head.npz").read_bytes()).hexdigest(),
    }
    (model_dir / "bge_head.json").write_text(json.dumps(metadata), encoding="utf-8")
    head = BgeHead(
        embedder=_fake_embedder(),
        seeds_path=_seeds_file(router_tmp),
        model_dir=model_dir,
        embedding_model="fake-bge",
    )
    head.warmup()
    assert head.kind == "logistic"
    assert head.model_id == "fake-bge:head:v9"
    assert head.threshold == 0.7
    assert head.uncertainty_band == 0.05
    result = head.route("上次那个马克杯")
    assert result.threshold == 0.7
    assert result.decision in {"retrieve", "skip"}


def test_bge_head_rejects_bad_sha(router_tmp: Path) -> None:
    model_dir = router_tmp / "head"
    model_dir.mkdir()
    np.savez(model_dir / "bge_head.npz", w=np.zeros(8, dtype=np.float32), b=np.float32(0))
    metadata = {
        "schema_version": 1,
        "model_id": "fake:head:v1",
        "embedding_model": "fake-bge",
        "dimension": 8,
        "sha256": "0" * 64,
    }
    (model_dir / "bge_head.json").write_text(json.dumps(metadata), encoding="utf-8")
    head = BgeHead(
        embedder=_fake_embedder(),
        seeds_path=_seeds_file(router_tmp),
        model_dir=model_dir,
        embedding_model="fake-bge",
    )
    with pytest.raises(RouterBackendError) as error:
        head.warmup()
    assert error.value.code == LOCAL_MODEL_MISMATCH


# --------------------------------------------------------------------------
# service lifecycle


def test_service_not_ready_returns_code() -> None:
    service = RouterService(_FixedRouter(ready=False), clock=lambda: 0.0)
    payload = service.route("上次说的那个", 180)
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_NOT_READY


def test_service_reports_unavailable_backend_as_not_ready() -> None:
    service = RouterService(None, unavailable_reason="local backend off")
    payload = service.route("上次说的那个", 180)
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_NOT_READY
    assert service.status()["state"] == "unavailable"


def test_service_busy_returns_immediately() -> None:
    service = RouterService(_FixedRouter())
    service.warmup(wait=True)
    assert service._lock.acquire(blocking=False)  # simulate memory_search in flight
    try:
        payload = service.route("上次说的那个", 180)
    finally:
        service._lock.release()
    # The task contract: busy still answers LOCAL_NOT_READY immediately so the
    # JS side degrades; LOCAL_BUSY stays available as a detail code.
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_NOT_READY
    assert payload["detail_code"] == LOCAL_BUSY
    assert payload["busy"] is True


def test_service_deadline_returns_budget_exceeded() -> None:
    service = RouterService(_FixedRouter(), clock=lambda: 0.0)
    service.warmup(wait=True)
    ticks = iter([100.0, 300.0])
    service._clock = lambda: next(ticks, 300.0)  # type: ignore[assignment]
    payload = service.route("上次说的那个", 180)
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_BUDGET_EXCEEDED


def test_service_warmup_and_uncertain_retrieve() -> None:
    service = RouterService(_FixedRouter(probability=0.52))
    assert service.status()["ready"] is False
    outcome = service.warmup(wait=True)
    assert outcome["state"] == "ready"
    payload = service.route("上次说的那个", 180)
    assert payload["ok"] is True
    assert payload["decision"] == "retrieve"
    assert payload["uncertain"] is True
    assert payload["latency_ms"] >= 0


def test_service_failed_warmup_stays_unavailable() -> None:
    class _Broken(_FixedRouter):
        def warmup(self) -> None:
            raise RouterBackendError(LOCAL_NOT_READY, "seeds missing")

    service = RouterService(_Broken())
    outcome = service.warmup(wait=True)
    assert outcome["state"] == "failed"
    payload = service.route("上次说的那个", 180)
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_NOT_READY


# --------------------------------------------------------------------------
# plugin entry integration


def _wait_ready(plugin: MemoryPlugin, timeout: float = 5.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = plugin.router_status()
        if status["ready"]:
            return status
        time.sleep(0.02)
    return plugin.router_status()


def test_router_config_nested_overrides_manifest_defaults(router_tmp: Path) -> None:
    from agent_memory.config import load_config

    config_path = router_tmp / "agent_memory_config.json"
    config_path.write_text(
        json.dumps(
            {
                "enable_memory": True,
                "rag_path": str(router_tmp / "chroma"),
                "router": {
                    "enabled": False,
                    "threshold": 0.7,
                    "timeout_policy": "bypass_main",
                    "local": {"backend": "off"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_config(config_path)
    assert loaded.router.enabled is False
    assert loaded.router.threshold == 0.7
    assert loaded.router.timeout_policy == "bypass_main"
    assert loaded.router.local_backend == "off"
    # Defaults still apply where the bridge file says nothing.
    assert loaded.router.uncertainty_band == 0.1


def test_plugin_route_requires_warmup_then_answers(router_tmp: Path) -> None:
    plugin = _plugin(_FakeStore())
    first = plugin.route_query("上次说的那个蓝色马克杯", 180)
    assert first["ok"] is False
    assert first["code"] == LOCAL_NOT_READY
    plugin.warmup_router()
    status = _wait_ready(plugin)
    assert status["ready"] is True
    assert status["backend"] == "bge-head"
    assert status["dimension"] == 8
    result = plugin.route_query("上次说的那个蓝色马克杯", 180)
    assert result["ok"] is True
    assert result["decision"] in {"retrieve", "skip"}
    assert 0.0 <= result["probability"] <= 1.0


def test_plugin_rejects_cloud_embedding_provider() -> None:
    plugin = _plugin(_FakeStore(provider="openai_compatible"))
    status = plugin.router_status()
    assert status["ready"] is False
    payload = plugin.route_query("上次说的那个", 180)
    assert payload["ok"] is False
    assert payload["code"] == LOCAL_NOT_READY


# --------------------------------------------------------------------------
# bridge protocol


def _write_stub_package(base: Path) -> Path:
    package = base / "agent_memory"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "plugin_entry.py").write_text(
        "\n".join(
            [
                "class _Plugin:",
                "    def route_query(self, text, deadline_ms=180):",
                "        return {'ok': True, 'decision': 'retrieve', 'probability': 0.9,",
                "                'confidence': 0.9, 'uncertain': False, 'threshold': 0.55,",
                "                'uncertainty_band': 0.1, 'latency_ms': 1.0,",
                "                'backend': 'bge-head', 'model_id': 'stub:head:v1',",
                "                'deadline_ms': deadline_ms, 'text_len': len(text)}",
                "    def router_status(self):",
                "        return {'ok': True, 'ready': True, 'backend': 'bge-head',",
                "                'model_id': 'stub:head:v1', 'state': 'ready'}",
                "    def warmup_router(self):",
                "        return {'ok': True, 'accepted': True, 'state': 'warming',",
                "                'backend': 'bge-head', 'model_id': 'stub:head:v1'}",
                "_plugin = _Plugin()",
                "",
                "def get_plugin():",
                "    return _plugin",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return base


def _run_bridge(lines: list[dict], path_prefix: Path) -> list[dict]:
    TEST_TMP.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(path_prefix) + os.pathsep + str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["DSH_DREAM_HF_CACHE"] = str(TEST_TMP / "router-bridge-hf")
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
    proc = subprocess.run(
        [sys.executable, "-u", str(BRIDGE)],
        cwd=str(ROOT),
        env=env,
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    responses = [
        json.loads(line)
        for line in proc.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    assert responses, (proc.stdout, proc.stderr)
    return responses


def test_bridge_route_ops_precede_holographic_fallback(router_tmp: Path) -> None:
    stub_root = _write_stub_package(router_tmp / "stub")
    responses = _run_bridge(
        [
            {"op": "route", "request_id": "r1", "text": "上次说的那个", "deadline_ms": 120},
            {"op": "router.status", "request_id": "r2"},
            {"op": "router.warmup", "request_id": "r3"},
            {"op": "route", "request_id": "r4", "text": "", "deadline_ms": 120},
            {"user_msg": "今天天气不错", "ai_msg": "适合散步", "request_id": "r5"},
        ],
        stub_root,
    )
    by_id = {item.get("request_id"): item for item in responses}
    route = by_id["r1"]
    assert route["decision"] == "retrieve"
    assert route["deadline_ms"] == 120
    assert route["text_len"] == len("上次说的那个")
    assert "system-injected turn rejected" not in json.dumps(by_id["r2"], ensure_ascii=False)
    assert by_id["r2"]["ready"] is True
    assert by_id["r3"]["accepted"] is True
    assert by_id["r4"] == {"request_id": "r4", "ok": False, "code": "INVALID_INPUT"}
    # The holographic write still falls through to the stub plugin (missing
    # attributes surface as an error), proving it is not swallowed by the
    # router branches.
    assert by_id["r5"]["ok"] is False

