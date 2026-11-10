"""Line-oriented Python bridge for DSH-DREAM.

The Node host owns environment discovery, uv bootstrapping, and cache
selection. This process trusts the injected environment:

    DSH_DREAM_RESOLVED_PYTHON
    DSH_DREAM_HF_CACHE / HF_HOME
    DSH_DREAM_CONFIG

Protocols:
    {"op": "warmup", "request_id": "..."}
    {"op": "query", "request_id": "...", "query": "...", "top_k": 3}
    {"op": "route", "request_id": "...", "text": "...", "deadline_ms": 180}
    {"op": "router.status", "request_id": "..."}
    {"op": "router.warmup", "request_id": "..."}
    {"user_msg": "...", "ai_msg": "..."}   # holographic write

The three ``router.*``/``route`` branches MUST stay above the holographic
write fallback: otherwise a router payload would be treated as a turn and
rejected for missing user/assistant text.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_HF_CACHE = Path(
    os.environ.get("DSH_DREAM_HF_CACHE")
    or os.environ.get("HF_HOME")
    or ""
).expanduser()
_CACHE_ERROR = ""
if str(_HF_CACHE) not in {"", "."}:
    try:
        _HF_CACHE.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(_HF_CACHE)
        os.environ["HF_HUB_CACHE"] = str(_HF_CACHE / "hub")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    except OSError as exc:
        _CACHE_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _CACHE_ERROR = "HF_HOME was not injected by the DSH host"

_PLUGIN_IMPORT_ERROR: str | None = None
try:
    from agent_memory.plugin_entry import get_plugin
except Exception as exc:  # keep the bridge alive to explain the failure
    _PLUGIN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    get_plugin = None  # type: ignore[assignment]

_SYSTEM_PREFIX = re.compile(
    r"^\s*(Time sampled|Current runtime context|Environment:|System:)\b",
    re.IGNORECASE,
)
_INJECTED_MARKERS = re.compile(
    r"^\s*<(?:system-reminder|available_skills|skill_content|memory-hint)\b",
    re.IGNORECASE,
)
_ANGLE_TAG_PREFIX = re.compile(r"^\s*<[^>]*>")


def _preview(value: object) -> str:
    text = " ".join(str(value or "").split())
    return text[:180] + ("…" if len(text) > 180 else "")


def _clean_dialogue(value: object) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text or _SYSTEM_PREFIX.match(text):
        return ""
    if _INJECTED_MARKERS.search(text) or _ANGLE_TAG_PREFIX.match(text):
        return ""
    return text


def _cache_label() -> str:
    return str(_HF_CACHE) if str(_HF_CACHE) not in {"", "."} else ""


def _dependency_error(request_id: object) -> dict[str, object]:
    return {
        "request_id": request_id,
        "ok": False,
        "code": "PYTHON_DEPENDENCY_ERROR",
        "error": "Python environment is missing required DSH-DREAM packages.",
        "detail": _PLUGIN_IMPORT_ERROR,
        "python": sys.executable,
        "hf_home": _cache_label(),
        "cache_error": _CACHE_ERROR,
        "hint": (
            "The Node host will provision .dream-venv automatically. "
            "If it fails, set DSH_DREAM_PYTHON to a Python with chromadb and "
            "sentence-transformers installed."
        ),
    }


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if _PLUGIN_IMPORT_ERROR is not None:
        print(f"[Bridge] {_PLUGIN_IMPORT_ERROR}", file=sys.stderr, flush=True)
        for raw in sys.stdin:
            request_id = None
            try:
                payload = json.loads(raw.strip())
                if isinstance(payload, dict):
                    request_id = payload.get("request_id")
            except Exception:
                pass
            _emit(_dependency_error(request_id))
        return 0

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        request_id = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            request_id = payload.get("request_id")

            if payload.get("op") == "warmup":
                get_plugin()
                _emit(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "warmup": True,
                        "python": sys.executable,
                        "hf_home": _cache_label(),
                        "cache_error": _CACHE_ERROR,
                    }
                )
                continue

            if payload.get("op") == "query":
                query = str(payload.get("query") or "").strip()
                if not query:
                    raise ValueError("query must be a non-empty string")
                result = get_plugin().memory_search(
                    query,
                    top_k=int(payload.get("top_k") or 5),
                    timeline_mode=payload.get("timeline_mode") is True,
                )
                _emit({"request_id": request_id, "ok": True, "result": result})
                continue

            # --- RouterEngine (v0.4.0) -------------------------------------
            # These branches deliberately precede the holographic fallback.
            if payload.get("op") == "router.warmup":
                try:
                    result = get_plugin().warmup_router()
                except Exception as exc:
                    _emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "code": "LOCAL_NOT_READY",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                _emit(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "accepted": bool(result.get("accepted")),
                        "state": result.get("state"),
                        "backend": result.get("backend"),
                        "model_id": result.get("model_id"),
                        "warmup_timeout_ms": result.get("warmup_timeout_ms"),
                        "error": result.get("error"),
                    }
                )
                continue

            if payload.get("op") == "router.status":
                try:
                    status = get_plugin().router_status()
                except Exception as exc:
                    _emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "code": "LOCAL_NOT_READY",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                _emit({"request_id": request_id, **status})
                continue

            if payload.get("op") == "route":
                text = str(payload.get("text") or "").strip()
                if not text:
                    _emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "code": "INVALID_INPUT",
                        }
                    )
                    continue
                try:
                    deadline_ms = int(payload.get("deadline_ms") or 180)
                except (TypeError, ValueError):
                    deadline_ms = 180
                deadline_ms = max(1, min(1000, deadline_ms))
                try:
                    result = get_plugin().route_query(text, deadline_ms)
                except Exception as exc:
                    _emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "code": "LOCAL_NOT_READY",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                _emit({"request_id": request_id, **result})
                continue

            user_text = _clean_dialogue(payload.get("user_msg"))
            ai_text = _clean_dialogue(payload.get("ai_msg"))
            if not user_text or not ai_text:
                _emit(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": "empty or system-injected turn rejected",
                    }
                )
                continue
            print(
                f"[Holo] turn user={_preview(user_text)} ai={_preview(ai_text)}",
                file=sys.stderr,
                flush=True,
            )
            result = get_plugin().memory_add_turn(
                user_text,
                ai_text,
                payload.get("extracted_fact"),
            )
            _emit(result)
        except Exception as exc:
            _emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
