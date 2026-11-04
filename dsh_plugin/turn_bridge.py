"""Line-oriented host bridge for forced holographic turn writes.

DSH Host starts this helper once and sends one JSON object per completed turn:

    {"user_msg": "...", "ai_msg": "..."}

The helper uses the same configured RAGStore/MemoryPlugin as the tool surface.
It intentionally never prints or reads the embedding API key.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    """Return the installed plugin root (the parent of ``dsh_plugin``)."""
    return Path(__file__).resolve().parent.parent


_PROJECT_ROOT = _project_root()


def _python_candidates():
    explicit = os.environ.get("DSH_DREAM_PYTHON", "").strip()
    if explicit:
        yield Path(explicit).expanduser()
    roots = [
        _PROJECT_ROOT,
        Path.cwd(),
        *Path.cwd().parents[:2],
    ]
    for root in dict.fromkeys(roots):
        candidates = (
            [root / ".venv" / "Scripts" / "python.exe"]
            if os.name == "nt"
            else [root / ".venv" / "bin" / "python"]
        )
        yield from candidates


def _runtime_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("agent_memory", "chromadb", "sentence_transformers")
    )


def _maybe_reexec_with_venv() -> None:
    """Rescue a global-Python launch by re-execing the adjacent venv."""
    if os.environ.get("DSH_DREAM_REEXEC") == "1" or _runtime_available():
        return
    current = Path(sys.executable).resolve()
    for candidate in _python_candidates():
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == current or not resolved.is_file():
            continue
        os.environ["DSH_DREAM_REEXEC"] = "1"
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.execv(str(resolved), [str(resolved), "-u", str(Path(__file__).resolve())])
    # No usable venv found: let the original import surface the real error.


_maybe_reexec_with_venv()


# Prefer the project-local cache when present; otherwise leave the host's
# HF_HOME untouched so npm-installed plugins can use a global model cache.
_LOCAL_HF_CACHE = _PROJECT_ROOT / ".hf-cache"
if _LOCAL_HF_CACHE.is_dir():
    os.environ["HF_HOME"] = str(_LOCAL_HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(_LOCAL_HF_CACHE / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from agent_memory.plugin_entry import get_plugin

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
    """Reject empty or DSH-injected environment/skill text at the boundary."""
    text = value.strip() if isinstance(value, str) else ""
    if not text or _SYSTEM_PREFIX.match(text):
        return ""
    if _INJECTED_MARKERS.search(text) or _ANGLE_TAG_PREFIX.match(text):
        return ""
    return text


def main() -> int:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        request_id = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("turn payload must be an object")
            request_id = payload.get("request_id")
            if payload.get("op") == "query":
                query = str(payload.get("query") or "").strip()
                if not query:
                    raise ValueError("query must be a non-empty string")
                top_k = int(payload.get("top_k") or 5)
                timeline_mode = payload.get("timeline_mode") is True
                result = get_plugin().memory_search(
                    query,
                    top_k=top_k,
                    timeline_mode=timeline_mode,
                )
                print(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "ok": True,
                            "result": result,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            print(
                f"[Holo] 收到 turn user={_preview(payload.get('user_msg'))} "
                f"assistant={_preview(payload.get('ai_msg'))}",
                file=sys.stderr,
                flush=True,
            )
            user_text = _clean_dialogue(payload.get("user_msg"))
            ai_text = _clean_dialogue(payload.get("ai_msg"))
            if not user_text or not ai_text:
                print(
                    "[Holo] 被过滤逻辑拦截: user/assistant 为空或含系统注入特征",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": "empty or system-injected turn rejected",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            print("[Holo] 准备写入 RAG...", file=sys.stderr, flush=True)
            result = get_plugin().memory_add_turn(
                user_text,
                ai_text,
                payload.get("extracted_fact"),
            )
            print(
                f"[Holo] 写入结果 ok={result.get('ok')} "
                f"timestamp={result.get('timestamp_ms')}",
                file=sys.stderr,
                flush=True,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:  # keep the bridge alive after a bad turn
            print(
                json.dumps(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
