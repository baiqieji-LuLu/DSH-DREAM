"""Line-oriented host bridge for forced holographic turn writes.

DSH Host starts this helper once and sends one JSON object per completed turn:

    {"user_msg": "...", "ai_msg": "..."}

The helper uses the same configured RAGStore/MemoryPlugin as the tool surface.
It intentionally never prints or reads the embedding API key.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_CHECKOUT_ROOT = Path(r"G:\deepseek_ragproject_workspace")


def _project_root() -> Path:
    """Locate the Python project root WITHOUT trusting ``__file__``.

    DSH loads this plugin through a ``link:`` install that lives outside the
    project (``G:\\dsh_plugin_test_workspace\\dsh_plugin``), so
    ``Path(__file__).parents[1]`` resolves to a directory with no ``.hf-cache``
    and every offline model load fails with "cannot initialize any local
    embedding model". Anchor instead on the directory holding the config the
    Host passes (it writes that file into the project root), falling back to
    the checkout itself.
    """
    configured = os.environ.get("AGENT_MEMORY_CONFIG", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve().parent
        if (candidate / ".hf-cache").is_dir():
            return candidate
    return _CHECKOUT_ROOT


_PROJECT_ROOT = _project_root()
# Hard-wired on purpose: the project cache is the only place the offline
# sentence-transformers models are allowed to come from. Never inherit HF_HOME
# from the Host process, which may point at a global cache that lacks them.
os.environ["HF_HOME"] = str(_PROJECT_ROOT / ".hf-cache")
os.environ["HF_HUB_CACHE"] = str(_PROJECT_ROOT / ".hf-cache" / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from agent_memory.plugin_entry import get_plugin

_SYSTEM_PREFIX = re.compile(
    r"^\s*(Time sampled|Current runtime context|Environment:|System:)\b",
    re.IGNORECASE,
)
_INJECTED_MARKERS = re.compile(
    r"^\s*<(?:system-reminder|available_skills|skill_content)\b",
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
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("turn payload must be an object")
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
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
