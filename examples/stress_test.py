# ruff: noqa: E402, RUF001
"""1000-round real-model stress test for the DSH-DREAM stack.

The test creates all mutable state inside a guarded temporary directory:

* anchors.json
* audit.db
* chroma/ (only when a real Chroma backend is selected)

It writes 1000 daily-chatter memories, hides one critical ``system_fact`` at
round 50, verifies the sliding window has discarded that old turn, and queries
at round 1001 to confirm retrieval and prompt injection still work.

The default backend is a deterministic, in-process Chroma-compatible mock when
``chromadb`` and ``sentence-transformers`` are not both installed. No model
download is required for the default run.

After cleanup, the script writes ``test_result3.md`` to the project root.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_memory._tokens import estimate_tokens
from agent_memory.anchor_manager import AnchorManager
from agent_memory.audit_logger import AuditLogger
from agent_memory.plugin_entry import MemoryPlugin
from agent_memory.rag_store import RAGStore
from agent_memory.window_manager import WindowManager

TOTAL_ROUNDS = 1_000
SECRET_ROUND = 50
QUERY_ROUND = TOTAL_ROUNDS + 1
WINDOW_BUDGET = 80
RAG_TOP_K = 3
SECRET_TEXT = "我的银行卡密码是 888888，绝对不能忘记"
QUERY_TEXT = "请调用 memory_search 帮我查一下我的银行卡密码是多少？"
WINDOW_MARKER = f"WINDOW-MARKER-{SECRET_ROUND:04d}"
REPORT_PATH = PROJECT_ROOT / "test_result3.md"

CHATTER = (
    "今天天气不错，适合散步。",
    "刚喝完一杯热茶，味道还可以。",
    "路上看到一只小猫，挺可爱的。",
    "下午整理了一下书架。",
    "晚饭吃了番茄鸡蛋面。",
    "今天的地铁稍微有点挤。",
    "窗外的阳光很好。",
    "听了一首老歌。",
    "把桌子收拾干净了。",
    "晚上准备早点休息。",
)

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap(query_tokens: set[str], document_tokens: set[str]) -> int:
    """Count exact or long-prefix matches for the deterministic mock backend."""
    matches = 0
    for query_token in query_tokens:
        if any(
            query_token == document_token
            or (
                len(query_token) >= 4
                and len(document_token) >= 4
                and (
                    document_token.startswith(query_token)
                    or query_token.startswith(document_token)
                )
            )
            for document_token in document_tokens
        ):
            matches += 1
    return matches


class MockCollection:
    """Chroma-compatible collection with lexical ranking for offline testing."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict[str, Any]]] = {}

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for memory_id, document, metadata in zip(
            ids, documents, metadatas, strict=True
        ):
            self.records[memory_id] = (document, metadata)

    def count(self) -> int:
        return len(self.records)

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        query_tokens = _tokens(query_texts[0])
        ranked: list[tuple[float, str, str, dict[str, Any]]] = []
        for memory_id, (document, metadata) in self.records.items():
            if where is not None and not self._matches(metadata, where):
                continue
            overlap = _overlap(query_tokens, _tokens(document))
            score = overlap / max(1, len(query_tokens))
            if score > 0:
                ranked.append((score, memory_id, document, metadata))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        rows = ranked[:n_results]
        return {
            "ids": [[row[1] for row in rows]],
            "documents": [[row[2] for row in rows]],
            "metadatas": [[row[3] for row in rows]],
            "distances": [[1.0 - row[0] for row in rows]],
        }

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, list[Any]]:
        del include
        memory_id = ids[0]
        if memory_id not in self.records:
            return {"ids": [], "documents": [], "metadatas": []}
        document, metadata = self.records[memory_id]
        return {
            "ids": [memory_id],
            "documents": [document],
            "metadatas": [metadata],
        }

    def delete(self, *, ids: list[str]) -> None:
        for memory_id in ids:
            self.records.pop(memory_id, None)

    @staticmethod
    def _matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        spec = where.get("source_type")
        actual = metadata.get("source_type")
        if isinstance(spec, str):
            return actual == spec
        if isinstance(spec, dict):
            return actual in spec.get("$in", [])
        return True


class MockClient:
    """Chroma-compatible client wrapper with no disk persistence."""

    def __init__(self) -> None:
        self.collection = MockCollection()

    def get_or_create_collection(self, **kwargs: Any) -> MockCollection:
        del kwargs
        return self.collection


def _chatter_for(round_no: int) -> str:
    prefix = f"{WINDOW_MARKER} " if round_no == SECRET_ROUND else ""
    topic = CHATTER[(round_no - 1) % len(CHATTER)]
    return f"第 {round_no} 轮：{prefix}{topic}（闲聊编号 {round_no:04d}）"


def _backend_available() -> bool:
    return bool(
        importlib.util.find_spec("chromadb")
        and importlib.util.find_spec("sentence_transformers")
    )


@contextmanager
def _workspace_temp_dir() -> Iterator[Path]:
    """Create a workspace-local temp directory and always remove it.

    The DSH Windows sandbox denies writes to system temp, while Chroma's Rust
    binding cannot open a SQLite database in a sandbox-created 0700 directory.
    A normal inherited-ACL directory inside the project is both writable and
    removable, and ``shutil.rmtree`` in the finally block guarantees cleanup.
    """
    path = PROJECT_ROOT / f".stress-tmp-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _make_store(backend: str, temp_dir: Path) -> tuple[RAGStore, str]:
    chroma_dir = temp_dir / "chroma"
    if backend == "chroma":
        print("---Start Init---", flush=True)
        return (
            RAGStore(
                persist_directory=chroma_dir,
                model_name="all-MiniLM-L6-v2",
            ),
            "real chromadb + sentence-transformers all-MiniLM-L6-v2",
        )
    return (
        RAGStore(
            persist_directory=chroma_dir,
            client=MockClient(),
        ),
        "deterministic mock Chroma",
    )


def _report(
    *,
    backend: str,
    started_at: datetime,
    elapsed_seconds: float,
    history_count: int,
    window_count: int,
    window_budget: int,
    marker_in_window: bool,
    secret_in_window: bool,
    ready: bool,
    rag_count: int,
    retrieval_seconds: float,
    hit_id: str,
    hit_text: str,
    hit_source: str,
    retrieved_ids: list[str],
    prompt_tokens: int,
    prompt_chars: int,
    hint_injected: bool,
    hint_after_query: bool,
    latest_turn_preserved: bool,
    audit_snapshot_id: int | None,
    audit_count: int,
    temp_path: Path,
    temp_cleaned: bool,
    root_leftovers: list[str],
    checks: dict[str, bool],
    events: list[str],
) -> str:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    overall = "PASS" if all(checks.values()) else "FAIL"
    leftover_text = "无" if not root_leftovers else ", ".join(root_leftovers)
    check_rows = "\n".join(
        f"| {name} | {'PASS' if value else 'FAIL'} |"
        for name, value in checks.items()
    )
    event_rows = "\n".join(f"- {event}" for event in events)

    return f"""# DSH-DREAM {TOTAL_ROUNDS} 轮压力测试报告

- 生成时间：{generated_at}
- 开始时间：{started_at.isoformat(timespec="seconds")}
- 测试规模：{TOTAL_ROUNDS} 轮对话 / 第 {QUERY_ROUND} 轮查询
- 向量后端：{backend}
- 总体结论：**{overall}**

## 1. 执行时间与耗时

| 项目 | 值 |
| --- | --- |
| 总耗时 | {elapsed_seconds:.3f} 秒 |
| 第 {QUERY_ROUND} 轮检索耗时 | {retrieval_seconds * 1000:.3f} ms |
| 写入 RAG 的记忆数量 | {rag_count} |
| 初始对话历史 | {history_count} 条 |
| 最终窗口 | {window_count} 条 |

## 2. 窗口截断验证

| 检查项 | 结果 |
| --- | --- |
| 初始历史 | {history_count} 条（{TOTAL_ROUNDS} 轮 × 2 条消息） |
| 窗口 Token 预算 | {window_budget} |
| 最终窗口保留 | {window_count} 条 |
| 第 {SECRET_ROUND} 轮标记 `{WINDOW_MARKER}` 在窗口内 | **{marker_in_window}** |
| 第 {SECRET_ROUND} 轮敏感事实在窗口内 | **{secret_in_window}** |
| 第 {SECRET_ROUND} 轮记忆是否在窗口内（应为 False） | **{marker_in_window}** |
| 最新一轮（第 {TOTAL_ROUNDS} 轮）是否保留 | **{latest_turn_preserved}** |

## 3. 极限召回验证

| 项目 | 值 |
| --- | --- |
| 查询轮次 | 第 {QUERY_ROUND} 轮 |
| 查询文本 | `{QUERY_TEXT}` |
| 检索耗时 | {retrieval_seconds * 1000:.3f} ms |
| 是否成功召回 `888888` | **{ready}** |
| 命中记忆 ID | `{hit_id}` |
| 命中来源 | `{hit_source}` |
| 检索结果 ID 列表 | `{retrieved_ids}` |

召回的完整文本：

> {hit_text}

Prompt 注入验证：

| 检查项 | 结果 |
| --- | --- |
| 敏感事实进入最终 Prompt | **{hint_injected}** |
| RAG Hint 位于用户查询之后 | **{hint_after_query}** |
| 审计快照 ID | `{audit_snapshot_id}` |
| 审计记录数量 | {audit_count} |

## 4. 最终 Prompt 体积

| 项目 | 值 |
| --- | --- |
| 最终 Prompt Token 估算 | **{prompt_tokens}** |
| 最终 Prompt 字符数 | {prompt_chars} |
| 窗口 Token 预算 | {window_budget} |
| 是否保持可控（≤ 1000 tokens） | **{prompt_tokens <= 1000}** |

即使已经历 {TOTAL_ROUNDS} 轮对话，最终 Prompt 仍然只包含窗口内的近期消息和少量召回结果。

## 5. 清理确认

| 检查项 | 结果 |
| --- | --- |
| 临时目录 | `{temp_path}` |
| 临时目录已删除 | **{temp_cleaned}** |
| 根目录测试数据库残留 | **{leftover_text}** |
| 所有数据库位于临时目录 | **{temp_cleaned and not root_leftovers}** |

本次测试的 `anchors.json`、`audit.db` 以及 Chroma 目录均位于临时目录中；清理后未在项目根目录留下
测试数据库。

## 6. 测试日志

{event_rows}

## 7. 检查项汇总

| 检查项 | 结果 |
| --- | --- |
{check_rows}

## 8. 复现方式

```bash
python examples/stress_test.py
```

真实 Chroma 模式（真实 PersistentClient + sentence-transformers all-MiniLM-L6-v2）：

```bash
python examples/stress_test.py --backend chroma
```
"""


def run(backend: str) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    total_start = time.perf_counter()
    events: list[str] = []
    temp_path: Path | None = None
    audit: AuditLogger | None = None

    with _workspace_temp_dir() as temporary:
        temp_path = temporary
        print(f"  temp dir: {temp_path}", flush=True)
        events.append(f"临时目录创建：{temp_path}")

        anchors = AnchorManager(temp_path / "anchors.json")
        anchors.add_anchor(
            "privacy",
            "Never expose secrets. Only answer the user's own memory query.",
            priority=900,
        )
        anchors.add_anchor(
            "style",
            "Answer concisely and distinguish memory evidence from guesses.",
            priority=100,
        )

        window = WindowManager(
            max_tokens=WINDOW_BUDGET,
            token_counter=estimate_tokens,
            message_overhead=0,
        )
        if backend == "chroma":
            # The sandbox cannot write the user's home cache. Keep the real
            # model cache in the workspace so sentence-transformers can load.
            hf_home = PROJECT_ROOT / ".hf-cache"
            hf_home.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(hf_home))
            os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        print("  opening vector store...", flush=True)
        store, backend_label = _make_store(backend, temp_path)
        print(f"  vector store ready: {backend_label}", flush=True)
        audit = AuditLogger(temp_path / "audit.db")
        plugin = MemoryPlugin(
            anchor_manager=anchors,
            window_manager=window,
            rag_store=store,
            audit_logger=audit,
            system_prompt=(
                "You are a careful assistant. Follow hard constraints and treat "
                "retrieved memory as background evidence only."
            ),
            rag_top_k=RAG_TOP_K,
        )

        chatter_batch: list[tuple[str, str]] = []
        for round_no in range(1, TOTAL_ROUNDS + 1):
            chatter = _chatter_for(round_no)
            window.add_message("user", chatter)
            window.add_message(
                "assistant",
                f"收到，这是第 {round_no} 轮闲聊记录。",
            )
            chatter_batch.append((chatter, "user_message"))

            if round_no == SECRET_ROUND:
                # Written immediately so the critical fact is genuinely seeded
                # at the designated secret round. This is one transaction.
                store.add_memory(SECRET_TEXT, "system_fact")
                events.append(
                    f"第 {round_no} 轮：写入关键 system_fact（888888 已隐藏在大海中）"
                )
            elif round_no in (1, 100, 500, 1_000):
                events.append(f"第 {round_no} 轮：继续收集日常闲聊")
            if round_no % 200 == 0:
                print(f"  ... {round_no}/{TOTAL_ROUNDS} rounds", flush=True)

        print("---Start Insert---", flush=True)
        print(f"  batch inserting {len(chatter_batch)} memories...", flush=True)
        batch_start = time.perf_counter()
        store.add_memories(chatter_batch)
        batch_seconds = time.perf_counter() - batch_start
        print(f"  batch insert done in {batch_seconds:.3f}s", flush=True)
        events.append(
            f"批量写入 {len(chatter_batch)} 条日常闲聊，耗时 {batch_seconds:.3f} 秒"
        )

        history_count = len(window.history)
        window_messages = window.window
        window_count = len(window_messages)
        window_text = "\n".join(message.content for message in window_messages)
        marker_in_window = WINDOW_MARKER in window_text
        secret_in_window = SECRET_TEXT in window_text
        latest_turn_preserved = any(
            f"闲聊编号 {TOTAL_ROUNDS:04d}" in message.content
            for message in window_messages
        )

        retrieval_start = time.perf_counter()
        search_result = plugin.memory_search(QUERY_TEXT)
        retrieval_seconds = time.perf_counter() - retrieval_start

        results = search_result["results"]
        hit = results[0] if results else None
        hit_id = hit["memory_id"] if hit is not None else "(none)"
        hit_text = hit["content"] if hit is not None else "(no retrieval result)"
        hit_source = hit["source_type"] if hit is not None else "(none)"
        ready = hit is not None and "888888" in hit["content"]
        retrieved_ids = [str(item["memory_id"]) for item in results]
        rag_count = store.count()

        context = plugin.get_rendered_context(QUERY_TEXT)
        prompt = context["prompt"]
        prompt_tokens = estimate_tokens(prompt)
        prompt_chars = len(prompt)
        hint_injected = "888888" in prompt
        label_index = prompt.find("[仅供背景参考]")
        query_end_index = prompt.find("<|user_query_end:")
        hint_after_query = (
            label_index != -1
            and query_end_index != -1
            and label_index > query_end_index
        )
        audit_snapshot_id = context["audit_snapshot_id"]
        audit_count = audit.count_snapshots()
        audit.close()
        audit = None

        events.append(f"第 {QUERY_ROUND} 轮：检索完成，耗时 {retrieval_seconds * 1000:.3f} ms")
        events.append(f"第 {QUERY_ROUND} 轮：Prompt 注入完成，Token 数 {prompt_tokens}")
        events.append(f"审计快照写入：{audit_snapshot_id}")
        events.append("释放内存组件，准备清理临时目录")

        # Release Chroma's SQLite/HNSW handles before Windows removes the
        # temporary persistence directory, then drop the remaining object graph.
        store.close()
        del plugin
        del store
        gc.collect()

    assert temp_path is not None
    temp_cleaned = not temp_path.exists()
    if not temp_cleaned:
        shutil.rmtree(temp_path, ignore_errors=True)
        temp_cleaned = not temp_path.exists()

    root_candidates = [
        PROJECT_ROOT / "audit.db",
        PROJECT_ROOT / "audit.db-wal",
        PROJECT_ROOT / "audit.db-shm",
        PROJECT_ROOT / "anchors.json",
        PROJECT_ROOT / ".chroma",
        PROJECT_ROOT / "chroma",
    ]
    root_leftovers = [str(path) for path in root_candidates if path.exists()]
    root_leftovers.extend(str(path) for path in PROJECT_ROOT.glob(".stress-tmp-*"))
    elapsed_seconds = time.perf_counter() - total_start

    checks = {
        f"{TOTAL_ROUNDS} 轮共 {TOTAL_ROUNDS * 2} 条历史": (
            history_count == TOTAL_ROUNDS * 2
        ),
        "窗口只保留近期消息": 0 < window_count < history_count,
        f"第 {SECRET_ROUND} 轮标记不在窗口": not marker_in_window,
        f"第 {SECRET_ROUND} 轮事实不在窗口": not secret_in_window,
        "最新一轮被保留": latest_turn_preserved,
        f"RAG 中包含 {TOTAL_ROUNDS + 1} 条记忆": rag_count == TOTAL_ROUNDS + 1,
        "成功召回 888888": ready,
        "敏感事实注入 Prompt": hint_injected,
        "RAG Hint 位于查询之后": hint_after_query,
        "Prompt 体积可控": prompt_tokens <= 1_000,
        "临时目录已清理": temp_cleaned,
        "根目录无数据库残留": not root_leftovers,
    }

    return {
        "backend": backend_label,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "history_count": history_count,
        "window_count": window_count,
        "window_budget": WINDOW_BUDGET,
        "marker_in_window": marker_in_window,
        "secret_in_window": secret_in_window,
        "ready": ready,
        "rag_count": rag_count,
        "retrieval_seconds": retrieval_seconds,
        "hit_id": hit_id,
        "hit_text": hit_text,
        "hit_source": hit_source,
        "retrieved_ids": retrieved_ids,
        "prompt_tokens": prompt_tokens,
        "prompt_chars": prompt_chars,
        "hint_injected": hint_injected,
        "hint_after_query": hint_after_query,
        "latest_turn_preserved": latest_turn_preserved,
        "audit_snapshot_id": audit_snapshot_id,
        "audit_count": audit_count,
        "temp_path": temp_path,
        "temp_cleaned": temp_cleaned,
        "root_leftovers": root_leftovers,
        "checks": checks,
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"DSH-DREAM {TOTAL_ROUNDS}-round stress test"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "mock", "chroma"),
        default="auto",
        help="Vector backend. auto uses chroma when full dependencies exist.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.backend == "auto":
        backend = "chroma" if _backend_available() else "mock"
    else:
        backend = args.backend
    if backend == "chroma" and not _backend_available():
        print(
            "错误：--backend chroma 需要安装 chromadb 与 sentence-transformers。",
            file=sys.stderr,
        )
        return 2

    print("=" * 72)
    print(f"DSH-DREAM {TOTAL_ROUNDS}-Round Stress Test")
    print(f"backend: {backend}")
    print("=" * 72)

    result = run(backend)
    report = _report(**result)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n--- RESULT ---")
    print(f"history={result['history_count']} window={result['window_count']}")
    print(f"marker_in_window={result['marker_in_window']}")
    print(f"secret_in_window={result['secret_in_window']}")
    print(f"retrieval={result['retrieval_seconds'] * 1000:.3f} ms")
    print(f"recalled={result['ready']} injected={result['hint_injected']}")
    print(f"prompt_tokens={result['prompt_tokens']}")
    print(f"temp_cleaned={result['temp_cleaned']} leftovers={result['root_leftovers']}")
    print(f"report={REPORT_PATH}")

    passed = all(result["checks"].values())
    print(f"overall={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
