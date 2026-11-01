"""One-call D-group timeline rerun after the state-slot/timeline fix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CST,
    EXP,
    MODEL,
    PRICE_CACHE_MISS,
    PRICE_OUTPUT,
    USD_CNY,
    count_tokens,
    cost_usd,
    is_peak,
    load_api_key,
    load_tokenizer,
    make_client,
    make_nonce,
    now_cst,
    usage_to_dict,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_memory.rag_store import RAGStore  # noqa: E402
from run_trivia_experiment import build_records  # noqa: E402

TRIVIA_DIR = EXP / "d-timeline-flat"
RUN_ID = "d-timeline-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID
PRIOR = Decimal("1.590302418")
CAP = Decimal("3.50")
QUERY = "我那个星巴克蓝色马克杯现在怎么样了？请告诉我它的完整历史。"


def main() -> int:
    load_api_key()
    if is_peak(now_cst()):
        raise SystemExit("STOP: peak hour")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if TRIVIA_DIR.exists():
        shutil.rmtree(TRIVIA_DIR, ignore_errors=True)
    records = build_records()
    store = RAGStore(
        persist_directory=TRIVIA_DIR,
        collection_name="trivia_100",
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
        flat_max_records=200_000,
    )
    texts = [r["content"] for r in records]
    store._collection.add(
        ids=[r["id"] for r in records],
        documents=texts,
        metadatas=[
            {
                "source_type": "user_message",
                "created_at": now_cst().isoformat(timespec="seconds"),
                "timestamp_ms": r["timestamp_ms"],
                "content_sha256": hashlib.sha256(r["content"].encode()).hexdigest(),
            }
            for r in records
        ],
        embeddings=store._embedding_function(texts),
    )
    hits = store.search_memory("星巴克 蓝色马克杯", top_k=3, timeline_mode=True)
    store.close()
    print("timeline_hits", [(h.memory_id, h.content) for h in hits])
    if [h.memory_id for h in hits] != ["trivia-010", "trivia-050", "trivia-090"]:
        raise SystemExit("STOP: timeline gate failed")

    context = "\n".join(
        f"[{i}] {h.content}" for i, h in enumerate(hits, start=1)
    )
    system = (
        "你是一个严谨的个人助理。只依据提供的记忆原文回答。"
        "材料中没有的信息，明确说不知道，不要编造。"
    )
    nonce = make_nonce("d-timeline")
    prompt = f"""{system}

实验批次：{nonce}（仅用于缓存隔离，不影响任务）

以下是按时间从早到晚排列的记忆原文：
<memory>
{context}
</memory>

问题：{QUERY}
请按时间顺序说明完整历史，不要只给最后状态。"""
    tokenizer = load_tokenizer()
    prompt_tokens = count_tokens(prompt, tokenizer)
    worst = (
        Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
        + Decimal(256) / Decimal(1_000_000) * PRICE_OUTPUT
    )
    if PRIOR + worst > CAP:
        raise SystemExit("STOP: budget cap")
    client = make_client()
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = usage_to_dict(response.usage)
    answer = (response.choices[0].message.content or "").strip()
    cost = cost_usd(usage)
    total = PRIOR + cost
    record = {
        "label": "d-timeline-query",
        "query": QUERY,
        "timeline_hits": [
            {
                "id": h.memory_id,
                "content": h.content,
                "timestamp_ms": h.metadata.get("timestamp_ms"),
            }
            for h in hits
        ],
        "answer": answer,
        "usage": usage,
        "cost_usd": str(cost),
        "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
        "latency_ms": round(latency_ms, 4),
        "finish_reason": response.choices[0].finish_reason,
        "requested_at_cst": now_cst().isoformat(timespec="seconds"),
        "prompt_tokens_est": prompt_tokens,
        "prompt": prompt,
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "response": response.model_dump() if hasattr(response, "model_dump") else {},
        "prior_spent_usd": str(PRIOR),
        "total_spent_usd": str(total),
    }
    write_json(RUN_DIR / "summary.json", record)
    write_json(EXP / "d_timeline_api.json", record)
    # Stash in the existing trivia report without deleting the pre-fix run.
    p = EXP / "trivia_report.json"
    report = json.loads(p.read_text(encoding="utf-8"))
    report["post_fix_timeline"] = record
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
