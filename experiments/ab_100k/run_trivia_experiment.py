"""Adversarial 100-record trivia test: diary vs RAG, with a blue mug."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
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

TRIVIA_DIR = EXP / "trivia-flat"
RUN_ID = "trivia-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID
PRIOR_SPENT = Decimal("1.589874168")
CAP_USD = Decimal("3.50")

TARGETS = {
    10: "今天逛街买了一个星巴克的蓝色马克杯。",
    50: "刚才不小心把那个蓝色马克杯摔碎了，有点可惜。",
    90: "又在网上买了一个同款的蓝色马克杯补上。",
}
QUERY = "我那个星巴克蓝色马克杯现在怎么样了？"
SUMMARY_PROMPT = "请总结这100条日常记录的核心事件。"

NOISE = (
    "今天中午吃了一碗牛肉面，味道还可以。",
    "早上坐地铁上班，车厢里人很多。",
    "下午开了两个小时的会，有点累。",
    "晚饭后沿着小区走了两圈。",
    "今天把阳台的绿萝浇了水。",
    "晚上看了一集纪录片，讲深海生物。",
    "下班路上顺便取了快递。",
    "今天风很大，窗户响了一晚上。",
    "中午和同事去楼下吃了盖饭。",
    "把上周的账单整理了一下。",
    "晚上煮了番茄鸡蛋面。",
    "早上差点没赶上公交。",
    "下午把下周的计划列了一遍。",
    "今天咖啡喝多了，晚上有点睡不着。",
    "把书桌收拾干净了。",
    "楼下便利店换了新老板。",
    "晚上给家里打了个电话。",
    "今天的工作比平时顺一点。",
    "午休时在工位上眯了十分钟。",
    "晚饭后看了一部老电影。",
)


def build_records() -> list[dict[str, Any]]:
    base = datetime(2026, 10, 1, 12, 0, tzinfo=CST)
    records = []
    for position in range(1, 101):
        content = TARGETS.get(position) or NOISE[(position - 1) % len(NOISE)]
        ts = int((base + timedelta(hours=position)).timestamp() * 1000)
        records.append(
            {
                "id": f"trivia-{position:03d}",
                "position": position,
                "timestamp_ms": ts,
                "content": content,
                "source_type": "user_message",
            }
        )
    return records


class Tracker:
    def __init__(self) -> None:
        self.spent = PRIOR_SPENT

    def precheck(self, prompt_tokens: int, max_tokens: int) -> None:
        worst = (
            Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
            + Decimal(max_tokens) / Decimal(1_000_000) * PRICE_OUTPUT
        )
        if self.spent + worst > CAP_USD:
            raise RuntimeError(f"budget cap: {self.spent} + {worst}")

    def record(self, usage: dict[str, int]) -> Decimal:
        cost = cost_usd(usage)
        self.spent += cost
        if self.spent > CAP_USD:
            raise RuntimeError(f"budget exceeded: {self.spent}")
        return cost


def call(
    client: Any,
    tokenizer: Any,
    tracker: Tracker,
    *,
    label: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any]:
    if is_peak(now_cst()):
        raise RuntimeError(f"peak hour: {now_cst().isoformat()}")
    prompt = f"{system}\n\n{user}"
    prompt_tokens_est = count_tokens(prompt, tokenizer)
    tracker.precheck(prompt_tokens_est, max_tokens)
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = usage_to_dict(response.usage)
    text = (response.choices[0].message.content or "").strip()
    if not usage or not text:
        raise RuntimeError("bad response")
    cost = tracker.record(usage)
    record = {
        "label": label,
        "answer": text,
        "usage": usage,
        "cost_usd": str(cost),
        "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
        "latency_ms": round(latency_ms, 4),
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens_est": prompt_tokens_est,
        "prompt": prompt,
        "requested_at_cst": now_cst().isoformat(timespec="seconds"),
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "response": response.model_dump() if hasattr(response, "model_dump") else {},
    }
    write_json(RUN_DIR / f"{label}.json", record)
    print(
        f"[{label}] in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
        f"cost=${cost:.6f} total=${tracker.spent:.6f} lat={latency_ms:.0f}ms",
        flush=True,
    )
    return record


def main() -> int:
    load_api_key()
    if is_peak(now_cst()):
        raise SystemExit("STOP: peak hour")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    tokenizer = load_tokenizer()
    client = make_client()
    tracker = Tracker()

    print("[1/4] build 100-record Flat index")
    if TRIVIA_DIR.exists():
        import shutil

        shutil.rmtree(TRIVIA_DIR, ignore_errors=True)
    store = RAGStore(
        persist_directory=TRIVIA_DIR,
        collection_name="trivia_100",
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
        flat_max_records=200_000,
    )
    texts = [r["content"] for r in records]
    vectors = store._embedding_function(texts)
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
        embeddings=vectors,
    )

    print("[2/4] search_memory + search_vector")
    memory_hits = store.search_memory("星巴克 蓝色马克杯", top_k=3)
    vector_hits = store.search_vector("星巴克 蓝色马克杯", top_k=3)
    memory_ids = [h.memory_id for h in memory_hits]
    vector_ids = [h.memory_id for h in vector_hits]
    target_ids = {f"trivia-{p:03d}" for p in TARGETS}
    print("search_memory:", [(h.memory_id, h.content) for h in memory_hits])
    print("search_vector:", [(h.memory_id, h.content) for h in vector_hits])
    print("targets:", target_ids)
    store.close()

    print("[3/4] diary")
    log = "\n".join(f"- {r['content']}" for r in records)
    diary = call(
        client,
        tokenizer,
        tracker,
        label="diary-summary",
        system="你是一个日记整理助手。只输出日记正文，不要解释。",
        user=f"下面是100条日常记录：\n{log}\n\n{SUMMARY_PROMPT}",
        max_tokens=400,
    )
    diary_text = diary["answer"]
    print("[DIARY]\n" + diary_text)

    print("[4/4] diary query + RAG query")
    query_system = (
        "你是一个严谨的个人助理。只依据提供的材料回答。"
        "材料中没有的信息，明确说不知道，不要编造。"
    )
    diary_query = call(
        client,
        tokenizer,
        tracker,
        label="diary-query",
        system=query_system,
        user=f"以下是你写的日记：\n<diary>\n{diary_text}\n</diary>\n\n问题：{QUERY}",
        max_tokens=256,
    )
    rag_context = "\n".join(
        f"[{i}] {h.content}" for i, h in enumerate(memory_hits, start=1)
    )
    rag_query = call(
        client,
        tokenizer,
        tracker,
        label="rag-query",
        system=query_system,
        user=f"以下是从记忆中检索到的原文：\n<memory>\n{rag_context}\n</memory>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    # Diagnostic only: if search_memory collapsed versions, show what exact
    # vector retrieval would have supplied.
    vector_context = "\n".join(
        f"[{i}] {h.content}" for i, h in enumerate(vector_hits, start=1)
    )
    vector_query = call(
        client,
        tokenizer,
        tracker,
        label="rag-vector-query",
        system=query_system,
        user=f"以下是从记忆中精确检索到的原文：\n<memory>\n{vector_context}\n</memory>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    total_c = tracker.spent - PRIOR_SPENT
    report = {
        "run_id": RUN_ID,
        "records": 100,
        "target_ids": sorted(target_ids),
        "query": QUERY,
        "search_memory_ids": memory_ids,
        "search_vector_ids": vector_ids,
        "search_memory_hits": [
            {"id": h.memory_id, "content": h.content, "distance": h.distance}
            for h in memory_hits
        ],
        "search_vector_hits": [
            {"id": h.memory_id, "content": h.content, "distance": h.distance}
            for h in vector_hits
        ],
        "diary": diary_text,
        "diary_query": diary_query["answer"],
        "rag_query": rag_query["answer"],
        "rag_vector_query": vector_query["answer"],
        "cost": {
            "diary_summary": diary["cost_usd"],
            "diary_query": diary_query["cost_usd"],
            "rag_query": rag_query["cost_usd"],
            "rag_vector_diagnostic": vector_query["cost_usd"],
            "total_c": str(total_c),
            "total_with_prior": str(tracker.spent),
        },
        "latency": {
            "diary_summary": diary["latency_ms"],
            "diary_query": diary_query["latency_ms"],
            "rag_query": rag_query["latency_ms"],
            "rag_vector_diagnostic": vector_query["latency_ms"],
        },
        "usage": {
            "diary_summary": diary["usage"],
            "diary_query": diary_query["usage"],
            "rag_query": rag_query["usage"],
            "rag_vector_diagnostic": vector_query["usage"],
        },
    }
    write_json(RUN_DIR / "summary.json", report)
    write_json(EXP / "trivia_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
