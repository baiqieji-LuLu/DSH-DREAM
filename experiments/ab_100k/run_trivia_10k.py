"""10k trivia corpus: 100 diary summaries vs RAG timeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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

INDEX_DIR = EXP / "trivia-10k-flat"
RUN_ID = "trivia-10k-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID
PRIOR = Decimal("1.590376518")
CAP = Decimal("3.50")
QUERY = "我那个星巴克蓝色马克杯现在怎么样了？请告诉我它的完整历史。"
SUMMARY_PROMPT = "请总结这100条日常记录的核心事件。"

TARGETS = {
    2_500: "今天逛街买了一个星巴克的蓝色马克杯，挺好看的。",
    5_000: "刚才不小心把那个蓝色马克杯摔碎了，有点可惜。",
    7_500: "又在网上买了一个同款的蓝色马克杯补上，希望这次别摔了。",
}
NOISE = (
    "今天中午吃了一碗牛肉面，味道还可以。",
    "早上坐地铁上班，车厢里人很多。",
    "下午开了两个小时的会，有点累。",
    "晚饭后沿着小区走了两圈。",
    "今天把客厅的绿萝浇了水。",
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
    base = datetime(2026, 9, 1, 12, 0, tzinfo=CST)
    records = []
    for position in range(1, 10_001):
        content = TARGETS.get(position) or NOISE[(position - 1) % len(NOISE)]
        # Add a date prefix to keep the fiction readable; targets remain exact.
        if position not in TARGETS:
            dt = base + timedelta(minutes=position * 3)
            content = f"{dt.month}月{dt.day}日，{content}"
        else:
            dt = base + timedelta(minutes=position * 3)
        records.append(
            {
                "id": f"trivia10k-{position:05d}",
                "position": position,
                "timestamp_ms": int(dt.timestamp() * 1000),
                "content": content,
                "source_type": "user_message",
            }
        )
    return records


class Tracker:
    def __init__(self) -> None:
        self.spent = PRIOR

    def precheck(self, prompt_tokens: int, max_tokens: int) -> None:
        worst = (
            Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
            + Decimal(max_tokens) / Decimal(1_000_000) * PRICE_OUTPUT
        )
        if self.spent + worst > CAP:
            raise RuntimeError(f"budget cap: {self.spent} + {worst}")

    def record(self, usage: dict[str, int]) -> Decimal:
        cost = cost_usd(usage)
        self.spent += cost
        if self.spent > CAP:
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
    return record


def summarize_group(
    client: Any,
    tokenizer: Any,
    tracker: Tracker,
    records: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    log = "\n".join(f"- {r['content']}" for r in records)
    return call(
        client,
        tokenizer,
        tracker,
        label=label,
        system="你是一个日记整理助手。只输出日记正文，不要解释。",
        user=f"下面是100条日常记录：\n{log}\n\n{SUMMARY_PROMPT}",
        max_tokens=400,
    )


def main() -> int:
    load_api_key()
    if is_peak(now_cst()):
        raise SystemExit("STOP: peak hour")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    tokenizer = load_tokenizer()
    client = make_client()
    tracker = Tracker()

    print("[1/4] 100 standard diary summaries")
    standard: dict[int, dict[str, Any]] = {}
    for group in range(1, 101):
        start = (group - 1) * 100
        result = summarize_group(
            client, tokenizer, tracker, records[start : start + 100],
            label=f"standard-{group:03d}",
        )
        standard[group] = result
        if group % 10 == 0:
            print(f"      {group}/100 spent=${tracker.spent:.6f}", flush=True)

    # The fixed 100-row groups put 2500/5000/7500 at the END.  To honor the
    # "target in the middle" requirement, also summarize three centered
    # 100-row windows.
    print("[2/4] 3 centered target windows (2451-2550, 4951-5050, 7451-7550)")
    centered: dict[int, dict[str, Any]] = {}
    for group in (25, 50, 75):
        center = group * 100
        start = center - 50  # 1-based inclusive start
        result = summarize_group(
            client, tokenizer, tracker, records[start : start + 100],
            label=f"centered-{group:03d}",
        )
        centered[group] = result

    print("[3/4] inspect diaries for the mug")
    matches = []
    for group, result in standard.items():
        text = result["answer"]
        if any(term in text for term in ("马克杯", "星巴克", "杯子")):
            matches.append((group, "standard", text))
    for group, result in centered.items():
        text = result["answer"]
        if any(term in text for term in ("马克杯", "星巴克", "杯子")):
            matches.append((group, "centered", text))
    for group, kind, text in matches:
        print(f"--- match {kind} {group} ---\n{text}\n")

    # C query uses the three centered diaries (target at row 50).
    c_context = "\n\n".join(
        f"【日记{group}】\n{centered[group]['answer']}" for group in (25, 50, 75)
    )
    c_query = call(
        client, tokenizer, tracker,
        label="c-query",
        system="你是一个严谨的个人助理。只依据提供的日记回答。材料中没有的信息，明确说不知道，不要编造。",
        user=f"以下是三篇日记：\n<diaries>\n{c_context}\n</diaries>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    print("[4/4] RAG Flat timeline")
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    store = RAGStore(
        persist_directory=INDEX_DIR,
        collection_name="trivia_10k",
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
    hits = store.search_memory(
        "星巴克 蓝色马克杯",
        top_k=5,
        timeline_mode=True,
    )
    store.close()
    print("timeline_hits", [(h.memory_id, h.content) for h in hits])

    rag_context = "\n".join(f"[{i}] {h.content}" for i, h in enumerate(hits, 1))
    rag_query = call(
        client, tokenizer, tracker,
        label="rag-query",
        system="你是一个严谨的个人助理。只依据提供的记忆原文回答。材料中没有的信息，明确说不知道，不要编造。",
        user=f"以下是按时间顺序排列的记忆原文：\n<memory>\n{rag_context}\n</memory>\n\n问题：{QUERY}",
        max_tokens=256,
    )

    total_c = tracker.spent - PRIOR
    report = {
        "run_id": RUN_ID,
        "records": 10_000,
        "targets": TARGETS,
        "standard_matches": [
            {"group": group, "diary": text} for group, kind, text in matches if kind == "standard"
        ],
        "centered_matches": [
            {"group": group, "diary": text} for group, kind, text in matches if kind == "centered"
        ],
        "centered_diaries": {str(group): centered[group]["answer"] for group in (25, 50, 75)},
        "standard_target_diaries": {str(group): standard[group]["answer"] for group in (25, 50, 75)},
        "diary_matches_count": len(matches),
        "c_query": c_query["answer"],
        "rag_hits": [
            {"id": h.memory_id, "content": h.content, "timestamp_ms": h.metadata.get("timestamp_ms")}
            for h in hits
        ],
        "rag_query": rag_query["answer"],
        "cost": {
            "standard_100": str(sum(Decimal(standard[g]["cost_usd"]) for g in standard)),
            "centered_3": str(sum(Decimal(centered[g]["cost_usd"]) for g in centered)),
            "c_query": c_query["cost_usd"],
            "rag_query": rag_query["cost_usd"],
            "total_c": str(total_c),
            "total_with_prior": str(tracker.spent),
        },
        "latency": {
            "c_query": c_query["latency_ms"],
            "rag_query": rag_query["latency_ms"],
        },
        "usage": {
            "c_query": c_query["usage"],
            "rag_query": rag_query["usage"],
        },
    }
    write_json(RUN_DIR / "summary.json", report)
    write_json(EXP / "trivia_10k_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
