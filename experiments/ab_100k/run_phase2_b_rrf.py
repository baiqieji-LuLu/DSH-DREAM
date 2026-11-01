"""Phase 2 B-only rerun against the RRF + Flat implementation."""

from __future__ import annotations

import json
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
    BASE_URL,
    CST,
    EXP,
    HARD_CAP_CNY,
    HARD_CAP_USD,
    MODEL,
    PRICE_CACHE_MISS,
    PRICE_OUTPUT,
    QUERY,
    TARGETS,
    USD_CNY,
    build_b_prompt,
    cost_usd,
    count_tokens,
    hit_to_dict,
    is_peak,
    load_api_key,
    load_tokenizer,
    make_client,
    make_nonce,
    now_cst,
    read_json,
    usage_to_dict,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_memory.rag_store import RAGStore, MemoryHit  # noqa: E402

FLAT_DIR = EXP / "chroma-flat"
COLLECTION = "verify_flat"
T1_ID = TARGETS[20_000]["id"]
T2_ID = TARGETS[50_000]["id"]
T3_ID = TARGETS[88_888]["id"]
RUN_ID = "phase2b-rrf-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID


class BudgetTracker:
    def __init__(self) -> None:
        self.spent = Decimal("0")

    def precheck(self, prompt_tokens: int, max_tokens: int) -> None:
        worst = (
            Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
            + Decimal(max_tokens) / Decimal(1_000_000) * PRICE_OUTPUT
        )
        if self.spent + worst > HARD_CAP_USD:
            raise RuntimeError(
                f"budget cap: spent=${self.spent:.4f} worst=${worst:.4f}"
            )

    def record(self, usage: dict[str, int]) -> Decimal:
        cost = cost_usd(usage)
        self.spent += cost
        if self.spent > HARD_CAP_USD:
            raise RuntimeError(f"budget exceeded: ${self.spent:.4f}")
        return cost


def call(
    client: Any,
    tokenizer: Any,
    tracker: BudgetTracker,
    *,
    trial: int,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    current = now_cst()
    if is_peak(current):
        raise RuntimeError(f"peak hour: {current.isoformat()}")
    prompt_tokens_est = count_tokens(prompt, tokenizer)
    tracker.precheck(prompt_tokens_est, max_tokens)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    t0 = time.perf_counter()
    response = client.chat.completions.create(**payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = usage_to_dict(response.usage)
    if not usage or not usage.get("prompt_tokens"):
        raise RuntimeError("missing usage")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("empty answer")
    cost = tracker.record(usage)
    record = {
        "arm": "B",
        "trial": trial,
        "step": "generate",
        "architecture": "RRF+Flat",
        "requested_at_cst": current.isoformat(timespec="seconds"),
        "prompt_tokens_est": prompt_tokens_est,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        "cost_usd": str(cost),
        "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
        "latency_ms": round(latency_ms, 4),
        "finish_reason": response.choices[0].finish_reason,
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "answer": text,
        "prompt_sha256": __import__("hashlib").sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "prompt": prompt,
        "usage": usage,
        "response": response.model_dump() if hasattr(response, "model_dump") else {},
    }
    write_json(RUN_DIR / f"b-t{trial}-generate.json", record)
    print(
        f"[B t{trial}] in={usage.get('prompt_tokens')} "
        f"out={usage.get('completion_tokens')} cost=${cost:.6f} "
        f"total=${tracker.spent:.6f} lat={latency_ms:.0f}ms",
        flush=True,
    )
    return record


def main() -> int:
    load_api_key()
    if is_peak(now_cst()):
        raise SystemExit("STOP: peak hour")
    if not FLAT_DIR.exists():
        raise SystemExit(f"STOP: Flat index missing: {FLAT_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] open Flat index and verify gate")
    store = RAGStore(
        persist_directory=FLAT_DIR,
        collection_name=COLLECTION,
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
        flat_max_records=200_000,
    )
    hits = store.search_memory("上周二 备用钥匙", top_k=3)
    ids = [hit.memory_id for hit in hits]
    if not hits or hits[0].memory_id != T3_ID:
        raise SystemExit(f"STOP: G1 failed: {ids}")
    if T1_ID in ids or T2_ID in ids:
        raise SystemExit(f"STOP: G2a failed: {ids}")
    print(f"      top3={ids}")
    b_hits = [hit_to_dict(hit) for hit in hits]
    write_json(RUN_DIR / "b_hits.json", b_hits)
    store.close()

    tokenizer = load_tokenizer()
    client = make_client()
    tracker = BudgetTracker()
    calls: list[dict[str, Any]] = []

    print("[2/3] 3 B generations")
    for trial in range(1, 4):
        nonce = make_nonce(f"b-rrf-t{trial}")
        prompt = build_b_prompt(
            [
                MemoryHit(
                    memory_id=item["memory_id"],
                    content=item["content"],
                    source_type=item["source_type"],
                    metadata=item.get("metadata") or {},
                    distance=item.get("distance"),
                    retrieval_source=item.get("retrieval_source", "hybrid"),
                )
                for item in b_hits
            ],
            nonce,
        )
        prompt_tokens = count_tokens(prompt, tokenizer)
        if prompt_tokens > 2_000:
            raise RuntimeError(f"B prompt unexpectedly large: {prompt_tokens}")
        if trial == 1:
            (RUN_DIR / "b_prompt_trial1.txt").write_text(prompt, encoding="utf-8")
            print(f"      prompt_tokens={prompt_tokens} nonce={nonce}")
        calls.append(call(client, tokenizer, tracker, trial=trial, prompt=prompt, max_tokens=256))

    summary = {
        "run_id": RUN_ID,
        "architecture": "RRF+Flat",
        "model": MODEL,
        "base_url": BASE_URL,
        "calls": len(calls),
        "spent_usd": str(tracker.spent),
        "spent_cny": str((tracker.spent * USD_CNY).quantize(Decimal("0.0001"))),
        "hard_cap_usd": str(HARD_CAP_USD),
        "hard_cap_cny": str(HARD_CAP_CNY),
        "b_prompt_tokens": calls[0]["prompt_tokens_est"] if calls else None,
        "gate": {"G1": True, "G2a": True, "top3": ids},
        "answers": [
            {
                "arm": c["arm"],
                "trial": c["trial"],
                "step": c["step"],
                "answer": c["answer"],
                "usage": c["usage"],
                "latency_ms": c["latency_ms"],
                "cost_usd": c["cost_usd"],
                "finish_reason": c["finish_reason"],
            }
            for c in calls
        ],
    }
    write_json(RUN_DIR / "summary.json", summary)
    write_json(EXP / "phase2b_rrf_summary.json", summary)
    print(
        f"[3/3] done calls={len(calls)} spent=${tracker.spent:.6f} "
        f"(~CNY {(tracker.spent * USD_CNY):.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
