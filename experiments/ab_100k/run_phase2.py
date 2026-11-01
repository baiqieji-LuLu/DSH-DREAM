"""Phase 2: run the 18 paid DeepSeek calls with a hard budget guard."""

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
    SCENARIO_DATE_LABEL,
    TARGETS,
    USD_CNY,
    build_a0_prompt,
    build_a1_map_prompt,
    build_a1_reduce_prompt,
    build_b_prompt,
    cost_usd,
    count_tokens,
    hit_from_dict,
    is_peak,
    load_api_key,
    load_corpus,
    load_tokenizer,
    make_client,
    make_nonce,
    now_cst,
    read_json,
    usage_to_dict,
    write_json,
)

T3_ID = TARGETS[88_888]["id"]
T1_ID = TARGETS[20_000]["id"]
T2_ID = TARGETS[50_000]["id"]
A1_BOUNDARIES = ((1, 33_333), (33_334, 66_666), (66_667, 100_000))
RUN_ID = "phase2-" + datetime.now(tz=CST).strftime("%Y%m%dT%H%M%S")
RUN_DIR = EXP / "runs" / RUN_ID


class BudgetExceeded(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self) -> None:
        self.spent = Decimal("0")

    def precheck(self, prompt_tokens: int, max_tokens: int) -> None:
        worst = (
            Decimal(prompt_tokens) / Decimal(1_000_000) * PRICE_CACHE_MISS
            + Decimal(max_tokens) / Decimal(1_000_000) * PRICE_OUTPUT
        )
        if self.spent + worst > HARD_CAP_USD:
            raise BudgetExceeded(
                f"precheck would exceed cap: spent=${self.spent:.4f} worst=${worst:.4f}"
            )
        if self.spent + worst > HARD_CAP_USD * Decimal("0.8"):
            print(
                f"[budget] warning: spent=${self.spent:.4f} next_worst=${worst:.4f}",
                flush=True,
            )

    def record(self, usage: dict[str, int]) -> Decimal:
        cost = cost_usd(usage)
        self.spent += cost
        if self.spent > HARD_CAP_USD:
            raise BudgetExceeded(
                f"actual spend ${self.spent:.4f} exceeds cap ${HARD_CAP_USD}"
            )
        return cost


def ensure_offpeak() -> None:
    current = now_cst()
    if is_peak(current):
        raise RuntimeError(f"peak hour: {current.isoformat()}")


def _usage_with_details(usage: Any) -> dict[str, int]:
    data = usage_to_dict(usage)
    if isinstance(usage, dict):
        raw = usage
    elif usage is not None and hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        raw = {}
    details = raw.get("prompt_tokens_details") if isinstance(raw, dict) else None
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int) and "prompt_cache_hit_tokens" not in data:
            data["prompt_cache_hit_tokens"] = cached
            data["prompt_cache_miss_tokens"] = max(
                0, int(data.get("prompt_tokens", 0)) - cached
            )
    return data


def call_api(
    client: Any,
    tokenizer: Any,
    tracker: BudgetTracker,
    *,
    arm: str,
    trial: int,
    step: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    ensure_offpeak()
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
    started_at = now_cst().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    response = client.chat.completions.create(**payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    usage = _usage_with_details(response.usage)
    if not usage or int(usage.get("prompt_tokens", 0)) <= 0:
        raise RuntimeError(f"missing usage in response: {getattr(response, 'id', None)}")
    text = ""
    finish_reason = None
    if response.choices:
        message = response.choices[0].message
        text = (getattr(message, "content", None) or "").strip()
        finish_reason = response.choices[0].finish_reason
    if not text:
        raise RuntimeError(
            f"empty completion: id={getattr(response, 'id', None)} finish={finish_reason}"
        )
    cost = tracker.record(usage)
    record = {
        "arm": arm,
        "trial": trial,
        "step": step,
        "requested_at_cst": started_at,
        "prompt_tokens_est": prompt_tokens_est,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        "cost_usd": str(cost),
        "cost_cny": str((cost * USD_CNY).quantize(Decimal("0.0001"))),
        "latency_ms": round(latency_ms, 4),
        "finish_reason": finish_reason,
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
    path = RUN_DIR / f"{arm.lower()}-t{trial}-{step}.json"
    write_json(path, record)
    print(
        f"[{arm} t{trial} {step}] in={usage.get('prompt_tokens')} "
        f"out={usage.get('completion_tokens')} "
        f"cost=${cost:.6f} total=${tracker.spent:.6f} "
        f"lat={latency_ms:.0f}ms finish={finish_reason}",
        flush=True,
    )
    return record


def main() -> int:
    ensure_offpeak()
    load_api_key()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    phase0 = read_json(EXP / "phase0_report.json")
    phase1 = read_json(EXP / "phase1_report.json")
    if not phase1["gates"]["G1"] or not phase1["gates"]["G2a"] or not phase1["gates"]["G3"]:
        raise RuntimeError(f"Phase 1 gates are not green: {phase1['gates']}")

    records = load_corpus()
    tokenizer = load_tokenizer()
    client = make_client()
    tracker = BudgetTracker()
    calls: list[dict[str, Any]] = []

    cutoff = int(phase0["a0"]["cutoff"])
    a0_records = records[:cutoff]
    # Build once per trial because the nonce is part of the prompt.
    for trial in range(1, 4):
        prompt = build_a0_prompt(a0_records, make_nonce(f"a0-t{trial}"))
        calls.append(
            call_api(
                client,
                tokenizer,
                tracker,
                arm="A0",
                trial=trial,
                step="full",
                prompt=prompt,
                max_tokens=256,
            )
        )

    for trial in range(1, 4):
        shard_outputs: list[str] = []
        for index, (start, end) in enumerate(A1_BOUNDARIES, start=1):
            prompt = build_a1_map_prompt(
                records[start - 1 : end],
                index,
                3,
                make_nonce(f"a1-t{trial}-s{index}"),
            )
            result = call_api(
                client,
                tokenizer,
                tracker,
                arm="A1",
                trial=trial,
                step=f"map{index}",
                prompt=prompt,
                max_tokens=400,
            )
            calls.append(result)
            shard_outputs.append(result["answer"])
        reduce_prompt = build_a1_reduce_prompt(
            shard_outputs,
            make_nonce(f"a1-t{trial}-reduce"),
        )
        calls.append(
            call_api(
                client,
                tokenizer,
                tracker,
                arm="A1",
                trial=trial,
                step="reduce",
                prompt=reduce_prompt,
                max_tokens=256,
            )
        )

    hits_payload = read_json(EXP / "b_hits.json")
    hits = [hit_from_dict(item) for item in hits_payload]
    hit_ids = {hit.memory_id for hit in hits}
    if T1_ID in hit_ids or T2_ID in hit_ids:
        raise RuntimeError("refusing B run: T1/T2 leaked into B prompt")
    for trial in range(1, 4):
        prompt = build_b_prompt(hits, make_nonce(f"b-t{trial}"))
        calls.append(
            call_api(
                client,
                tokenizer,
                tracker,
                arm="B",
                trial=trial,
                step="generate",
                prompt=prompt,
                max_tokens=256,
            )
        )

    summary = {
        "run_id": RUN_ID,
        "scenario_date": SCENARIO_DATE_LABEL,
        "model": MODEL,
        "base_url": BASE_URL,
        "started_at_cst": calls[0]["requested_at_cst"] if calls else None,
        "finished_at_cst": now_cst().isoformat(timespec="seconds"),
        "calls": len(calls),
        "spent_usd": str(tracker.spent),
        "spent_cny": str((tracker.spent * USD_CNY).quantize(Decimal("0.0001"))),
        "hard_cap_usd": str(HARD_CAP_USD),
        "hard_cap_cny": str(HARD_CAP_CNY),
        "phase1_gates": phase1["gates"],
        "a0_cutoff": cutoff,
        "answers": [
            {
                "arm": call["arm"],
                "trial": call["trial"],
                "step": call["step"],
                "answer": call["answer"],
                "usage": call["usage"],
                "latency_ms": call["latency_ms"],
                "cost_usd": call["cost_usd"],
                "finish_reason": call["finish_reason"],
            }
            for call in calls
        ],
    }
    write_json(RUN_DIR / "summary.json", summary)
    write_json(EXP / "phase2_summary.json", summary)
    print(
        f"[done] calls={len(calls)} spent=${tracker.spent:.6f} "
        f"(~CNY {(tracker.spent * USD_CNY):.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
