"""Phase 0: corpus assertions, prompt token accounting, budget quote."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    EXP,
    HARD_CAP_USD,
    MANIFEST_PATH,
    PRICE_CACHE_MISS,
    PRICE_OUTPUT,
    TARGETS,
    build_a0_prompt,
    build_a1_map_prompt,
    build_a1_reduce_prompt,
    count_tokens,
    is_peak,
    load_api_key,
    load_corpus,
    load_tokenizer,
    make_nonce,
    now_cst,
    write_json,
)

A0_MIN = 25_000
A0_MAX = 49_000
A0_PROMPT_LIMIT = 950_000
A1_PROMPT_LIMIT = 950_000
A1_BOUNDARIES = ((1, 33_333), (33_334, 66_666), (66_667, 100_000))


def main() -> int:
    if is_peak(now_cst()):
        raise SystemExit(f"STOP: current time is peak: {now_cst().isoformat()}")
    load_api_key()  # fail early if missing; never print it.

    records = load_corpus()
    tokenizer = load_tokenizer()
    if len(records) != 100_000:
        raise SystemExit(f"STOP: corpus has {len(records)} records")

    # Re-assert target rows after JSONL round-trip.
    for position, target in TARGETS.items():
        record = records[position - 1]
        assert record["position"] == position
        assert record["content"] == target["text"]
        assert record["target_label"] == target["label"]
    timestamps = [r["timestamp_ms"] for r in records]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:])), "timestamps not increasing"

    nonce = make_nonce("phase0")

    def a0_at(cutoff: int) -> tuple[str, int]:
        prompt = build_a0_prompt(records[:cutoff], nonce)
        return prompt, count_tokens(prompt, tokenizer)

    prompt, tokens = a0_at(A0_MIN)
    if tokens > A0_PROMPT_LIMIT:
        raise SystemExit(f"STOP: A0 25k already uses {tokens} tokens")
    low, high = A0_MIN, A0_MAX
    best_cutoff, best_prompt, best_tokens = A0_MIN, prompt, tokens
    while low <= high:
        mid = (low + high) // 2
        candidate_prompt, candidate_tokens = a0_at(mid)
        if candidate_tokens <= A0_PROMPT_LIMIT:
            best_cutoff, best_prompt, best_tokens = mid, candidate_prompt, candidate_tokens
            low = mid + 1
        else:
            high = mid - 1
    assert best_cutoff < 50_000, "A0 must exclude T2"
    assert best_cutoff >= 20_000, "A0 must include T1"

    shard_prompts: list[str] = []
    shard_tokens: list[int] = []
    for index, (start, end) in enumerate(A1_BOUNDARIES, start=1):
        prompt = build_a1_map_prompt(records[start - 1 : end], index, 3, nonce)
        token_count = count_tokens(prompt, tokenizer)
        shard_prompts.append(prompt)
        shard_tokens.append(token_count)
        if token_count > A1_PROMPT_LIMIT:
            raise SystemExit(f"STOP: A1 shard {index} uses {token_count} tokens")
    reduce_tokens = count_tokens(
        build_a1_reduce_prompt(["NO_MATCH", "NO_MATCH", "NO_MATCH"], nonce),
        tokenizer,
    )

    # Per-trial input: A0 1 call; A1 3 maps + 1 reduce. 3 trials.
    a0_in = best_tokens * 3
    a1_in = (sum(shard_tokens) + reduce_tokens) * 3
    b_in_estimate = 1_500 * 3
    # Conservative output caps: A0 256; A1 3*400+256; B 256.
    a0_out = 256 * 3
    a1_out = (400 * 3 + 256) * 3
    b_out = 256 * 3
    input_cost = (a0_in + a1_in + b_in_estimate) / 1_000_000 * float(PRICE_CACHE_MISS)
    output_cost = (a0_out + a1_out + b_out) / 1_000_000 * float(PRICE_OUTPUT)
    quoted = input_cost + output_cost
    buffer = quoted * 0.10
    if quoted + buffer > float(HARD_CAP_USD):
        raise SystemExit(
            f"STOP: quote ${quoted:.4f} + 10% buffer exceeds cap ${HARD_CAP_USD}"
        )

    report = {
        "phase": 0,
        "checked_at_cst": now_cst().isoformat(timespec="seconds"),
        "offpeak": True,
        "corpus_sha256": hashlib.sha256(
            Path(EXP / "corpus.jsonl").read_bytes()
        ).hexdigest(),
        "a0": {
            "cutoff": best_cutoff,
            "tokens": best_tokens,
            "prompt_sha256": hashlib.sha256(best_prompt.encode("utf-8")).hexdigest(),
            "includes_t1": best_cutoff >= 20_000,
            "excludes_t2": best_cutoff < 50_000,
        },
        "a1": {
            "boundaries": [list(x) for x in A1_BOUNDARIES],
            "shard_tokens": shard_tokens,
            "reduce_tokens": reduce_tokens,
            "total_input_tokens_per_trial": sum(shard_tokens) + reduce_tokens,
        },
        "quote": {
            "a0_input_tokens_3_trials": a0_in,
            "a1_input_tokens_3_trials": a1_in,
            "b_input_tokens_3_trials_estimate": b_in_estimate,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "quoted_usd": round(quoted, 6),
            "with_10pct_buffer_usd": round(quoted + buffer, 6),
            "hard_cap_usd": float(HARD_CAP_USD),
        },
        "status": "green",
    }
    write_json(EXP / "phase0_report.json", report)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = "phase0_green"
    manifest["phase0"] = report
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
