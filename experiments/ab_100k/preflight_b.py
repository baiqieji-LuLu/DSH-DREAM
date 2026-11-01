"""Phase 1: build the temporary Chroma index and enforce G1/G2a/G3."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HOME", str(Path(r"G:\deepseek_ragproject_workspace") / ".hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(r"G:\deepseek_ragproject_workspace") / ".hf-cache" / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from common import (  # noqa: E402
    B_COLLECTION,
    B_DIR,
    EXP,
    QUERY,
    RETRIEVAL_QUERY,
    SYSTEM_PROMPT,
    TARGETS,
    count_tokens,
    hit_to_dict,
    iso_utc_from_ms,
    load_corpus,
    load_tokenizer,
    make_nonce,
    sha256_text,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_memory.rag_store import RAGStore  # noqa: E402
from agent_memory.renderer import build_prompt  # noqa: E402


def _add_batch(store: RAGStore, batch: dict[str, Any], vectors: Any) -> None:
    try:
        store._collection.add(**batch, embeddings=vectors)
    except (TypeError, ValueError):
        store._collection.add(**batch, embeddings=vectors.tolist())


def _rank_of(hits: list[Any], memory_id: str) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit.memory_id == memory_id:
            return index
    return None


def main() -> int:
    if B_DIR.exists():
        shutil.rmtree(B_DIR, ignore_errors=True)
        if B_DIR.exists():
            raise SystemExit(f"STOP: cannot remove old B_DIR {B_DIR}")

    records = load_corpus()
    tokenizer = load_tokenizer()
    t3_id = TARGETS[88_888]["id"]
    t1_id = TARGETS[20_000]["id"]
    t2_id = TARGETS[50_000]["id"]

    print("[1/6] init RAGStore")
    started = time.perf_counter()
    store = RAGStore(
        persist_directory=B_DIR,
        collection_name=B_COLLECTION,
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
    )
    init_seconds = time.perf_counter() - started
    model_name = store.embedding_model_name
    print(f"      model={model_name} init={init_seconds:.2f}s")
    if model_name != "BAAI/bge-small-zh-v1.5":
        raise SystemExit(f"STOP: embedding fallback detected: {model_name}")

    dimension = len(store._embedding_function(["dimension_check"])[0])
    print(f"      dim={dimension}")

    print("[2/6] encode 100k documents")
    texts = [r["content"] for r in records]
    vectors: list[Any] = []
    encode_started = time.perf_counter()
    for start in range(0, len(texts), 128):
        stop = min(start + 128, len(texts))
        vectors.extend(store._embedding_function(texts[start:stop]))
        if stop % 20_000 == 0:
            elapsed = time.perf_counter() - encode_started
            print(f"      encoded {stop:,}/{len(texts):,} ({elapsed:.1f}s)")
    encode_seconds = time.perf_counter() - encode_started
    print(f"      encoded={len(vectors):,} in {encode_seconds:.2f}s")

    ids = [r["id"] for r in records]
    metadatas = [
        {
            "source_type": "user_message",
            "created_at": iso_utc_from_ms(r["timestamp_ms"]),
            "timestamp_ms": r["timestamp_ms"],
            "content_sha256": sha256_text(r["content"]),
        }
        for r in records
    ]

    print("[3/6] write 100k records")
    write_started = time.perf_counter()
    for start in range(0, len(records), 5000):
        stop = min(start + 5000, len(records))
        _add_batch(
            store,
            {
                "ids": ids[start:stop],
                "documents": texts[start:stop],
                "metadatas": metadatas[start:stop],
            },
            vectors[start:stop],
        )
    write_seconds = time.perf_counter() - write_started
    count = store.count()
    print(f"      write={write_seconds:.2f}s count={count}")
    if count != 100_000:
        raise SystemExit(f"STOP: count={count}, expected 100000")

    print("[4/6] verify targets")
    for target in TARGETS.values():
        hit = store.get_memory(target["id"])
        if hit is None or hit.content != target["text"]:
            raise SystemExit(f"STOP: target readback failed: {target['label']}")
    print("      targets OK")

    print("[5/6] retrieval timing and gates")
    cold_started = time.perf_counter()
    cold_hits = store.search_memory(RETRIEVAL_QUERY, top_k=3)
    cold_ms = (time.perf_counter() - cold_started) * 1000
    for _ in range(5):
        store.search_memory("今天天气怎么样", top_k=3)
    timings: list[float] = []
    hits: list[Any] = []
    for _ in range(20):
        t0 = time.perf_counter()
        hits = store.search_memory(RETRIEVAL_QUERY, top_k=3)
        timings.append((time.perf_counter() - t0) * 1000)
    timings_sorted = sorted(timings)
    p50 = timings_sorted[len(timings_sorted) // 2]
    p95 = timings_sorted[min(len(timings_sorted) - 1, int(len(timings_sorted) * 0.95))]

    top12 = store.search_vector(RETRIEVAL_QUERY, top_k=12)
    explicit_terms = store._explicit_keyword_terms(RETRIEVAL_QUERY)
    keyword_hits = store._keyword_search(
        explicit_terms,
        query_text=RETRIEVAL_QUERY,
        top_k=3,
        selected_types=None,
    )
    candidate_by_id: dict[str, Any] = {}
    for hit in list(top12) + list(keyword_hits):
        candidate_by_id.setdefault(hit.memory_id, hit)

    t3_rank_top3 = _rank_of(hits, t3_id)
    t3_rank_top12 = _rank_of(top12, t3_id)
    t1_rank_top12 = _rank_of(top12, t1_id)
    t2_rank_top12 = _rank_of(top12, t2_id)
    t1_keyword_rank = _rank_of(keyword_hits, t1_id)
    t2_keyword_rank = _rank_of(keyword_hits, t2_id)
    t3_keyword_rank = _rank_of(keyword_hits, t3_id)
    t1_in_top3 = _rank_of(hits, t1_id) is not None
    t2_in_top3 = _rank_of(hits, t2_id) is not None
    old_leakage = t1_in_top3 or t2_in_top3

    # Diagnostic conflict-group flags, using the actual shipped predicate and
    # the actual explicit entity terms (not an empty list).
    t3_hit = candidate_by_id.get(t3_id) or next(
        (h for h in hits if h.memory_id == t3_id), None
    )
    t1_hit = candidate_by_id.get(t1_id)
    t2_hit = candidate_by_id.get(t2_id)
    group_t1 = bool(
        t3_hit is not None
        and t1_hit is not None
        and store._same_conflict_group(t3_hit, t1_hit, explicit_terms)
    )
    group_t2 = bool(
        t3_hit is not None
        and t2_hit is not None
        and store._same_conflict_group(t3_hit, t2_hit, explicit_terms)
    )

    old_recalled = [
        (name, rank, keyword_rank, grouped)
        for name, rank, keyword_rank, grouped in (
            ("T1", t1_rank_top12, t1_keyword_rank, group_t1),
            ("T2", t2_rank_top12, t2_keyword_rank, group_t2),
        )
        if rank is not None or keyword_rank is not None
    ]

    if old_leakage:
        mechanism = "leakage"
    elif not old_recalled:
        mechanism = "not_recalled"
    else:
        grouped = [item for item in old_recalled if item[3]]
        if len(grouped) == len(old_recalled):
            mechanism = "latest_wins"
        elif grouped:
            mechanism = "mixed"
        else:
            mechanism = "ranking_only"

    # G3: the rendered prompt must not contain T1/T2 IDs.
    fixed_nonce = make_nonce("b-preflight")
    rendered = build_prompt(
        SYSTEM_PROMPT,
        QUERY,
        anchors=None,
        window=None,
        rag_hint=hits,
        nonce=fixed_nonce,
    )
    prompt_tokens = count_tokens(rendered.text, tokenizer)
    leaked_ids = [mid for mid in rendered.retrieved_chunk_ids if mid in {t1_id, t2_id}]
    g3 = not leaked_ids

    # Extra sensitivity: full natural-language query for retrieval, no extra API.
    full_hits = store.search_memory(QUERY, top_k=3)
    full_top1 = full_hits[0].memory_id if full_hits else None
    full_old_leak = any(h.memory_id in {t1_id, t2_id} for h in full_hits)

    report = {
        "phase": 1,
        "offpeak": True,
        "model": model_name,
        "dimension": dimension,
        "init_seconds": round(init_seconds, 4),
        "encode_seconds": round(encode_seconds, 4),
        "write_seconds": round(write_seconds, 4),
        "count": count,
        "index_size_mb": round(
            sum(p.stat().st_size for p in B_DIR.rglob("*") if p.is_file()) / 1024 / 1024,
            2,
        ),
        "retrieval": {
            "query": RETRIEVAL_QUERY,
            "cold_ms": round(cold_ms, 4),
            "warm_ms": [round(x, 4) for x in timings],
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "min_ms": round(min(timings), 4),
            "max_ms": round(max(timings), 4),
        },
        "hits": [hit_to_dict(h) for h in hits],
        "top12": [hit_to_dict(h) for h in top12],
        "keyword_hits": [hit_to_dict(h) for h in keyword_hits],
        "explicit_terms": explicit_terms,
        "ranks": {
            "t3_top3": t3_rank_top3,
            "t3_top12": t3_rank_top12,
            "t1_top12": t1_rank_top12,
            "t2_top12": t2_rank_top12,
            "t3_keyword": t3_keyword_rank,
            "t1_keyword": t1_keyword_rank,
            "t2_keyword": t2_keyword_rank,
            "t1_in_top3": t1_in_top3,
            "t2_in_top3": t2_in_top3,
        },
        "conflict_group": {"t3_t1": group_t1, "t3_t2": group_t2},
        "old_leakage": old_leakage,
        "latest_wins_mechanism": mechanism,
        "gates": {
            "G1": bool(hits and hits[0].memory_id == t3_id),
            "G2a": not old_leakage,
            "G2b_mechanism": mechanism,
            "G3": g3,
        },
        "b_prompt_tokens": prompt_tokens,
        "b_prompt_sha256": __import__("hashlib").sha256(rendered.text.encode("utf-8")).hexdigest(),
        "b_prompt_nonce": fixed_nonce,
        "b_prompt_text": rendered.text,
        "b_hits_json": [hit_to_dict(h) for h in hits],
        "fullquery_sensitivity": {
            "query": QUERY,
            "top1": full_top1,
            "old_leakage": full_old_leak,
            "hits": [hit_to_dict(h) for h in full_hits],
        },
    }
    write_json(EXP / "phase1_report.json", report)
    write_json(EXP / "b_hits.json", report["b_hits_json"])
    (EXP / "b_prompt_preflight.txt").write_text(rendered.text, encoding="utf-8")

    # Always release Chroma before any cleanup; Windows file handles matter.
    store.close()
    del store, vectors
    gc.collect()

    print(
        json.dumps(
            {
                "gates": report["gates"],
                "mechanism": mechanism,
                "ranks": report["ranks"],
                "conflict_group": report["conflict_group"],
                "b_prompt_tokens": prompt_tokens,
                "retrieval_p50_ms": report["retrieval"]["p50_ms"],
                "top3": [
                    {
                        "rank": index,
                        "id": hit.memory_id,
                        "distance": hit.distance,
                        "content": hit.content,
                    }
                    for index, hit in enumerate(hits, start=1)
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not (report["gates"]["G1"] and report["gates"]["G2a"] and report["gates"]["G3"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
