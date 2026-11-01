"""Local 100k verification: RRF + exact Flat + generic latest-wins."""

from __future__ import annotations

import gc
import json
import os
import shutil
import statistics
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
    EXP,
    RETRIEVAL_QUERY,
    TARGETS,
    iso_utc_from_ms,
    load_corpus,
    sha256_text,
    write_json,
)

ROOT = Path(r"G:\deepseek_ragproject_workspace")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent_memory.rag_store import RAGStore  # noqa: E402

FLAT_DIR = EXP / "chroma-flat"
T3_ID = TARGETS[88_888]["id"]
T1_ID = TARGETS[20_000]["id"]
T2_ID = TARGETS[50_000]["id"]


def _add(store: RAGStore, batch: dict[str, Any], vectors: Any) -> None:
    try:
        store._collection.add(**batch, embeddings=vectors)
    except (TypeError, ValueError):
        store._collection.add(**batch, embeddings=vectors.tolist())


def _rank(hits: list[Any], memory_id: str) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit.memory_id == memory_id:
            return index
    return None


def main() -> int:
    if FLAT_DIR.exists():
        shutil.rmtree(FLAT_DIR, ignore_errors=True)
    records = load_corpus()
    print(f"[1/5] {len(records):,} records")

    store = RAGStore(
        persist_directory=FLAT_DIR,
        collection_name="verify_flat",
        model_name="BAAI/bge-small-zh-v1.5",
        device="cpu",
        flat_max_records=200_000,
    )
    print(f"      model={store.embedding_model_name} dim={len(store._embedding_function(['x'])[0])}")

    texts = [r["content"] for r in records]
    vectors: list[Any] = []
    started = time.perf_counter()
    for start in range(0, len(texts), 128):
        stop = min(start + 128, len(texts))
        vectors.extend(store._embedding_function(texts[start:stop]))
        if stop % 20_000 == 0:
            print(f"      encoded {stop:,}/{len(texts):,} ({time.perf_counter()-started:.1f}s)")
    encode_seconds = time.perf_counter() - started

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
    write_started = time.perf_counter()
    for start in range(0, len(records), 5000):
        stop = min(start + 5000, len(records))
        _add(
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
    print(f"[2/5] encoded={encode_seconds:.1f}s write={write_seconds:.1f}s count={count}")
    if count != 100_000:
        raise SystemExit(f"count={count}")

    for target in TARGETS.values():
        hit = store.get_memory(target["id"])
        if hit is None or hit.content != target["text"]:
            raise SystemExit(f"target readback failed: {target['label']}")

    print("[3/5] first Flat search (builds corpus cache)")
    cold_started = time.perf_counter()
    cold_vector = store.search_vector(RETRIEVAL_QUERY, top_k=12)
    flat_cold_ms = (time.perf_counter() - cold_started) * 1000
    for _ in range(5):
        store.search_vector(RETRIEVAL_QUERY, top_k=12)
    flat_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        vector_hits = store.search_vector(RETRIEVAL_QUERY, top_k=12)
        flat_times.append((time.perf_counter() - t0) * 1000)

    print("[4/5] first hybrid search (builds BM25)")
    cold_started = time.perf_counter()
    cold_hits = store.search_memory(RETRIEVAL_QUERY, top_k=3)
    hybrid_cold_ms = (time.perf_counter() - cold_started) * 1000
    for _ in range(5):
        store.search_memory(RETRIEVAL_QUERY, top_k=3)
    hybrid_times = []
    hits: list[Any] = []
    for _ in range(20):
        t0 = time.perf_counter()
        hits = store.search_memory(RETRIEVAL_QUERY, top_k=3)
        hybrid_times.append((time.perf_counter() - t0) * 1000)

    print("[5/5] gates")
    vector_ranks = {
        "T1": _rank(vector_hits, T1_ID),
        "T2": _rank(vector_hits, T2_ID),
        "T3": _rank(vector_hits, T3_ID),
    }
    top_ids = [hit.memory_id for hit in hits]
    t3_top1 = bool(hits and hits[0].memory_id == T3_ID)
    old_leak = T1_ID in top_ids or T2_ID in top_ids
    groups = {
        "T3_T1": bool(store._same_conflict_group(
            next(h for h in vector_hits if h.memory_id == T3_ID),
            next(h for h in vector_hits if h.memory_id == T1_ID),
        )),
        "T3_T2": bool(store._same_conflict_group(
            next(h for h in vector_hits if h.memory_id == T3_ID),
            next(h for h in vector_hits if h.memory_id == T2_ID),
        )),
    }

    report = {
        "phase": "rrf_flat_verify",
        "records": count,
        "model": store.embedding_model_name,
        "flat_max_records": store.flat_max_records,
        "encode_seconds": round(encode_seconds, 2),
        "write_seconds": round(write_seconds, 2),
        "flat_cold_ms": round(flat_cold_ms, 2),
        "flat_p50_ms": round(statistics.median(flat_times), 2),
        "hybrid_cold_ms": round(hybrid_cold_ms, 2),
        "hybrid_p50_ms": round(statistics.median(hybrid_times), 2),
        "vector_ranks_top12": vector_ranks,
        "hits": [
            {
                "id": hit.memory_id,
                "content": hit.content,
                "distance": hit.distance,
                "source": hit.retrieval_source,
            }
            for hit in hits
        ],
        "t3_top1": t3_top1,
        "old_leakage": old_leak,
        "conflict_groups": groups,
        "gates": {
            "G1": t3_top1,
            "G2a": not old_leak,
        },
    }
    write_json(EXP / "rrf_flat_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    store.close()
    del store, vectors
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
