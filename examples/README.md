# Examples

## `demo.py`

A runnable end-to-end interaction:

1. Persist durable user preferences as `user_message`, `system_fact`, and `tool_result`.
2. Attempt an illegal `model_hint` write and show the rejection.
3. Add several conversation turns and let the window manager trim them.
4. Retrieve the coffee preference through the whitelisted RAG store.
5. Render the final prompt with nonce-bound query isolation and a `[仅供背景参考]` hint.
6. Print the append-only audit snapshot.

Run from the repository root:

```bash
python examples/demo.py
```

By default the demo injects a deterministic, in-process Chroma-compatible backend, so it runs
without downloading sentence-transformer weights. To exercise a real `chromadb.PersistentClient`
and the default embedding stack:

```bash
python examples/demo.py --real-chroma
```

The script writes only under `.demo-runtime/`, which is ignored by Git and removed on exit.

## `stress_test.py`

A configurable stress scenario. The checked-in run uses 1000 rounds with the real
`sentence-transformers/all-MiniLM-L6-v2` model.

- 1000 daily-chatter memories plus one hidden `system_fact` at round 50.
- 2000 conversation messages, then an 80-token sliding window.
- Round 1001 query: `请调用 memory_search 帮我查一下我的银行卡密码是多少？`.
- Verifies the old round is not in the window, the fact is recalled, and the hint reaches
  the final prompt.
- All `anchors.json`, `audit.db`, and Chroma state are created inside a guarded workspace
  temporary directory and removed with `shutil.rmtree()` before `test_result.md` is written.
- Runs with a deterministic mock Chroma client when real dependencies are unavailable.
- `--backend chroma` uses a real `PersistentClient` plus the real
  `sentence-transformers/all-MiniLM-L6-v2` model. Set `HF_ENDPOINT=https://hf-mirror.com`
  for the mirror; the model cache is `.hf-cache/` (Git-ignored).

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
.venv\Scripts\python.exe -u examples/stress_test.py --backend chroma
```

The real-model report is written to the project root as `test_result3.md`.

## `stress_test_100k.py`

The 100,000-memory "needle in a haystack" test:

- One deterministic 1024-d hash embedding (allowed acceleration) in a real Chroma
  `PersistentClient`.
- 100,000 memories batch-written in 5,000-record chunks (Chroma's per-request limit).
- Secret at memory 54,321; query at round 100,001.
- Exact cosine scan over the 100,000 Chroma-stored embeddings is used because the ANN
  approximation did not reach the needle at this scale.
- Verifies persona anchors stay at the trusted head and RAG remains a background hint.
- Cleans the database and report state on exit.

```bash
.venv\Scripts\python.exe -u examples/stress_test_100k.py
```

The report is written to `test_result_100k.md`.

## `hybrid_test_10k.py`

The 10,000-memory real-vector hybrid retrieval test:

- Real `all-MiniLM-L6-v2` embeddings and cosine HNSW.
- Measures pure `search_vector()` and hybrid `search_memory()` separately.
- Explicit entities such as `密码` trigger the Chroma `$contains` keyword fallback; no
  full vector scan is used.
- Includes persona-anchor ordering and prompt token-budget checks.

```bash
.venv\Scripts\python.exe -u examples/hybrid_test_10k.py
```

The report is written to `test_result_10k_hybrid.md`.

## `hybrid_test_10k_conflicts.py`

The multi-version conflict test:

- Writes three password versions at memory positions 100, 5,000, and 9,000.
- Each write stores a strictly increasing millisecond `timestamp_ms`.
- `search_memory()` groups same-entity/high-similarity hits and applies latest-wins.
- Verifies that only `07210721` reaches the result and prompt; older `114514`/`1919810`
  versions stay out of the RAG hint.
- Reports pure HNSW and timestamp-aware hybrid latency separately.

```bash
.venv\Scripts\python.exe -u examples/hybrid_test_10k_conflicts.py
```

The report is written to `test_result_10k_multiversion.md`.

## `hybrid_test_100k_real.py`

The full-scale real-vector conflict test:

- 100,000 memories embedded with real `all-MiniLM-L6-v2`.
- 2,000-record batches with a progress bar to avoid OOM.
- Three password versions at 100, 50,000, and 99,000.
- Pure HNSW latency and timestamp-aware hybrid retrieval are reported separately.
- The newest version must be the only password in the result and prompt.

```bash
.venv\Scripts\python.exe -u examples/hybrid_test_100k_real.py
```

The report is written to `test_result_100k_real.md`.

## `chat_demo.py`

A persistent interactive chat demo:

- Chroma storage at `./chat_memory_db` (not a temporary directory).
- First run preloads 50,000 daily/preference memories with a fast local hash embedding.
- Each chat turn calls `search_memory()` with hybrid retrieval, renders the prompt, prints
  retrieved memories, and writes the user message back to Chroma.
- Commands: `/prompt` shows the last rendered prompt; `/exit` leaves.
- Filters historical user questions from retrieved hits; Mock replies join all retrieved
  `system_fact` entries and prioritize drink-related facts.

```bash
.venv\Scripts\python.exe -u examples/chat_demo.py
# optional real DeepSeek client (DeepSeek base_url/model are the defaults)
.venv\Scripts\python.exe -m pip install -e ".[llm]"
$env:DEEPSEEK_API_KEY="..."
.venv\Scripts\python.exe -u examples/chat_demo.py --llm
```
