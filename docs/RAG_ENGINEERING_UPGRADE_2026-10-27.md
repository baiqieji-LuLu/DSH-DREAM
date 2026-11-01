# RAG 工程化升级报告（BAAI + BM25 + 自动回退）

- 日期：2026-10-27
- 环境：`.venv` / Python 3.12.10
- 状态：**PASS**

## 1. 默认模型升级

`src/agent_memory/rag_store.py`：

```python
_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_FALLBACK_MODEL = "all-MiniLM-L6-v2"
```

`RAGStore()` 现在默认使用中文优化的 `BAAI/bge-small-zh-v1.5`。

补充的 BGE 查询处理：

- 对 BGE 系列自动加中文 query instruction：
  `为这个句子生成表示以用于检索相关文章：`
- 存储侧文档不加 instruction，避免污染文档向量。
- 默认 HNSW `search_ef=200`，可通过 `index_metadata` 覆盖。

README 已改为正向推荐：

- 默认 BAAI 中文模型；
- 纯英文环境可显式切回 `all-MiniLM-L6-v2`；
- BAAI 本地不可用时自动回退。

## 2. BM25 Rerank 兜底

新增依赖：

```toml
"rank-bm25>=0.2.2"
```

`search_memory()` 的检索顺序：

1. 真实向量 HNSW 召回至少 Top-10 候选；
2. 用 `rank_bm25.BM25Okapi` 对候选做词法重排；
3. BM25 只做软提权，不直接覆盖向量分数：

```text
combined = vector_score + 0.35 * normalized_bm25
```

4. 显式实体查询（密码/密钥等）仍保留关键词候选优先 + latest-wins；
5. 最后截断到调用方 `top_k`。

中文 BM25 使用 **bigram 分词**，避免“冰”把“冰箱”误判成“冰美式”：

```text
冰美式 -> 冰美 / 美式
冰箱   -> 冰箱
```

## 3. 自动回退

- `RAGStore.__init__` 创建 embedding 失败时，自动使用 `all-MiniLM-L6-v2`。
- `RAGStore.ensure_embedding_ready()` 在本地模型缺失时切换 fallback。
- `MemoryAgent.__init__` 调用 `ensure_embedding_ready()`，捕获异常并继续，不因模型缺失崩溃。
- 显式注入的 embedding function 不会被替换。

实际探测：

```text
default => BAAI/bge-small-zh-v1.5
missing => all-MiniLM-L6-v2
```

## 4. 10k 旧版密码冲突回归

`examples/hybrid_test_10k_conflicts.py`（脚本显式使用 MiniLM）真实终端：

```text
batch insert done: 10000 memories in 13.200s
version timestamps: {100: 1793106359660, 5000: 1793106359863, 9000: 1793106366177}
vector_ms=10.616 vector_hit='主人的银行卡密码是 1919810'
hybrid_ms=14.224 source=keyword hybrid_hit='主人的银行卡密码是 07210721'
old_in_prompt=False
prompt_tokens=645
temp_cleaned=True
overall=PASS
```

验收：

| 项目 | 结果 |
| --- | --- |
| 10,000 条写入 | 13.200s |
| 纯 HNSW | 10.616 ms |
| 混合检索 | 14.224 ms |
| < 50ms | PASS |
| latest-wins | PASS，`07210721` |
| 旧版本进入 Prompt | 否 |

## 5. 10k BAAI + BM25 + 查询扩展测试

真实终端：

```text
=== EMBEDDING MODEL ===
BAAI/bge-small-zh-v1.5
=== PRELOAD 10,000 REAL VECTORS ===
  inserted 2000/10000
  inserted 4000/10000
  inserted 6000/10000
  inserted 8000/10000
  inserted 10000/10000
preload_seconds=13.841
count_after_preload=10000
=== AGENT RESULT ===
extracted_fact: 用户最近特别喜欢喝冰美式。
rewritten_query: 咖啡 冰美式 拿铁 星巴克
reply: 冰美式提神醒脑
fact_memory_id: mem-42e945e2f4914e65b6222289ad23b4b3
user_memory_id: mem-06284c10e68d4f91932d0509a98dc7d4
assistant_memory_id: mem-6b0d11701bcb4def872b65b9a6589853
expanded_query_used_for_search: True
chat_seconds=0.061
=== TURN ROWS ===
{"id": "mem-6b0d11701bcb4def872b65b9a6589853", "source_type": "assistant_message", "timestamp_ms": 1793106299723, "content": "冰美式提神醒脑"}
{"id": "mem-42e945e2f4914e65b6222289ad23b4b3", "source_type": "system_fact", "timestamp_ms": 1793106299723, "content": "用户最近特别喜欢喝冰美式。"}
{"id": "mem-06284c10e68d4f91932d0509a98dc7d4", "source_type": "user_message", "timestamp_ms": 1793106299723, "content": "我最近特别喜欢喝冰美式"}
turn_timestamp_count: 1 value: [1793106299723]
search_seconds=0.010
=== SEARCH EXPANDED QUERY TOP-3 ===
{"source_type": "user_message", "timestamp_ms": 1793106299723, "content": "我最近特别喜欢喝冰美式", "retrieval_source": "vector"}
{"source_type": "assistant_message", "timestamp_ms": 1793106299723, "content": "冰美式提神醒脑", "retrieval_source": "vector"}
{"source_type": "system_fact", "timestamp_ms": 1793106299723, "content": "用户最近特别喜欢喝冰美式。", "retrieval_source": "vector"}
=== ASSISTANT LABEL IN PROMPT ===
has_label= True assistant_content_in_prompt= True
=== SUMMARY ===
preload_seconds=13.841
chat_seconds=0.061
search_seconds=0.010
total_seconds=19.899
RESULT=PASS
```

验收：

| 项目 | 结果 |
| --- | --- |
| 默认模型 | BAAI/bge-small-zh-v1.5 |
| 10,000 条真实向量 | PASS |
| 三写 timestamp 一致 | PASS，`1793106299723` |
| assistant 召回 | PASS，Top-2 |
| 检索来源 | vector |
| assistant 标签 | PASS |
| 其他记忆 | Top-3 同时保留 user 和 fact |
| BM25 | 在 Top-10 候选上软提权 |
| 查询扩展 | `咖啡 冰美式 拿铁 星巴克` |

## 6. 全量单元测试

```text
83 passed in 1.57s
```

`pip check`：

```text
No broken requirements found.
```

## 7. 清理

- `.test-tmp/` 已删除
- 临时脚本与临时 10k 数据库已删除
- 10k 密码报告已归档到 `docs/test_result_10k_multiversion.md`
- `__pycache__` / pytest 缓存已清理
- `.venv` 保留，`rank-bm25` 已安装
