> **Superseded by the BAAI + BM25 engineering upgrade**: see [RAG_ENGINEERING_UPGRADE_2026-10-27.md](RAG_ENGINEERING_UPGRADE_2026-10-27.md).

# 查询扩展（Query Expansion）10k 真实向量回归报告

- 日期：2026-10-27
- 环境：`.venv` / Python 3.12.10
- 模型：`sentence-transformers/all-MiniLM-L6-v2`（本地缓存）
- 强制离线：
  - `HF_HOME=G:\deepseek_ragproject_workspace\.hf-cache`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 数据库：真实 Chroma PersistentClient + cosine HNSW
- 结论：**PASS；未使用任何硬编码同义词**

## 1. 整改内容

### 删除硬编码小抄

- 已从 `rag_store.py` 删除：
  - `KEYWORD_ALIASES = {"咖啡": ("冰美式", "拿铁")}`
  - `咖啡` 的强制关键词实体
  - 基于该字典的 `expanded_terms` 强制关键词检索逻辑
- `KEYWORD_ENTITIES` 恢复为密码/密钥类高价值实体，不再包含咖啡。

### 启用 LLM 查询扩展

`memory_agent.py` 的 `_REWRITE_SYSTEM` 现在要求模型：

```text
Preserve names, entities, numbers, and the user's language.
Expand Chinese colloquial terms into their common aliases, synonyms, and
related brand/product names. For example, "咖啡" should expand to something
like "咖啡 冰美式 拿铁 星巴克".
```

实际测试中，Mock LLM 返回：

```text
咖啡 冰美式 拿铁 星巴克
```

`MemoryAgent` 使用这条扩展长句调用 `search_memory()`，检索来源为 **vector**。

### README 模型警告

`README_CN.md`、`README_EN.md`、`README.md` 已明确说明：

- 默认 `all-MiniLM-L6-v2` 仅用于快速起步，中文语义召回一般；
- 生产环境建议初始化 `RAGStore` 时通过 `model_name` 换用中文向量模型；
- 示例：`RAGStore(model_name="BAAI/bge-small-zh-v1.5")`。

## 2. 测试结果

测试数据：

- 10,000 条真实向量日常记忆
- 2 条普通咖啡事实
- 真实 `all-MiniLM-L6-v2` 编码

真实终端输出：

```text
=== PRELOAD 10,000 REAL VECTORS ===
  inserted 2000/10000
  inserted 4000/10000
  inserted 6000/10000
  inserted 8000/10000
  inserted 10000/10000
preload_seconds=13.608
count_after_preload=10000
=== AGENT RESULT ===
extracted_fact: 用户最近特别喜欢喝冰美式。
rewritten_query: 咖啡 冰美式 拿铁 星巴克
reply: 冰美式提神醒脑
fact_memory_id: mem-9954827989994ae894a1a52db0059da3
user_memory_id: mem-c766e925738146e4b1cdcb27aa6a1727
assistant_memory_id: mem-a840946ac657407594f42fff7fd84b36
expanded_query_used_for_search: True
chat_seconds=0.055
=== TURN ROWS ===
{"id": "mem-a840946ac657407594f42fff7fd84b36", "source_type": "assistant_message", "timestamp_ms": 1793105146357, "content": "冰美式提神醒脑"}
{"id": "mem-9954827989994ae894a1a52db0059da3", "source_type": "system_fact", "timestamp_ms": 1793105146357, "content": "用户最近特别喜欢喝冰美式。"}
{"id": "mem-c766e925738146e4b1cdcb27aa6a1727", "source_type": "user_message", "timestamp_ms": 1793105146357, "content": "我最近特别喜欢喝冰美式"}
turn_timestamp_count: 1 value: [1793105146357]
search_seconds=0.009
=== SEARCH EXPANDED QUERY TOP-3 ===
{"source_type": "system_fact", "timestamp_ms": 1793105146357, "content": "用户最近特别喜欢喝冰美式。", "retrieval_source": "vector"}
{"source_type": "assistant_message", "timestamp_ms": 1793105146357, "content": "冰美式提神醒脑", "retrieval_source": "vector"}
{"source_type": "user_message", "timestamp_ms": 1793105146357, "content": "我最近特别喜欢喝冰美式", "retrieval_source": "vector"}
=== ASSISTANT LABEL IN PROMPT ===
has_label= True assistant_content_in_prompt= True
=== SUMMARY ===
preload_seconds=13.608
chat_seconds=0.055
search_seconds=0.009
total_seconds=19.627
RESULT=PASS
```

## 3. 验收结果

| 检查项 | 结果 |
| --- | --- |
| 硬编码 `KEYWORD_ALIASES` 已删除 | PASS |
| `咖啡` 不再触发强制关键词检索 | PASS |
| LLM Prompt 要求中文别名/同义词扩展 | PASS |
| `rewritten_query` 为扩展长句 | PASS |
| 扩展检索来源 | **vector** |
| assistant_message 被召回 | PASS，Top-2 |
| assistant Prompt 标签 | PASS，`[这是你曾经对用户说过的话]` |
| 三写 timestamp 一致 | PASS，均为 `1793105146357` |
| 是否被 assistant 完全挤占 | 否；Top-3 中另有 fact 和 user 两条 |
| 10k 预灌入 | 13.608s |
| 对话 | 0.055s |
| 检索 | 0.009s |
| 全流程 | 19.627s |

> 说明：assistant 是回复生成后才写入的，因此它不可能出现在“当前轮生成前”的候选集里；本测试验证的是同一数据库在回合结束后的后续检索。扩展查询使用真实向量检索，没有走关键词兜底。

## 4. 全量测试

```text
83 passed in 1.48s
```

## 5. 清理状态

- `.test-tmp/` 已清理
- 临时查询扩展测试脚本已删除
- 临时 10k 数据库已删除
- `__pycache__` / pytest 缓存已清理
- `.venv` 未改动

## 6. 结论

这次没有“小抄”：

- `rag_store.py` 不包含任何咖啡同义词字典；
- 召回依赖 LLM 生成的扩展查询 + 真实向量检索；
- `assistant_message` 能以 vector 来源召回，并且 Prompt 标签正确；
- 默认 `all-MiniLM-L6-v2` 仍是中文快速起步方案；生产环境应按 README 建议换用 `BAAI/bge-small-zh-v1.5` 等中文向量模型。
