# latest-wins / Timeline Mode 修复验证

> **状态：PASS（本地验证，无生成 API）**
>
> 修改文件：`src/agent_memory/rag_store.py`、`src/agent_memory/memory_agent.py`
>
> 新增测试：无新增 API 调用；仅本地 100k + 100 条索引复测。

---

## 1. 改了什么

### 1.1 冲突分组：只看状态槽位，不再看“共享名词”

`_same_conflict_group()` 不再用“共享蓝色马克杯”或字符 Jaccard 直接归组。

新逻辑：

1. 从文本中提取状态谓词槽位：
   - 位置类：`放在 / 放到 / 塞在 / 藏在 / 位于 ...`
   - 状态类：`是 / 改成 / 换成 / 设置为 ...`
2. 只有**同一谓词族 + 值不同**，且两条记录共享实体短语时，才判为冲突组。
3. “买了一个杯子 / 摔碎了 / 又买了一个补上”没有状态槽位，因此**不会被 latest-wins 合并**。

密钥示例仍然成立：

```text
钥匙放在鞋柜  vs. 钥匙放在花盆  vs. 钥匙放在床头柜
=> 同一 location 谓词，值不同，共享“钥匙” => 冲突组
```

马克杯示例不再成立：

```text
买了一个蓝色马克杯 / 摔碎了 / 又买了一个补上
=> 没有状态槽位 => 不是冲突组
```

### 1.2 `timeline_mode`

`RAGStore.search_memory(..., timeline_mode=True)`：

- 跳过 `_rerank_recency()`；
- 取 RRF 融合后的 Top-K；
- 按 `timestamp_ms` 升序返回；
- `search()` 别名同步支持该参数。

### 1.3 Agentic 意图判断

`MemoryAgent._rewrite_query()` 的 LLM Prompt 改为返回：

```json
{"keywords": "...", "intent": "latest|timeline"}
```

- `intent=timeline` 时，`_last_timeline_mode=True`；
- `chat_with_memory()` 将该标志传给 `_retrieve()`；
- `_retrieve()` 调用 `store.search_memory(..., timeline_mode=True)`；
- LLM 重排后的 `selected` 再按 `timestamp_ms` 升序排序，保证 Prompt 里的时间线顺序；
- 旧后端不支持该参数时，仅在该模式下回退到普通检索（兼容测试 double）。

---

## 2. 本地复测结果

### 2.1 D 组：100 条蓝色马克杯

| 调用 | 结果 |
|---|---|
| `search_memory("星巴克 蓝色马克杯", top_k=3)` | T1、T2、T3 全部返回 |
| `search_memory(..., timeline_mode=True)` | **T1 → T2 → T3**（timestamp 升序） |

实际 ID 顺序：

```text
trivia-010  (买)
trivia-050  (摔碎)
trivia-090  (补买)
```

### 2.2 B 组：100k 备用钥匙

| 调用 | 结果 |
|---|---|
| `search_memory("上周二 备用钥匙", top_k=3)` | **T3 Top-1**；T1/T2 不在 Top-3 |
| `search_memory(..., timeline_mode=True)` | **T1 → T2 → T3**（timestamp 升序） |

默认模式仍然剔除旧版本；时间线模式则保留完整历史。两者语义已分开。

### 2.3 Agentic 意图

用 fake LLM 返回 `{"keywords":"...","intent":"timeline"}`：

```text
_rewrite_query => "星巴克 蓝色马克杯"
_last_timeline_mode => True
_retrieve kwargs => {"top_k": 10, "min_vector_score": 0.35, "timeline_mode": True}
```

### 2.4 10k 最终测试

在 10,000 条语料上，目标位于第 2,500 / 5,000 / 7,500 条；为避免标准 100 条分组把靶点放在末尾，另用中心窗口 2451–2550、4951–5050、7451–7550 生成 3 篇中心日记，并将日记限制为 200 字。

结果：

- 100 篇标准日记中，只有第 50、75 篇提到杯子；
- 3 篇中心日记中，只有第 50 篇提到“摔碎了一个蓝色马克杯”；第 25、75 篇完全没提；
- C 查询无法还原品牌、购买、补买；
- RAG `timeline_mode=True` 的 Top-5 命中 T1/T2/T3，回答完整还原“买→摔碎→补买”。

原始数据：[trivia_10k_v2_report.json](trivia_10k_v2_report.json)

### 2.5 测试

```text
94 passed
mypy: Success: no issues found in 10 source files
```

---

## 3. 重要边界

- 本地检索验证之外，已补跑 D 组 `timeline_mode` 实弹生成：100 条探针（Input 150 / Output 86，$0.000074，1671 ms）与 10k 最终测试（Input 170 / Output 112，$0.000093，1208 ms）均完整还原时间线。
- `EXPERIMENT_FINAL_COMPARISON.md` §6.6.3 已替换为修复后的新回答；修复前的 `search_memory` 失败记录保留在报告的历史说明中。
- B 组 100k 的默认/时间线两种模式都已本地验证；B 组历史 API 回答仍是默认模式下的结果。

---

## 4. 代码位置

- [`rag_store.py`](../../../../src/agent_memory/rag_store.py)：`STATE_SLOT_FAMILIES`、`_state_slots()`、`_same_conflict_group()`、`search_memory(..., timeline_mode=...)`
- [`memory_agent.py`](../../../../src/agent_memory/memory_agent.py)：`_REWRITE_SYSTEM`、`_rewrite_query()`、`_retrieve(timeline_mode=...)`、`MemoryAgentResult.timeline_mode`
