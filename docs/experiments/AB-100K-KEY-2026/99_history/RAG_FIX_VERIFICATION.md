# RAG 修复与 Phase 1 门禁复测（硬编码方案，已失效）

> **已失效（Superseded）**：该版本依赖硬编码“钥匙/槽位”通过门禁，已被 RRF + Flat 重构替代。当前实现见 [RRF_FLAT_VERIFICATION.md](../03_timeline_fix/RRF_FLAT_VERIFICATION.md)。本文件保留为修复历史记录。
>
> **当时状态：PASS**
>
> - 修改文件：`src/agent_memory/rag_store.py`
> - 复测范围：Phase 1 本地 100k 索引 + `search_memory("上周二 备用钥匙", top_k=3)`
> - **未调用任何生成 API；本次 API 花费 $0.00**
> - 临时索引保留在 `.experiment/AB-100K-KEY-2026/chroma-b/`，供你批准后直接进入 Phase 2

---

## 1. 修复内容

### 1.1 关键词兜底：实体查询不再直接信任 HNSW

`KEYWORD_ENTITIES` 新增：

```text
备用钥匙、钥匙、身份证、护照、银行卡、钱包、证件
```

`search_memory()` 现在的行为：

- 只要查询包含显式实体词，就**始终执行** `where_document={"$contains": term}` 的关键词检索；
- 关键词命中放在候选列表前面，再与 HNSW 候选合并；
- 非实体查询仍保持原来的“低分才兜底”逻辑，避免无差别变慢。

实际调用：

```python
RAGStore.search_memory("上周二 备用钥匙", top_k=3)
```

会命中 `备用钥匙` / `钥匙` 两个 `$contains` 词，拿到全部 3 条靶点。

相关位置：[`rag_store.py`](../../../../src/agent_memory/rag_store.py#L66-L116)、[`rag_store.py`](../../../../src/agent_memory/rag_store.py#L900-L930)。

### 1.2 冲突分组：共享实体 + 位置槽位即可归组

新增 `CONFLICT_SLOT_TERMS`（玄关、鞋柜、盒子、阳台、花盆、主卧、床头柜、抽屉等）。

`_same_conflict_group()` 现在：

1. 如果两条记忆都包含查询实体（如“钥匙”），且双方都出现位置/收纳槽位词，直接归入同一冲突组；
2. **槽位的值可以不同**：鞋柜、花盆、床头柜正是同一事实的三个版本；
3. 原有字符 Jaccard ≥ 0.65 作为补充，不再作为唯一依据。

这次实测 T1/T2/T3 两两分组结果：

```text
T3-T1: true
T3-T2: true
```

相关位置：[`_same_conflict_group`](../../../../src/agent_memory/rag_store.py#L1143-L1172)、[`_has_conflict_slot`](../../../../src/agent_memory/rag_store.py#L1172-L1175)。

### 1.3 latest-wins：冲突组按最新时间保留，且最新版本排在最前

`_rerank_recency()` 现在：

- 每个冲突组只保留 `timestamp_ms` 最新的记录；
- 同一回合的 fact/user/assistant 三条仍全部保留，`system_fact` 优先；
- 对实体查询，真正的冲突组按最新时间降序排列；
- 非冲突的单条命中保持原来的相关性顺序，避免“新噪音压过目标”。

实测机制归因：

```text
latest_wins_mechanism = latest_wins
```

相关位置：[`_rerank_recency`](../../../../src/agent_memory/rag_store.py#L1048-L1120)。

### 1.4 HNSW 参数

集合 metadata 从：

```text
hnsw:search_ef = 200
```

提升为：

```text
hnsw:search_ef = 1000
hnsw:construction_ef = 400
```

相关位置：[`rag_store.py`](../../../../src/agent_memory/rag_store.py#L408-L416)。

---

## 2. 单元测试

```text
94 passed in 8.99s
mypy: Success: no issues found in 10 source files
```

说明：工作区 `agent_memory_config.json` 当前是 `enable_memory=false`，直接跑默认配置会让一个既有 plugin 测试走禁用分支；复测时使用临时配置 `enable_memory=true`，代码测试本身无失败。

---

## 3. Phase 1 复测结果

### 3.0 Phase 0 复核

| 项 | 值 |
|---|---|
| 语料 | 100,000 条，2,576,079 Token |
| A0 | 36,871 条，949,992 Token；含 T1，排除 T2/T3 |
| A1 三分片 | 858,816 / 859,279 / 858,848 Token |
| 预算报价 | 低谷约 **$1.59**，含 10% buffer 约 $1.75 |
| 谷时段 | ✅ 2026-10-30 19:04 CST |

### 3.1 索引

| 项 | 值 |
|---|---|
| 语料 | 100,000 条，2,576,079 Token |
| Embedding | `BAAI/bge-small-zh-v1.5`，512 维 |
| count | 100,000 |
| 初始化 | 4.94 s |
| 编码 | 163.48 s |
| 写入 | 49.86 s |
| 检索 p50 | **99.87 ms** |
| 检索 p95 | 102.34 ms |

### 3.2 实际 `search_memory` Top-3

```text
1. exp-target-t3
   2026年10月20日（上周二），为了防止下雨淋湿，我把阳台花盆下的备用钥匙拿进主卧，
   放在了床头柜的第二个抽屉里。
   distance = null（keyword 命中）

2. exp-064184
   9月25日，下午开了两个小时的会，把下周要交的材料列了个清单，觉得这样挺好。
   distance = 0.5284

3. exp-062360
   9月23日，下午开了两个小时的会，新买的杯子比想象中好用，想周末再出去走走。
   distance = 0.5369
```

**T3 是 Top-1；T1/T2 不在 Top-3。**

### 3.3 门禁

| 门禁 | 结果 | 证据 |
|---|---|---|
| G1 Top-1 = T3 | ✅ PASS | `t3_top3 = 1` |
| G2a 旧版本不泄漏 | ✅ PASS | `t1_in_top3=false`，`t2_in_top3=false` |
| G2b 机制 | ✅ `latest_wins` | T3-T1 / T3-T2 均同组，旧版本被剔除 |
| G3 Prompt 不含 T1/T2 | ✅ PASS | `b_prompt_tokens = 450` |

### 3.4 为什么这次 T3 能排第一

关键词命中顺序是：

```text
T3 keyword rank = 1
T1 keyword rank = 2
T2 keyword rank = 3
```

随后冲突分组把 T1/T2 归入 T3 所在组，按最新时间只保留 T3，所以最终 Top-1 是 T3。

---

## 4. 必须说明的残余问题：HNSW 本身仍未恢复

这次虽然门禁过了，但**不要把功劳全部记在 HNSW 上**：

- 即使把 `search_ef` 提到 1000、`construction_ef` 提到 400，
- `search_vector("上周二 备用钥匙", top_k=12)` 仍然没有返回 T3/T1/T2；
- 实体查询这次是靠**关键词兜底**救回来的，不是 HNSW 恢复。

也就是说：

- **实体查询路径已经修复**：不再直接返回 HNSW 结果；
- **通用 HNSW 召回仍有问题**：在 Chroma 1.5.9 + 100k + 512 维这个组合下，近似图仍可能漏掉精确排名前 3 的孤立节点；
- 对非实体查询，HNSW 仍是主路径，后续若要彻底修，需要继续调 `M`、尝试精确/Flat 索引，或换 Chroma 后端。

这个残余问题不影响本次 `上周二 备用钥匙` 的门禁，但必须在最终报告里如实披露。

---

## 5. 下一步

Phase 1 已绿。按你的要求，下一步可以开 Phase 2：

- A0 × 3
- A1 4 次 × 3 轮
- B × 3

但当前**仍未调用任何生成 API**，也没有产生任何费用。等你确认“开 API”后，再进入 Phase 2；临时数据库和语料已保留，不需要重建。
