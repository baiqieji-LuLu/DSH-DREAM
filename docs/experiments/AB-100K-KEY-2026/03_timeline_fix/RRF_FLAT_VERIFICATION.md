# RRF + Flat 重构验证报告

> **状态：PASS**
>
> - 修改文件：`src/agent_memory/rag_store.py`
> - 验证规模：100,000 条真实 BAAI 向量
> - **未调用任何生成 API；本次 API 花费 $0.00**
> - 临时 Flat 索引已删除，语料与验证数据保留

---

## 1. 做了什么

### 1.1 删除硬编码实体词与槽位词

`KEYWORD_ENTITIES`、`CONFLICT_SLOT_TERMS`、`ENTITY_VECTOR_CONFIDENCE` 以及
`_explicit_keyword_terms` / `_keyword_search` / `_keyword_relevance` /
`_has_conflict_slot` 全部移除。

代码中不再出现“钥匙/鞋柜/花盆/主卧/床头柜/玄关”等业务词。

### 1.2 RRF 混合检索

`search_memory()` 现在对每个查询同时执行：

1. **向量检索**：100k 时走 Flat 精确余弦；超过阈值才走 HNSW。
2. **BM25 全文检索**：对全部 100k 文档建立一次 BM25 索引，不是候选窗口重排。
3. **RRF 融合**：

```text
score(doc) = Σ 1 / (RRF_K + rank_i)
RRF_K = 60
```

4. 融合结果再进入通用的 latest-wins 冲突处理。

没有“如果包含 XXX 就走关键词”的分支。

### 1.3 Flat 精确索引

`RAGStore` 新增构造参数：

```python
RAGStore(..., flat_max_records=200_000)
```

当 `count() < flat_max_records` 时：

- 从 Chroma 一次性读取全部 `ids/documents/metadatas/embeddings`；
- 对 embedding 做 L2 归一化；
- 查询时用 `numpy` 做全量矩阵乘法；
- 按 cosine 距离排序，召回率 100%；
- 不使用 HNSW。

`flat_max_records=0` 可显式关闭 Flat。

### 1.4 泛化冲突分组

`_same_conflict_group()` 不再依赖任何实体/槽位字典，改用三个通用信号：

1. 内容二元词 + ASCII 词的**重合系数 ≥ 0.30**；
2. 两条记录共享长度 **≥ 4** 的中文连续子串；
3. 字符集 Jaccard ≥ 0.65（高相似度兜底）。

`_rerank_recency()` 在每个冲突组内只保留最新 `timestamp_ms`；组在 RRF 融合结果中的相对位置保持不变。

---

## 2. 本地 100k 验证

### 2.1 构建

| 项 | 值 |
|---|---|
| 语料 | 100,000 条 |
| Embedding | `BAAI/bge-small-zh-v1.5`，512 维 |
| 编码 | 167.9 s |
| 写入 | 44.6 s |
| count | 100,000 |
| Flat 阈值 | 200,000（本次启用） |

### 2.2 Flat 精确召回

`search_vector("上周二 备用钥匙", top_k=12)`：

| 排名 | ID | distance | 内容 |
|---:|---|---:|---|
| 1 | exp-target-t1 | 0.4396 | 2026年7月15日……玄关鞋柜的蓝色盒子 |
| 2 | **exp-target-t3** | **0.4868** | 2026年10月20日（上周二）……主卧床头柜第二个抽屉 |
| 3 | exp-target-t2 | 0.5107 | 2026年9月10日……阳台花盆下面 |

Flat 精确排序是 **T1 > T3 > T2**，说明向量本身没有丢靶点；旧 HNSW 的问题被绕开。

### 2.3 RRF 融合

RRF 融合后的前 3：

```text
1. T1（hybrid）
2. T3（hybrid）
3. T2（hybrid）
```

**RRF 单独仍会把旧版本 T1 排在 Top-1。** 这是预期行为：RRF 只负责融合召回，不负责时间冲突。

### 2.4 latest-wins 后

通用冲突分组结果：

```text
T3-T1: true
T3-T2: true
```

冲突组只保留最新 T3，最终：

```text
1. exp-target-t3  [hybrid]  distance=0.4868
   2026年10月20日（上周二），为了防止下雨淋湿，我把阳台花盆下的备用钥匙
   拿进主卧，放在了床头柜的第二个抽屉里。

2. exp-093252     [vector]  distance=0.5524
   10月23日，下午开了两个小时的会……

3. exp-005729     [vector]  distance=0.5498
   2月26日，周末把房间从头到尾打扫了一遍……
```

**T3 是 Top-1；T1/T2 不在 Top-3。**

### 2.5 延迟

| 路径 | 中位耗时 |
|---|---:|
| Flat 精确向量检索（warm） | **23.89 ms** |
| RRF 混合检索（warm） | **167.93 ms** |
| RRF 混合检索（cold，Flat 缓存已建、首次建 BM25） | 约 **1.29 s** |
| Flat 缓存首次加载（cold） | 约 **6.29 s** |

Flat 的 100k 矩阵缓存首次加载约 6.3 s；BM25 首查约 1.3 s；warm 查询稳定在约 168 ms。

---

## 3. 门禁

| 门禁 | 结果 |
|---|---|
| G1：T3 Top-1 | ✅ PASS |
| G2a：T1/T2 不进入 Top-3 | ✅ PASS |
| 无硬编码实体/槽位 | ✅ 代码中已无相关常量与分支 |
| Flat 启用 | ✅ `count=100000 < 200000` |
| 单测 | ✅ 94 passed |
| mypy | ✅ Success: no issues found in 10 source files |

---

## 4. 残余说明

- Flat 的精确召回是 **100%**，但代价是内存：100k × 512 float32 约 200 MB 向量矩阵，另加文档/metadata 与 BM25 结构。
- RRF 的 BM25 首查建索引约 6 秒；这是每进程一次性的，不是每次查询都付。
- 100k 时不再使用 HNSW；超过 200k 才会回退 HNSW。若未来超过 200k，仍需单独验证 HNSW 召回。
- 最终 Top-3 仍会带 2 条无关噪音；T1/T2 已剔除，T3 在 Top-1。B Prompt 仍很短。
- 本次只做本地验证，**没有重跑 Phase 2 生成 API**；先前的 A/B 费用与回答属于上一版实现，不应与本次 RRF/Flat 结果混为一谈。

---

## 5. 复现

```powershell
.venv\Scripts\python.exe experiments\ab_100k\verify_rrf_flat.py
```

脚本会重建临时 Flat 索引、跑 100k 召回与 RRF，并输出 `rrf_flat_report.json`。
