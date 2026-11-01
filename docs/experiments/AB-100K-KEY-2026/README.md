# AB-100K-KEY-2026：A/B/C/D 记忆实验总览

> 一个 100,000 条合成记忆、约 2.58M Token 的对照实验，用来回答：
> **当记忆总量远超单次上下文时，RAG、日记摘要、裸跑分别能记住什么？**

## 最终结论（一页纸）

| 组 | 方法 | 结果 | 结论 |
|---|---|---|---|
| A0 | 1M 截断裸跑 | 0/3 正确；2 次 UNKNOWN，1 次把旧版“鞋柜”当答案 | 截断会丢失最新记忆 |
| A1 | 3 分片 + 汇总裸跑 | 0/3 正确；9 次分片抽取全部 `NO_MATCH` | 分片全量也没有恢复靶点 |
| B | RRF + Flat Top-3 | 3/3 正确；T3 Top-1，T1/T2 被剔除 | 100k 下成本/延迟/精度综合最优 |
| C | 日记法（3 篇摘要，100k 主实验） | 3/3 正确 | 靶点恰好是分组最后一条，不能据此证明日记法普遍可靠 |
| D-100 | 100 条蓝色马克杯探针 | 修复前 RAG `search_memory` 0/1；修复后 `timeline_mode` 1/1 | latest-wins 时间线缺陷已修 |
| D-10k | **10,000 条蓝色马克杯（最终）** | **日记 0/1；RAG `timeline_mode` 1/1** | **200 字日记丢失品牌/购买/补买；RAG 完整还原 T1→T2→T3** |

**关键结论**：

1. **A0/A1 的失败是真实的**：1M 截断和分片 map-reduce 都没有保住 T3。
2. **B 用 450 Token 左右解决了问题**：RRF + Flat 在 100k 下不必依赖 HNSW 的近似召回。
3. **C 组不能证明“日记法必然有损”**：T1/T2/T3 都是 100 条分组的最后一条，摘要复述了“最后一条记录”。这是位置偏差。
4. **D-10k 是真正的有损压缩测试**：靶点居中、每篇日记 200 字后，三篇中心日记只剩“摔碎了一个蓝色杯子”，品牌/购买/补买全部丢失；RAG `timeline_mode` 直接返回三条原文，完整还原“买→摔碎→补买”。

## 实际账单

| 项 | 数值 |
|---|---:|
| 正式对比调用 | 134 次 |
| 实际计费调用 | 241 次（含 C pilot、10k pilot） |
| 总实际花费 | **$1.662004 / ¥11.9664** |
| 硬上限 | $3.50 / ¥25.00 |

模型：`deepseek-flash`（DeepSeek-V4.1-Flash-1017）。所有价格按 2026-10-30 低谷价计算。

## 目录

```text
docs/experiments/AB-100K-KEY-2026/
├── README.md
├── 01_design/
│   └── EXPERIMENT_DESIGN.md
├── 02_main_benchmark/
│   ├── EXPERIMENT_FINAL_COMPARISON.md
│   ├── EXPERIMENT_LOG_10K.md
│   ├── C_DIARIES.md
│   └── EXPERIMENT_USAGE.csv
├── 03_timeline_fix/
│   ├── TIMELINE_FIX_VERIFICATION.md
│   ├── RRF_FLAT_VERIFICATION.md
│   ├── trivia_report.json
│   ├── trivia_10k_v2_report.json
│   ├── trivia_10k_corpus.jsonl
│   └── trivia_10k_diaries.jsonl
└── 99_history/
    ├── EXPERIMENT_GATE_FAILURE.md
    ├── EXPERIMENT_GATE_EVIDENCE.json
    ├── RAG_FIX_VERIFICATION.md
    └── trivia_10k_pilot_report.json
```

## 阅读顺序

1. [实验设计](01_design/EXPERIMENT_DESIGN.md)
2. [最终对比报告](02_main_benchmark/EXPERIMENT_FINAL_COMPARISON.md)
3. [C 组日记原文](02_main_benchmark/C_DIARIES.md)
4. [10k 过程审计日志](02_main_benchmark/EXPERIMENT_LOG_10K.md)
5. [D 组时间线修复验证](03_timeline_fix/TIMELINE_FIX_VERIFICATION.md)
6. [10k 最终原始数据](03_timeline_fix/trivia_10k_v2_report.json)
7. [RRF + Flat 架构验证](03_timeline_fix/RRF_FLAT_VERIFICATION.md)
8. 历史翻车记录：[门禁失败](99_history/EXPERIMENT_GATE_FAILURE.md)、[硬编码修复](99_history/RAG_FIX_VERIFICATION.md)、[10k 前导实验](99_history/trivia_10k_pilot_report.json)

## 重要边界

- N=3，主实验只做描述统计，不做显著性检验。
- A0 是人为 1M 截断，不是“模型注意了全部 2.5M Token”。
- C 主实验的靶点恰好是 100 条分组的最后一条，存在位置偏差；D-10k 用中心窗口专门消除了这一点。
- D-10k 使用 10,000 条**合成**日常记录，每篇日记限制 200 字；真实用户日志的损失程度可能不同。
- D-10k 的 RAG 回答基于 `timeline_mode=True` 的单次实弹；100 条探针和修复前失败记录作为科研足迹保留。
- B 组历史 API 回答使用保留最新状态的默认模式；D 组使用时间线模式，两者语义不同。
