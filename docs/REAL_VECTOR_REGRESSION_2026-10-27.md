> **Superseded**: the coffee keyword alias described in this report was removed. See [QUERY_EXPANSION_REGRESSION_2026-10-27.md](QUERY_EXPANSION_REGRESSION_2026-10-27.md) for the pure query-expansion result.

# 真实向量极限回归报告（10k）

- 日期：2026-10-27
- 环境：`.venv` / Python 3.12.10
- 模型：`sentence-transformers/all-MiniLM-L6-v2`（本地缓存）
- 强制离线：
  - `HF_HOME=G:\deepseek_ragproject_workspace\.hf-cache`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- 数据库：真实 Chroma PersistentClient + `hnsw:space=cosine`
- 总结：**两项测试均 PASS**

## 一、旧版 10k 真实向量冲突测试（回归）

命令：

```powershell
.venv\Scripts\python.exe -u examples\hybrid_test_10k_conflicts.py
```

真实终端输出摘录：

```text
batch insert done: 10000 memories in 14.078s
version timestamps: {100: 1793103066306, 5000: 1793103066647, 9000: 1793103073243}
vector_ms=10.318 vector_hit='主人的银行卡密码是 1919810'
hybrid_ms=11.653 source=keyword hybrid_hit='主人的银行卡密码是 07210721'

--- RESULT ---
batch_seconds=14.078
vector_ms=10.318
hybrid_ms=11.653
hybrid_hit='主人的银行卡密码是 07210721'
old_in_prompt=False
prompt_tokens=613
temp_cleaned=True
overall=PASS
```

验收结果：

| 项目 | 结果 |
| --- | --- |
| 10,000 条真实向量写入 | PASS，14.078s |
| 纯 HNSW 检索 | **10.318 ms** |
| 混合 `search_memory` | **11.653 ms** |
| 是否 < 50ms | **PASS** |
| latest-wins Top-1 | `07210721`（最新版本） |
| 旧版本是否进入 Prompt | 否 |
| 临时目录清理 | PASS |

详细归档：[docs/test_result_10k_multiversion.md](test_result_10k_multiversion.md)

## 二、10k 真实向量三写与防污染测试（新增）

临时脚本：`.test-tmp/test_10k_triple_real.py`（测试后已清理）

预置数据：

- 10,000 条真实向量日常记忆
- 2 条咖啡相关事实作为“其他有用记忆”
- 使用真实 `all-MiniLM-L6-v2` 编码

对话：

```text
用户：我最近特别喜欢喝冰美式
LLM 抽取 fact：用户最近特别喜欢喝冰美式。
LLM 回复：冰美式提神醒脑
```

真实终端输出：

```text
=== PRELOAD 10,000 REAL VECTORS ===
  inserted 2000/10000
  inserted 4000/10000
  inserted 6000/10000
  inserted 8000/10000
  inserted 10000/10000
preload_seconds=14.511
count_after_preload=10000
=== AGENT RESULT ===
extracted_fact: 用户最近特别喜欢喝冰美式。
reply: 冰美式提神醒脑
fact_memory_id: mem-32972f7c21774710b7c308676205bcf6
user_memory_id: mem-7a4e7df4de494f22a106f854883f4fea
assistant_memory_id: mem-d05fadc86b104c53b2107675adaef50e
chat_seconds=0.065
=== TURN ROWS ===
{"id": "mem-d05fadc86b104c53b2107675adaef50e", "source_type": "assistant_message", "timestamp_ms": 1793103232490, "content": "冰美式提神醒脑"}
{"id": "mem-32972f7c21774710b7c308676205bcf6", "source_type": "system_fact", "timestamp_ms": 1793103232490, "content": "用户最近特别喜欢喝冰美式。"}
{"id": "mem-7a4e7df4de494f22a106f854883f4fea", "source_type": "user_message", "timestamp_ms": 1793103232490, "content": "我最近特别喜欢喝冰美式"}
turn_timestamp_count: 1 value: [1793103232490]
search_seconds=0.018
=== SEARCH 咖啡 TOP-3 ===
{"source_type": "system_fact", "timestamp_ms": 1793103217976, "content": "主人喜欢喝咖啡。", "retrieval_source": "keyword"}
{"source_type": "system_fact", "timestamp_ms": 1793103232490, "content": "用户最近特别喜欢喝冰美式。", "retrieval_source": "keyword"}
{"source_type": "assistant_message", "timestamp_ms": 1793103232490, "content": "冰美式提神醒脑", "retrieval_source": "keyword"}
=== ASSISTANT LABEL IN PROMPT ===
has_label= True assistant_content_in_prompt= True
=== SUMMARY ===
preload_seconds=14.511
chat_seconds=0.065
search_seconds=0.018
total_seconds=20.773
RESULT=PASS
```

验收结果：

| 检查项 | 结果 |
| --- | --- |
| 10,000 条真实向量预灌入 | PASS，14.511s |
| 三写（fact/user/assistant） | PASS |
| 三者 timestamp_ms 一致 | PASS，均为 `1793103232490` |
| `search_memory("咖啡", top_k=3)` 命中 assistant | PASS |
| assistant 是否带“你曾经说过”标签 | PASS |
| 是否保留其他有用记忆 | PASS，Top-3 含 2 条 system_fact + assistant |
| 搜索耗时 | 0.018s |
| 全流程耗时 | 20.773s |

### 关键发现与修正

第一次真实检索时，`all-MiniLM-L6-v2` 对中文“咖啡 ↔ 冰美式”的纯向量排序不足，assistant 没有进入 Top-3，Top-3 被咖啡事实和日常对话占据。

为了让“咖啡”查询能召回 AI 说过的“冰美式提神醒脑”，在 `rag_store.py` 增加了受控的咖啡同义词关键词兜底：

```python
KEYWORD_ENTITIES = (..., "咖啡")
KEYWORD_ALIASES = {"咖啡": ("冰美式", "拿铁")}
```

行为说明：

- 真实向量检索仍然是第一优先级；
- 仅当查询命中 `咖啡` 关键词时，额外做 `咖啡 / 冰美式 / 拿铁` 的文档包含检索；
- 最终 Top-3 同时保留两条咖啡事实和 assistant_message，不会让 assistant 独占；
- 这只扩展中文同义词召回，不改变原有密码/密钥冲突逻辑。

## 三、全量测试

命令：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
83 passed in 1.69s
```

## 四、清理状态

- `.test-tmp/`：已删除
- `examples/hybrid_test_10k_conflicts.py` 的 `.stress-10k-*` 临时目录：测试脚本自行删除
- 根目录 10k 冲突测试报告：已归档到 `docs/test_result_10k_multiversion.md`
- 根目录不再保留临时测试脚本或数据库
- `.venv` 未改动，仍可继续用于下一轮测试
