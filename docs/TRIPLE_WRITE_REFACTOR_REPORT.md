# 三写与 Assistant 记忆隔离重构报告

- 生成时间：2026-10-27
- 范围：`src/agent_memory/rag_store.py`、`memory_agent.py`、`renderer.py`、`plugin_entry.py`
- 结论：**PASS**

## 1. 本次目标

让 AI 也记住自己说过的话，并确保同一轮对话的：

- 用户完整原文（`user_message`）
- AI 完整原话（`assistant_message`）
- 抽取事实（`system_fact`，如果存在）

共享同一个 `timestamp_ms`，作为一个完整回合绑定。

## 2. 代码改动

### 2.1 `rag_store.py`

- `ALLOWED_SOURCE_TYPES` 新增 `assistant_message`：
  - 允许：`user_message`、`system_fact`、`tool_result`、`assistant_message`
  - 仍然永久拒绝：`model_hint`
- `SourceType` 同步扩展。
- 新增公开方法 `RAGStore.next_timestamp_ms()`：
  - 供 `MemoryAgent` 在没有抽取到 fact 时先申请时间戳，再让 user/assistant 共用。
- `add_memory()` 已支持显式 `timestamp_ms`，用于同回合绑定。

### 2.2 `memory_agent.py`

- `MemoryAgentResult` 新增 `assistant_memory_id`。
- `_finish()` 改为三写：
  1. 生成或复用本回合 `turn_timestamp_ms`
  2. 写完整用户原文为 `user_message`
  3. 写 AI 原话为 `assistant_message`
  4. 两者使用同一个 `timestamp_ms`
- 时间戳规则：
  - 如果本轮抽取了 `system_fact`，复用 fact 的 `timestamp_ms`；
  - 如果没抽取 fact，调用 `store.next_timestamp_ms()` 生成一次，user/assistant 共用。
- 原文和 AI 回复都按完整内容写入，不做摘要。

### 2.3 `renderer.py`

- 当 `MemoryHit.source_type == "assistant_message"` 时，RAG Hint 强制加：

```text
[这是你曾经对用户说过的话]
```

- 防止模型把 AI 自己说过的“我喜欢喝茶”误认为用户偏好。

### 2.4 `plugin_entry.py`

- `SOURCE_TYPE_VALUES` 和工具 Schema 的 source_type enum 新增 `assistant_message`。

## 3. 真实 Chroma 测试

测试文件：`tests/test_triple_write.py`

输入：

```text
我今天吃了火锅
```

Mock LLM 抽取事实：

```text
用户今天吃了火锅。
```

Mock LLM 最终回复：

```text
好吃吗？
```

实际数据库记录：

```text
=== TRIPLE_WRITE_ROWS ===
{"id": "mem-e58b938fcad44d668cfac4d451838358", "source_type": "system_fact", "timestamp_ms": 1793102094025, "content": "用户今天吃了火锅。"}
{"id": "mem-022bc43d9e2e43a1a4d751f2e077201b", "source_type": "user_message", "timestamp_ms": 1793102094025, "content": "我今天吃了火锅"}
{"id": "mem-380f426d7fc845b9af20315db7a1f32e", "source_type": "assistant_message", "timestamp_ms": 1793102094025, "content": "好吃吗？"}
```

`search_memory("好吃吗？")` 命中：

```text
=== ASSISTANT_SEARCH_HIT ===
{"source_type": "assistant_message", "content": "好吃吗？", "retrieval_source": "vector"}
```

验证结果：

- 三条记录都存在于真实 Chroma；
- `system_fact`、`user_message`、`assistant_message` 共享同一个 `timestamp_ms`；
- 用户原文完整保留，不是摘要；
- AI 原话完整保存；
- 从 `search_memory` 能正确召回 `assistant_message`；
- renderer 会为 assistant 记忆加上“你曾经说过”标签。

## 4. 全量测试

执行：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
83 passed in 1.68s
```

基线要求 80+，当前通过 83 项。

## 5. 行为说明与边界

- 有 fact：fact、user、assistant 三条共享同一 timestamp。
- 无 fact：user、assistant 共享同一 timestamp；不会再出现两条不同时间戳。
- 写入顺序：fact 先写（便于本轮检索），user/assistant 在 `_finish` 写入。
- 当前三写不是单一数据库事务；如果某一写失败，其余记录仍会保留。若后续要求原子三写，可增加 `RAGStore.add_turn()` 批量接口。
- `chat_demo.py` 不需要单独改写入逻辑，它通过 `MemoryAgent` 自动获得三写行为。
- `assistant_message` 与 `user_message` 在 Prompt 中会明确区分来源，assistant 内容带强制引用标签。
