# Holographic Storage Regression Report

## 1. 目标

验证 Chroma → sqlite-vec 重构后，全息模式仍具备以下能力：

- 一轮对话的 `user_message` 与 `assistant_message` 无条件双写；
- 两条记录共享同一个 `timestamp_ms`；
- 系统注入的 `<memory-hint>` 不会进入数据库；
- 超过 500 字的长文本不会被截断。

## 2. 测试实现

新增：

```text
tests/test_holographic_storage.py
```

测试使用真实 `RAGStore` + `SqliteVecBackend`，直接读取：

```text
<临时目录>/store/agent_memory.sqlite3
```

为避免测试依赖外部模型下载，使用确定性的 4 维测试 embedding；存储层、SQLite 表、metadata 和时间戳逻辑均为真实实现。

写入入口为：

```python
MemoryPlugin.memory_add_turn(user_msg, ai_msg)
```

测试运行：

```bash
pytest -q -s tests/test_holographic_storage.py
```

## 3. 测试结果

### 测试一：无差别双写

输入：

```text
user: 今天天气真不错，我想吃个苹果
assistant: 苹果很健康，祝你今天有个好心情
```

实际从 sqlite-vec 数据库抓取到：

```json
[
  {
    "document": "今天天气真不错，我想吃个苹果",
    "metadata": {
      "source_type": "user_message",
      "timestamp_ms": 1793814215602
    }
  },
  {
    "document": "苹果很健康，祝你今天有个好心情",
    "metadata": {
      "source_type": "assistant_message",
      "timestamp_ms": 1793814215602
    }
  }
]
```

断言：

- 2 条记录；
- `user_message` / `assistant_message` 各一条；
- 原文逐字一致；
- `timestamp_ms` 完全一致。

**通过。**

### 测试二：防回声污染

写入内容：

```text
<memory-hint>
[System Note: optional background memories]
1. fake injected memory
</memory-hint>
```

结果：

```text
ok=false
数据库记录数：0
未出现包含 <memory-hint> 的记录
```

实现上增加了一层独立的写入保护：

```python
_INJECTED_DIALOGUE = re.compile(
    r"^\s*<(?:system-reminder|available_skills|skill_content|memory-hint)\b",
    re.IGNORECASE,
)
```

`MemoryPlugin.memory_add_turn()` 在写入前拒绝系统注入前缀。

**通过。**

### 测试三：长文本不截断

输入内容为：

```text
"哈哈哈" × 220 + "。"
```

实际长度：

```text
661 chars
```

数据库中保存的 `user_message` 长度：

```text
661 chars
```

逐字内容与输入完全一致。

**通过。**

## 4. 测试汇总

```text
3 passed
```

数据库实测记录已在上面的测试一输出中列出。

## 5. 变更文件

```text
src/agent_memory/plugin_entry.py
tests/test_holographic_storage.py
docs/HOLOGRAPHIC_STORAGE_TEST_REPORT.md
```

## 6. 结论

全息模式的核心双写、时间戳绑定、注入内容防污染和长文本保真在 sqlite-vec 后端上全部通过。
