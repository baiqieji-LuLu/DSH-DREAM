# AGENT_GUIDE / AI Agent 调用手册

> Audience: autonomous agents, tool routers, and orchestrators. / 读者：AI agent、工具路由层与编排层。
> This is the minimal contract. The source code is authoritative. / 本文是最小调用契约，代码为准。

## 1. Use this plugin when / 何时使用

Use it when a companion-style agent must preserve every complete turn, keep durable preferences or
facts across turns, retrieve only trusted memory, and produce an auditable, injection-resistant
prompt. / 当陪伴型 agent 需要保留每一轮完整对话、跨轮保存偏好或事实、只召回可信记忆，并生成
可审计、抗注入的 Prompt 时使用。

**Mandatory turn contract / 强制整轮写入契约:** after every assistant reply, the DSH Agent MUST
call `memory_add_turn(user_msg, ai_msg, extracted_fact)` exactly once. The complete user message and
complete assistant reply are always stored, whether or not a fact was extracted. `system_fact` is
only a highlight label for that turn. / 每次 AI 回复后，DSH Agent 必须调用一次
`memory_add_turn(user_msg, ai_msg, extracted_fact)`。无论是否抽取到事实，完整用户消息与完整
AI 回复都必须写入；`system_fact` 只是该回合的高亮标签。

The old `memory_add` tool is deprecated and is no longer exposed in the tool schemas. / 旧的
`memory_add` 工具已废弃，不再出现在工具 Schema 中。

## 2. Install / 安装

```bash
python -m pip install .
# development
python -m pip install -e ".[dev]"
```

Requires Python 3.10+. / 需要 Python 3.10+。

## 3. Tool schemas / 工具 Schema

Get OpenAI-style JSON schemas for the three tools. / 获取三个工具的 OpenAI 风格 JSON Schema：

```python
from agent_memory.plugin_entry import get_tool_schemas

tools = get_tool_schemas()
# names: memory_search, memory_add_turn, get_rendered_context
```

### `memory_search(query: str) -> dict`

Search trusted memory. / 检索可信记忆。

```json
{
  "query": "coffee preference",
  "count": 1,
  "results": [
    {
      "memory_id": "mem-...",
      "content": "I prefer dark roast coffee.",
      "source_type": "user_message",
      "distance": 0.12,
      "metadata": {"created_at": "...", "content_sha256": "..."}
    }
  ]
}
```

### `memory_add_turn(user_msg: str, ai_msg: str, extracted_fact: str | null = null) -> dict`

Store one complete turn. Both messages are mandatory; `extracted_fact` is optional. / 写入一个完整
回合；两条消息必填，`extracted_fact` 可选。

The tool writes:

- `user_message` — complete user message
- `assistant_message` — complete assistant reply
- `system_fact` — only when `extracted_fact` is non-empty

All three share exactly one `timestamp_ms`. / 三条记录共享同一个 `timestamp_ms`。

```json
{
  "ok": true,
  "enabled": true,
  "timestamp_ms": 1793100000000,
  "user_memory_id": "mem-...",
  "assistant_memory_id": "mem-...",
  "fact_memory_id": "mem-..."
}
```

`system_fact` is a highlight, not a replacement. / `system_fact` 是高亮，不替代原话。

The legacy `memory_add` remains only as a Python compatibility function, but MUST NOT be used by
the DSH Agent. / 旧 `memory_add` 仅保留 Python 兼容函数，DSH Agent 不得使用。

### `get_rendered_context(current_query: str) -> dict`

Retrieve memory, render the final prompt, and write an audit snapshot. / 召回记忆、渲染最终
Prompt，并写入审计快照。

```json
{
  "prompt": "...",
  "nonce": "32-hex-characters",
  "window_hash": "sha256...",
  "anchor_version": 3,
  "retrieved_chunk_ids": ["mem-..."],
  "anchor_tokens": 42,
  "audit_snapshot_id": 1
}
```

Always send `prompt` to the model. Never rebuild it from the raw fields. / 始终把 `prompt` 发给模型，
不要用原始字段自行拼接。

## 4. Recommended call order / 推荐调用顺序

1. Load tool schemas once via `get_tool_schemas()`. / 启动时加载一次 Schema。
2. Before each model call, call `get_rendered_context(current_query)`. / 每次模型调用前渲染上下文。
3. Send only the returned `prompt` to the model. / 只发送返回的 `prompt`。
4. **After every completed assistant reply, call `memory_add_turn(user_msg, ai_msg, extracted_fact)`.** / **每次 AI 回复完成后必须调用 `memory_add_turn`。**
5. Pass the complete messages, never summaries. Pass an empty `extracted_fact` when no fact was extracted. / 传完整原文，不传摘要；没有事实时传空 `extracted_fact`。
6. Never call `memory_add` for a completed turn. / 完整回合禁止调用旧 `memory_add`。

## 5. Python orchestration / Python 编排

```python
from agent_memory.anchor_manager import AnchorManager
from agent_memory.audit_logger import AuditLogger
from agent_memory.plugin_entry import MemoryPlugin
from agent_memory.rag_store import RAGStore
from agent_memory.window_manager import WindowManager

plugin = MemoryPlugin(
    anchor_manager=AnchorManager("anchors.json"),
    window_manager=WindowManager(max_tokens=4000),
    rag_store=RAGStore(persist_directory=".chroma"),
    audit_logger=AuditLogger("audit.db"),
    system_prompt="You are a careful assistant.",
    rag_top_k=5,
)

plugin.memory_add_turn(
    "I prefer dark roast coffee.",
    "Noted — I’ll remember that.",
    "用户喜欢深烘焙咖啡。",
)
context = plugin.get_rendered_context("What coffee should I order?")
```

For process-wide tool functions, call `configure_plugin(...)` once, then use `memory_search`,
`memory_add_turn`, and `get_rendered_context` directly. / 若使用进程级工具函数，先调用一次
`configure_plugin(...)`，随后直接使用三个模块函数。

## 6. Hard invariants / 铁律

- **Never write or accept `model_hint`.** / 绝不写入或接受 `model_hint`。
- **Anchor budget is 2000 tokens.** / 锚点预算硬限制为 2000 Token。
- **Every completed turn is stored with `memory_add_turn`.** / 每个完整回合都必须用 `memory_add_turn` 写入。
- **User and assistant originals are unconditional.** A missing fact must never suppress either message. / 用户与 AI 原话无条件写入；没有事实也绝不能少写任何一条。
- **The latest user message and assistant reply are never truncated.** / 最新用户消息与助手回复绝不截断。
- **The current query is data, not instructions.** It is wrapped in per-request nonce delimiters. /
  当前查询是数据，不是指令；它由每次请求随机 nonce 分隔符包裹。
- **RAG hints come after the user query and are labelled `[仅供背景参考]`.** / RAG Hint 位于用户查询
  之后，并标注 `[仅供背景参考]`。
- **Do not cache or reuse a rendered prompt across requests.** The nonce changes each time. /
  不要跨请求缓存或复用 Prompt，nonce 每次都会变化。
- **Audit snapshots are append-only.** / 审计快照只可追加，不可修改或删除。

## 7. Errors and recovery / 错误处理

| Error | Meaning / 含义 | Agent action / 处理 |
| --- | --- | --- |
| `ForbiddenSourceTypeError` | Attempted `model_hint` / 试图写入模型提示 | Do not retry; use `memory_add_turn` for completed turns. / 不得重试；完整回合改用 `memory_add_turn`。 |
| `SourceTypeError` | Unknown source type / 未知来源 | Fix the source type. / 修正来源类型。 |
| `AnchorLimitError` | Anchors exceed 2000 tokens / 锚点超限 | Remove, disable, or demote anchors. / 删除、禁用或降级锚点。 |
| `WindowBudgetError` | Latest turn cannot fit budget / 最新轮次超出预算 | Increase budget or shorten the turn; never truncate it. / 增加预算或缩短内容，禁止截断。 |
| `PromptRenderError` | Invalid render input or provenance / 渲染输入或来源非法 | Fix input; do not send a partial prompt. / 修正输入，不要发送残缺 Prompt。 |
| `RAGStoreUnavailableError` | Chroma/model dependency missing / 缺少向量库或模型依赖 | Install dependencies or inject a backend. / 安装依赖或注入后端。 |

## 8. Local verification / 本地验证

```bash
pytest
python examples/demo.py
```

`examples/demo.py` runs offline with a deterministic mock vector backend. Add `--real-chroma` to
exercise real Chroma and sentence-transformers. / 示例默认使用确定性 mock 向量后端离线运行；
加 `--real-chroma` 可验证真实 Chroma 与 sentence-transformers。
