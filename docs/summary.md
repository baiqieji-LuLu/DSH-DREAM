# DSH-DREAM 四阶段实施总结

> 项目路径：`G:\deepseek_ragproject_workspace`
> 项目名称：`DSH-DREAM`
> 版本：`0.1.0`
> 结论：四个阶段均已实施完成；最终 pytest 收集 **67 项测试，全部通过**。

本文按阶段记录“做了什么、为什么这样做、如何验证”，便于逐项检查是否达成任务目标。

## 0. 总览

| 阶段 | 目标 | 主要交付 | 验证结果 |
| --- | --- | --- | --- |
| 阶段一 | 工程初始化与目录基建 | `src` 布局、构建配置、双语 README、MIT、公共 API | 5 项冒烟测试通过；wheel 构建成功 |
| 阶段二 | 四层核心架构 | 锚点、滑动窗口、白名单 RAG、SQLite 审计 | 36 项测试通过；ruff/mypy 通过 |
| 阶段三 | 渲染引擎与插件入口 | nonce 安全渲染器、JSON-Schema 工具函数 | 60 项测试通过；本地模拟链路跑通 |
| 阶段四 | 测试与交付 | pytest 验收测试、`demo.py`、`AGENT_GUIDE.md` | **67 passed**；demo、ruff、mypy、wheel/sdist 全部通过 |

### 最终验证快照

| 检查项 | 命令 | 结果 |
| --- | --- | --- |
| pytest | `python -m pytest -p no:cacheprovider` | **67 passed** |
| ruff | `ruff check src tests examples` | **All checks passed** |
| mypy strict | `mypy --config-file pyproject.toml` | **8 个源文件无问题** |
| 端到端 demo | `python examples/demo.py` | **退出码 0** |
| 1000 轮真实模型压力测试 | `python examples/stress_test.py --backend chroma` | **PASS**；2000 条历史 → 4 条窗口 |
| wheel 构建 | `python -m build` | wheel + sdist 成功 |
| wheel 导入 | 从构建 wheel 直接导入 | 四个核心类与工具函数可用 |

---

## 1. 阶段一：工程初始化与目录基建

### 任务要求

- 创建标准 Python 包结构：`src/agent_memory/`、`tests/`、`examples/`
- 生成 `pyproject.toml`，包含 `chromadb`、`sentence-transformers`、`jinja2` 等依赖，支持 `pip install .`
- 生成中英双语 `README.md`、`.gitignore`、MIT `LICENSE`
- 在 `src/agent_memory/__init__.py` 定义公共接口
- 展示目录树后等待阶段二

### 实际完成

- 创建 `src` 布局、`tests`、`examples` 和 `py.typed`。
- 使用 Hatchling + PEP 621 配置构建；要求 Python 3.10+。
- 运行依赖固定为：
  - `chromadb>=0.5.0`
  - `jinja2>=3.1.0`
  - `sentence-transformers>=2.7.0`
- 开发依赖包含 `pytest`、`pytest-cov`、`ruff`、`mypy`、`build`。
- 配置 pytest 的 `pythonpath = ["src"]`、ruff、mypy strict、coverage。
- README 使用中英双语说明边界、架构、安全铁律、公共 API 和安装方式。
- `__init__.py` 声明 8 个公共导出，并使用惰性导入避免包导入时加载 Chroma 或模型：
  - `AnchorManager`
  - `WindowManager`
  - `RAGStore`
  - `AuditLogger`
  - `render_prompt`
  - `memory_search`
  - `memory_add`
  - `get_rendered_context`
- 添加 `tests/test_public_api.py`，验证版本、导出集合、名称唯一性和未知属性行为。
- wheel 实际构建成功，包内包含 `__init__.py`、`py.typed`、LICENSE 元数据。

### 对应文件

- [pyproject.toml](pyproject.toml)
- [README.md](README.md)
- [LICENSE](LICENSE)
- [.gitignore](.gitignore)
- [src/agent_memory/__init__.py](src/agent_memory/__init__.py)
- [tests/test_public_api.py](tests/test_public_api.py)

### 验收结论

阶段一目标全部完成。该阶段测试为 5 项，最终仍包含在 67 项测试中。

---

## 2. 阶段二：核心四层架构

### 任务要求

实现四个模块，每个模块具有完善英文 Docstring 与 Type Hint：

- `anchor_manager.py`：加载 `anchors.json`，Token 硬上限 2000，提供 `promote()` / `demote()`
- `window_manager.py`：按 Token 预算裁剪历史，保护最新用户消息与回复
- `rag_store.py`：Chroma `PersistentClient`；`add_memory()` 必须带 `source_type`，只允许三种来源，拒绝 `model_hint`
- `audit_logger.py`：SQLite 记录 `timestamp`、`retrieved_chunk_ids`、`window_hash`、`anchor_version`

### 2.1 共享 Token 估算

新增 [src/agent_memory/_tokens.py](src/agent_memory/_tokens.py)：

- 确定性中英文 Token 估算，避免运行时依赖 tokenizer 下载。
- 支持注入自定义 `TokenCounter`，便于接入精确 tokenizer。
- 校验计数器返回值必须为非负整数。

### 2.2 `AnchorManager`

对应文件：[src/agent_memory/anchor_manager.py](src/agent_memory/anchor_manager.py)

实现内容：

- JSON 文件格式包含 `schema_version`、单调递增 `version` 和 `anchors` 数组。
- 每个锚点包含 `id`、`content`、`priority`、`enabled`。
- `MAX_ANCHOR_TOKENS = 2000`。
- `max_tokens` 只能在 `1..2000` 内配置，不能突破 2000。
- 加载、添加、启用锚点都会再次校验总量；超限抛出 `AnchorLimitError`，不修改内存或磁盘。
- 支持：
  - `add_anchor()`
  - `remove_anchor()`
  - `promote()`：提高优先级、启用锚点
  - `demote()`：降低优先级但不删除
  - `set_enabled()`
  - `render()`
  - `reload()`
  - `save()`
- 写入使用临时文件 + `os.replace` 原子替换。
- 版本号在每次成功变更后递增，供审计使用。
- 线程安全（`RLock`）。

### 2.3 `WindowManager`

对应文件：[src/agent_memory/window_manager.py](src/agent_memory/window_manager.py)

实现内容：

- 默认预算 `4000` tokens，支持自定义 tokenizer 与每条消息的 framing overhead。
- 历史消息使用不可变 `Message` 数据类，支持 `system`、`user`、`assistant`、`tool`。
- 裁剪算法：
  - 找到最后一条 `user` 消息；
  - 从该消息到末尾的整个“最新轮次”强制保留；
  - 从新到旧向前补齐，直到预算耗尽；
  - 不跳过消息，结果始终是连续后缀。
- 如果最新轮次本身超过预算，抛出 `WindowBudgetError`，绝不截断最新用户消息或回复。
- 提供稳定 SHA-256 `window_hash`，并通过 `hash_messages()` 复用。
- 支持追加、批量追加、清空、显式历史裁剪和当前窗口属性。

### 2.4 `RAGStore`

对应文件：[src/agent_memory/rag_store.py](src/agent_memory/rag_store.py)

实现内容：

- 使用 `chromadb.PersistentClient` 打开本地持久化集合，默认目录 `.chroma`。
- 默认嵌入函数使用 Chroma 的 `SentenceTransformerEmbeddingFunction`；支持注入自定义 client / embedding function，便于测试和宿主集成。
- `add_memory(content, source_type)` 强制要求 `source_type`。
- 写入白名单只包含：
  - `user_message`
  - `system_fact`
  - `tool_result`
- `model_hint` 在内容向量化前被 `ForbiddenSourceTypeError` 拒绝。
- 统一来源校验函数 `validate_source_type()`，被 RAGStore 与插件入口共用。
- 读取时再次校验来源；即使数据库被手工写入 `model_hint`，`search()` 与 `get_memory()` 也不会返回。
- 元数据保留 `source_type`、`created_at`、`content_sha256`，调用方不能覆盖这些保留字段。
- 支持按来源过滤、Top-K 查询、按 ID 读取和删除。
- Chroma 与 sentence-transformers 均惰性导入，不阻塞包导入。

### 2.5 `AuditLogger`

对应文件：[src/agent_memory/audit_logger.py](src/agent_memory/audit_logger.py)

实现内容：

- 使用标准库 `sqlite3`，默认数据库 `audit.db`。
- 表 `request_snapshots` 包含：
  - `timestamp`
  - `retrieved_chunk_ids`（JSON 数组）
  - `window_hash`
  - `anchor_version`
- 时间统一为 UTC；naive datetime 按 UTC 处理。
- 提供 `record_snapshot()`、`get_snapshot()`、`list_snapshots()`、`count_snapshots()`。
- 使用 SQLite 触发器禁止 `UPDATE` 与 `DELETE`，保证审计记录追加式不可变。
- 线程安全，支持文件数据库与 `:memory:`。

### 阶段二验证

- 新增测试：
  - [tests/test_anchor_manager.py](tests/test_anchor_manager.py)
  - [tests/test_window_manager.py](tests/test_window_manager.py)
  - [tests/test_rag_store.py](tests/test_rag_store.py)
  - [tests/test_audit_logger.py](tests/test_audit_logger.py)
- 阶段二结束时累计 **36 项测试通过**。
- ruff 与 mypy strict 均通过。

### 验收结论

四层核心模块均实现，安全边界明确，测试通过。

---

## 3. 阶段三：渲染引擎与插件入口

### 任务要求

- `renderer.py`：组装系统 Prompt、锚点、窗口、用户查询、RAG Hint
- 使用不可伪造的特殊分隔符包裹用户查询
- RAG Hint 必须位于用户查询之后，并标注 `[仅供背景参考]`
- `plugin_entry.py`：提供 `memory_search`、`memory_add`、`get_rendered_context`
- 使用本地模拟数据跑通

### 3.1 `renderer.py`

对应文件：[src/agent_memory/renderer.py](src/agent_memory/renderer.py)

实现内容：

- 每次渲染生成 128-bit 随机 nonce：`secrets.token_hex(16)`。
- 用户查询包裹为：

```text
<|user_query_start:{nonce}|>
...
<|user_query_end:{nonce}|>
```

- nonce 参与窗口、查询、RAG Hint 的边界，攻击者无法预先知道闭合边界。
- 不可信文本中的 `<|` 和 `|>` 会转义为 `<\|`、`\|>`；nonce 文本会被脱敏。
- 历史窗口也使用 nonce 边界包裹，明确标记为不可信上下文。
- RAG Hint 固定放在用户查询之后，模板中包含 `[仅供背景参考]`。
- 支持：
  - 系统 Prompt
  - `AnchorManager` / 字符串 / 字符串序列作为锚点
  - `WindowManager` / 消息序列作为窗口
  - 字符串或 `MemoryHit` 序列作为 RAG Hint
- 锚点渲染时再次校验 2000 Token 硬上限。
- 结构化 `MemoryHit` 会校验来源，非法 `model_hint` 不能进入 Prompt。
- 使用 Jinja2 固定内部模板；调用方不能重排安全区块。
- `render_prompt()` 返回字符串。
- `build_prompt()` 额外返回 nonce、窗口哈希、锚点版本、召回 ID、锚点 Token 数，供审计使用。

### 3.2 `plugin_entry.py`

对应文件：[src/agent_memory/plugin_entry.py](src/agent_memory/plugin_entry.py)

实现内容：

- 标准 JSON-Schema Tool Calling 结构，工具：
  - `memory_search(query: str)`
  - `memory_add(content: str, source_type: str)`
  - `get_rendered_context(current_query: str)`
- `get_tool_schemas()` 返回深拷贝的 OpenAI 风格 schemas。
- `memory_add` 的 schema 中 `source_type` 枚举只包含三个合法值，明确不包含 `model_hint`。
- `MemoryPlugin` 封装 Anchor/Window/RAG/Audit 组件，支持：
  - `memory_search()`
  - `memory_add()`
  - `get_rendered_context()`
  - `record_turn()`
- `configure_plugin()` 支持依赖注入，便于本地测试、替换后端和宿主集成。
- 模块级函数使用惰性创建的默认插件；默认插件包含 AuditLogger。
- `get_rendered_context()` 的返回包含：
  - `prompt`
  - `nonce`
  - `window_hash`
  - `anchor_version`
  - `retrieved_chunk_ids`
  - `anchor_tokens`
  - `audit_snapshot_id`
- 每次渲染都会写入审计快照（配置了 AuditLogger 时）。

### 阶段三验证

- 新增测试：
  - [tests/test_renderer.py](tests/test_renderer.py)
  - [tests/test_plugin_entry.py](tests/test_plugin_entry.py)
- 阶段三结束时累计 **60 项测试通过**。
- 本地模拟链路验证：
  - 写入偏好
  - 拦截 `model_hint`
  - 多轮窗口
  - 召回偏好
  - 生成 Prompt
  - 写入审计快照

### 验收结论

安全渲染器与插件入口完成；Prompt 注入防护、Hint 后置和来源校验均通过测试。

---

## 4. 阶段四：测试与交付

### 任务要求

- 在 `tests/` 下编写 pytest 测试，覆盖：
  - 锚点降级
  - 滑动窗口截断
  - RAG 召回准确率
  - 非法 hint 写入被拒绝的异常
- 在 `examples/` 下编写 `demo.py`，模拟真实交互
- 运行测试，确保全绿
- 生成中英双语 `AGENT_GUIDE.md`

### 4.1 pytest 验收测试

对应文件：[tests/test_delivery.py](tests/test_delivery.py)

覆盖内容：

| 测试 | 验证点 |
| --- | --- |
| `test_anchor_demotion_lowers_priority_and_rank` | 降级后优先级下降、排序改变、版本递增、锚点仍存在 |
| `test_window_truncation_keeps_protected_latest_turn` | 3 种预算下只裁剪旧消息，完整保留最新轮次 |
| `test_window_fails_closed_when_latest_turn_cannot_fit` | 最新轮次超预算时抛出 `WindowBudgetError`，不截断 |
| `test_rag_recall_accuracy_on_local_corpus` | 5 个查询，使用确定性 Chroma 兼容集合，recall@2 = 1.0，目标命中第 1 位 |
| `test_model_hint_write_is_rejected_before_embedding` | 非法 hint 与大小写变体均被拒绝，底层 add 调用次数为 0 |

### 4.2 端到端 demo

对应文件：[examples/demo.py](examples/demo.py)

执行流程：

1. 写入 `user_message`、`system_fact`、`tool_result` 三条偏好/事实。
2. 尝试写入 `model_hint`，展示拒绝异常。
3. 写入三轮对话，窗口管理后保留 4 条消息（共 6 条历史）。
4. 搜索 “coffee preference”，召回咖啡偏好。
5. 渲染最终 Prompt。
6. 打印追加式审计快照。

实际运行摘要：

```text
[3] Multi-turn conversation
  stored history: 6 messages
  active window:  4 messages
[4] Retrieve preference
  user_message: I prefer dark roast coffee, no sugar.
[5] Render final prompt
  nonce: 30a63cf7b058907d39f4922d55bdb991
  audit snapshot: 1
  hint after query: True
```

默认使用确定性本地 mock 向量后端，不下载模型：

```bash
python examples/demo.py
```

需要真实 Chroma 与 sentence-transformers 时：

```bash
python examples/demo.py --real-chroma
```

### 4.3 AI Agent 调用手册

对应文件：[AGENT_GUIDE.md](AGENT_GUIDE.md)

中英双语，包含：

- 何时使用 / 不使用
- 安装方式
- 三个工具的输入输出契约
- 推荐调用顺序
- Python 编排示例
- 安全铁律
- 错误类型与恢复动作
- 本地验证命令

### 阶段四最终验证

| 检查项 | 结果 |
| --- | --- |
| pytest | **67 passed** |
| ruff（src/tests/examples） | **All checks passed** |
| mypy strict | **8 个源文件无问题** |
| demo | **退出码 0** |
| wheel + sdist | 构建成功 |
| sdist 内容 | 包含 `AGENT_GUIDE.md`、`examples/demo.py`、`tests/test_delivery.py` |

### 验收结论

阶段四交付物齐全，pytest 全绿，demo 可离线运行，AI 调用手册完成。

---

## 5. 任务目标逐项对照

| # | 原始要求 | 实现 | 状态 |
| --- | --- | --- | --- |
| 1 | 标准 Python 包结构 | `src` 布局、`tests`、`examples`、`py.typed` | 完成 |
| 2 | `pyproject.toml` 与核心依赖 | Hatchling、Chroma、sentence-transformers、Jinja2、dev 依赖 | 完成 |
| 3 | 支持 `pip install .` | wheel 与 sdist 均构建成功，wheel 可导入 | 完成 |
| 4 | 中英双语 README | 架构、安全、API、安装、开发命令 | 完成 |
| 5 | `.gitignore` + MIT LICENSE | 完整 Python/项目忽略规则与 MIT 文本 | 完成 |
| 6 | `__init__.py` 公共接口 | 8 个惰性导出 + 类型检查导入 | 完成 |
| 7 | 锚点加载与 2000 Token 硬限制 | `AnchorManager`、`AnchorLimitError`、原子写入 | 完成 |
| 8 | `promote()` / `demote()` | 优先级升降、排序变化、版本递增 | 完成 |
| 9 | 滑动窗口裁剪 | 默认 4000；最新 user/assistant 轮次完整保留 | 完成 |
| 10 | 最新消息不被截断 | 超预算时 fail closed，抛 `WindowBudgetError` | 完成 |
| 11 | Chroma 本地向量库 | `chromadb.PersistentClient` 路径 | 完成 |
| 12 | 来源白名单 | 写入 + 读取双重校验，`model_hint` 永久拒绝 | 完成 |
| 13 | SQLite 审计字段 | `timestamp`、`retrieved_chunk_ids`、`window_hash`、`anchor_version` | 完成 |
| 14 | 安全渲染器 | nonce 隔离、转义、固定区块顺序、Jinja2 | 完成 |
| 15 | RAG Hint 后置与标注 | 固定位于查询之后，含 `[仅供背景参考]` | 完成 |
| 16 | 三个插件入口 | `memory_search`、`memory_add`、`get_rendered_context` | 完成 |
| 17 | JSON Schema Tool Calling | `TOOL_SCHEMAS` + `get_tool_schemas()` | 完成 |
| 18 | 锚点降级测试 | pytest 验收测试 | 完成 |
| 19 | 窗口截断测试 | 参数化预算测试 | 完成 |
| 20 | RAG 召回准确率测试 | recall@2 = 1.0 | 完成 |
| 21 | 非法 hint 异常测试 | `ForbiddenSourceTypeError`，零写入 | 完成 |
| 22 | 本地模拟 demo | `examples/demo.py` 完整链路 | 完成 |
| 23 | 全部测试通过 | 67 passed | 完成 |
| 24 | 中英双语 AI 指南 | `AGENT_GUIDE.md` | 完成 |
| 25 | 追加：1000 轮真实模型压力测试 | `examples/stress_test.py` + `test_result3.md` | 完成 |
| 26 | 追加：100,000 条记忆地狱级测试 | `examples/stress_test_100k.py` + `test_result_100k.md` | 完成 |
| 27 | 追加：10,000 条真实向量混合检索 | `examples/hybrid_test_10k.py` + `test_result_10k_hybrid.md` | 完成 |
| 28 | 追加：多版本密码冲突与时间戳重排 | `examples/hybrid_test_10k_conflicts.py` + `test_result_10k_multiversion.md` | 完成 |
| 29 | 追加：100,000 条真实向量多版本冲突 | `examples/hybrid_test_100k_real.py` + `test_result_100k_real.md` | 完成 |
| 30 | 追加：持久化交互聊天 demo | `examples/chat_demo.py` + `chat_memory_db/` | 完成 |
| 31 | 追加：LLM 认知记忆 Agent | `src/agent_memory/memory_agent.py` + `tests/test_memory_agent.py` | 完成 |
| 32 | 追加：DSH 原生设置面板与配置联动 | `manifest.json` + `dsh_plugin/` + `config.py` | 完成 |

---

## 6. 如何复核

从项目根目录执行：

```bash
cd G:\deepseek_ragproject_workspace

# 安装开发环境
python -m pip install -e ".[dev]"

# 全量测试
python -m pytest -p no:cacheprovider

# 端到端演示
python examples/demo.py

# 1000 轮真实模型压力测试（本地模型缓存）
$env:HF_ENDPOINT="https://hf-mirror.com"
.venv\Scripts\python.exe -u examples/stress_test.py --backend chroma

# 代码风格与类型检查
ruff check src tests examples
mypy --config-file pyproject.toml

# 构建 wheel + sdist
python -m build
```

预期结果：

```text
67 passed
```

demo 预期输出包含：

```text
blocked: ForbiddenSourceTypeError
retrieved ids: [...]
hint after query: True
audit snapshot: 1
```

---

## 7. 最终交付物清单

```text
DSH-DREAM/
├── src/agent_memory/
│   ├── __init__.py
│   ├── _tokens.py
│   ├── anchor_manager.py
│   ├── window_manager.py
│   ├── rag_store.py
│   ├── config.py
│   ├── memory_agent.py
│   ├── audit_logger.py
│   ├── renderer.py
│   ├── plugin_entry.py
│   └── py.typed
├── tests/
│   ├── test_public_api.py
│   ├── test_anchor_manager.py
│   ├── test_window_manager.py
│   ├── test_rag_store.py
│   ├── test_audit_logger.py
│   ├── test_renderer.py
│   ├── test_plugin_entry.py
│   └── test_delivery.py
├── examples/
│   ├── demo.py
│   ├── stress_test.py
│   ├── stress_test_100k.py
│   ├── hybrid_test_10k.py
│   ├── hybrid_test_10k_conflicts.py
│   ├── hybrid_test_100k_real.py
│   ├── chat_demo.py
│   └── README.md
├── test_result.md
├── test_result2.md
├── test_result3.md
├── test_result_100k.md
├── test_result_100k_real.md
├── test_result_10k_hybrid.md
├── test_result_10k_multiversion.md
├── AGENT_GUIDE.md
├── summary.md
├── README.md
├── pyproject.toml
├── LICENSE
└── .gitignore
```

根目录的 `【任务目标草案_请勿完全参考】.txt` 是外部任务说明文件，不属于项目代码交付物。

---

## 8. 已知边界与说明

1. **真实 Chroma 端到端未在当前环境执行**
   当前 DSH 运行时没有安装 `chromadb` 与 `sentence-transformers`。代码中的生产路径使用
   `chromadb.PersistentClient`；测试和默认 demo 使用具有相同接口与返回结构的确定性本地
   Chroma 兼容后端，以避免下载大模型。安装完整依赖后可运行：

   ```bash
   python examples/demo.py --real-chroma
   ```

2. **RAG 召回准确率测试是确定性词法基准**
   `test_rag_recall_accuracy_on_local_corpus` 使用词法相似度的 Chroma 兼容集合，验证召回
   pipeline、排序、来源过滤和返回结构，不评估真实 embedding 模型的语义质量。

3. **Jinja2 与 pytest 为验证环境安装**
   项目已在 `pyproject.toml` 中声明这些依赖；当前 DSH 运行时安装了它们以完成本地验证。
   `chromadb` / `sentence-transformers` 未安装。

4. **未初始化 Git 仓库**
   当前环境没有 `git` 命令，因此未执行 `git init`；`.gitignore` 和 MIT LICENSE 已就绪。

5. **pytest 缓存警告**
   在 DSH Windows 沙箱下普通 `pytest` 可能出现 `.pytest_cache` 权限警告，不影响测试结果；
   使用 `python -m pytest -p no:cacheprovider` 可获得干净输出。

---

## 9. 补充验证：1000 轮真实模型压力测试

脚本：[examples/stress_test.py](examples/stress_test.py)
报告：[test_result3.md](test_result3.md)

该补充测试使用工作区隔离临时目录 + `shutil.rmtree()` 清理全部数据库：

- 1000 轮日常闲聊，每轮 2 条消息，共 2000 条历史。
- 第 50 轮隐藏一条 `system_fact`：`我的银行卡密码是 888888，绝对不能忘记`。
- 80 Token 窗口最终保留 4 条消息；第 50 轮标记与敏感事实均不在窗口内。
- 第 1001 轮通过 `memory_search` 检索，命中来源 `system_fact`。
- 真实 Chroma `PersistentClient` + **真实 sentence-transformers all-MiniLM-L6-v2**。
- 检索耗时 **9.906 ms**；最终 Prompt **401 Token**。
- 总耗时 **14.273 秒**（本地模型缓存加载 + 1000 条真实向量计算）。
- 批量写入 1000 条耗时 1.277 秒。
- 临时目录删除成功，根目录无 `audit.db`、`anchors.json`、`.chroma` 或 `chroma` 残留。

结果：**PASS**。模型缓存位于 Git-ignored 的 `.hf-cache/`（约 87 MB）。

---

## 10. 追加验证：100,000 条记忆地狱级测试

脚本：[examples/stress_test_100k.py](examples/stress_test_100k.py)
报告：[test_result_100k.md](test_result_100k.md)

- 真实 Chroma `PersistentClient` + 1024 维确定性哈希 embedding。
- 100,000 条记忆批量写入，按 Chroma 5,000 条上限分块；批量写入 **44.404 秒**。
- 密码隐藏在第 54,321 条；第 100,001 轮查询。
- 精确余弦扫描命中密码：**8,220.499 ms**，Top-1，来源 `system_fact`。
- 人格锚点位于 Prompt 可信头部；RAG Hint 在用户查询之后，密码只出现在 Hint。
- 最终 Prompt **683 tokens**，严格 <1000。
- 总耗时 **53.015 秒**；临时目录与数据库清理成功。

**发现：** 100k 条 1024 维哈希向量下，Chroma HNSW 近似检索未访问到该密码记录，
因此报告使用对 Chroma 真实存储 embedding 的精确扫描来验证“精准召回”。这是当前
ANN 参数/实现的限制，不应视为生产级毫秒检索性能。

---

## 11. 追加验证：10,000 条真实向量混合检索

脚本：[examples/hybrid_test_10k.py](examples/hybrid_test_10k.py)
报告：[test_result_10k_hybrid.md](test_result_10k_hybrid.md)

- 真实 Chroma `PersistentClient` + 真实 `all-MiniLM-L6-v2`，`hnsw:space=cosine`。
- 10,000 条真实向量批量写入 **12.724 秒**。
- 纯 HNSW：**19.695 ms**，<50 ms，Top-1 直接命中密码。
- `search_memory()` 混合检索：**15.821 ms**，查询含“密码”触发关键词兜底，命中来源
  `keyword`，无全量向量扫描。
- 最终 Prompt **687 tokens**；人格锚点、RAG 后置和密码只出现在 Hint 均 PASS。
- 总耗时 **27.245 秒**，临时目录和数据库清理成功。

结果：**PASS**。这证明真实语义向量的 HNSW 路径可在 50 ms 内精准命中；关键词兜底
用于实体查询和低置信度场景。

---

## 12. 追加验证：多版本密码冲突与时间戳重排

脚本：[examples/hybrid_test_10k_conflicts.py](examples/hybrid_test_10k_conflicts.py)
报告：[test_result_10k_multiversion.md](test_result_10k_multiversion.md)

- 三版密码分别写入第 100、5,000、9,000 条，每条写入 metadata 严格递增的 `timestamp_ms`。
- 10,000 条真实向量批量写入 **14.885 秒**。
- 纯 HNSW **8.421 ms**，但返回旧版 `1919810`，说明纯 ANN 不解决冲突。
- 混合 `search_memory()` **10.347 ms**，关键词召回冲突组后按时间戳降序并 latest-wins，
  Top-1 为最新版 `07210721`。
- 旧版 `114514`/`1919810` 不出现在返回结果，也不出现在 Prompt。
- 最终 Prompt **613 tokens**；人格锚点、RAG 后置、清理全部 PASS；总耗时 **21.191 秒**。

结果：**PASS**。

---

## 13. 追加验证：100,000 条真实向量多版本冲突

脚本：[examples/hybrid_test_100k_real.py](examples/hybrid_test_100k_real.py)
报告：[test_result_100k_real.md](test_result_100k_real.md)

- 真实 `all-MiniLM-L6-v2`，每 2,000 条一批，共 50 批，批量写入 **165.586 秒**。
- 三版密码写入第 100 / 50,000 / 99,000 条，时间戳严格递增。
- 纯 HNSW **21.200 ms**（<100 ms），但 Top-1 命中闲聊，未命中密码。
- 时间戳混合检索 **51.536 ms**，命中最新版 `129745`；旧版不进入结果或 Prompt。
- 最终 Prompt **645 tokens**；人格锚点与 RAG 后置全部 PASS；总耗时 **172.735 秒**。
- 清理成功，无测试数据库残留。

**结论：** HNSW 时延达标但在 100k 真实向量上单独召回不足；关键词候选 + 时间戳
latest-wins 是保证冲突精度的必要层。

---

## 14. 追加验证：持久化交互聊天 Demo

脚本：[examples/chat_demo.py](examples/chat_demo.py)
持久化数据库：`chat_memory_db/`（Git-ignored）

- 首次运行灌入 50,000 条日常/偏好记忆，使用 1024 维确定性哈希 embedding。
- Chroma 持久化在 `./chat_memory_db`，第二次启动直接加载 50,003 条（含测试对话），不重复灌入。
- 对话链路：输入 → `search_memory()` → 旧记忆 Hint → `build_prompt()` → Mock LLM → 打印回复与召回项。
- 每条用户输入实时 `add_memory(source_type="user_message")` 写回 Chroma。
- 实测查询：喜欢喝什么、生日、名字，均召回对应 `system_fact`。
- 支持 `/prompt` 查看上次渲染 Prompt，`/exit` 退出；`--demo` 可非交互验证。
- 额外防止历史提问自匹配：过滤与当前输入完全相同的旧问题。

启动：

```bash
.venv\Scripts\python.exe -u examples/chat_demo.py
```

---

## 15. 追加验证：LLM 认知记忆 Agent

模块：[src/agent_memory/memory_agent.py](src/agent_memory/memory_agent.py)
测试：[tests/test_memory_agent.py](tests/test_memory_agent.py)

- `MemoryAgent(store, llm_client=...)` 支持注入任意 OpenAI 兼容客户端。
- `chat_with_memory()` 四步链路：偏好抽取（写 `system_fact`）→ 查询改写 →
  Top-10 召回与 LLM Rerank 至 Top-3 → renderer 受保护 Prompt + 最终生成。
- 无 `llm_client` 时降级到 `RAGStore.search_memory()`，打印终端提示，并使用核心默认
  fallback responder。
- `MemoryAgentResult` 暴露 reply、prompt、candidates、selected、抽取事实、改写词和
  `llm_calls`，便于审计。
- `build_deepseek_client()`、`filter_historical_questions()` 和默认兜底回复均已移入
  `memory_agent.py`；chat_demo 不再包含 LLM 客户端工厂、Prompt、Rerank 或过滤策略。
- fake OpenAI client 单元测试验证完整四步链路（4 次调用）、DeepSeek 默认配置和降级
  路径；pytest **72 passed**。
- chat_demo 只保留示例种子数据、交互循环和终端展示；`--llm` 优先读取
  `DEEPSEEK_API_KEY`（回退 `OPENAI_API_KEY`），默认
  `base_url=https://api.deepseek.com/v1`、模型 `deepseek-chat`。

---

## 16. 追加验证：DSH 原生设置面板与配置联动

- DSH 原生插件清单采用 `package.json` + `cordis.patch.yml`，设置表单由 Host 插件
  的 `Config` schema 自动生成；根目录 `manifest.json` 是可移植 schema/默认值。
- `dsh_plugin/` 提供四个 volatile 字段：`enable_memory` 开关、`rag_path` 只读绝对
  路径、`max_window_tokens` 数字（默认 1500）、`retrieval_top_k` 数字（默认 5）。
- Host 插件把解析后的配置写入 `AGENT_MEMORY_CONFIG` / `agent_memory_config.json`；
  Python `config.py` 读取该桥接文件。
- `plugin_entry.get_plugin()` 使用 `max_window_tokens` 初始化 `WindowManager`，使用
  `retrieval_top_k` 控制召回，使用 `rag_path` 打开 Chroma。
- `memory_search()` 在 `enable_memory` 关闭时直接返回空结果，不触碰 Chroma。
- `get_plugin_status()` 返回 `enabled`、绝对 `rag_path`、`memory_count`。
- 新增 80 项 pytest 全通过。
