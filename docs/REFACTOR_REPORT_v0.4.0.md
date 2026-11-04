# DSH-DREAM v0.4.0 架构重构报告

> 日期：2026-11-04  
> 主题：环境自举、模块化解耦与零配置 Python bootstrapping  
> 版本：v0.4.0

---

## 1. 重构目标

根据审计报告，v0.4.0 重点解决两件事：

1. `dsh_plugin/index.js` 过度膨胀，单文件承担了太多职责；
2. 新机器首次运行时，不能假设用户已经预装 Python 依赖或模型缓存。

本次重构不改变核心检索架构：

```text
ChromaDB + BGE
```

只重构宿主端的环境装配与模块边界。

---

## 2. 模块拆分

### 2.1 新结构

```text
dsh_plugin/
  index.js              # 组装入口、Config、holographic、settings/http
  runtime_env.js        # Python/缓存探测、uv 环境自举、状态机
  bridge_process.js     # Python 进程、warmup、query、stdio
  memory_tools.js       # memory_search 工具注册
  pre_retrieval.js      # agent/pre-step + <memory-hint>
  holographic_events.js # 原有全息事件纯函数
  turn_bridge.py        # 精简后仅信任宿主注入环境
```

### 2.2 index.js 行数

| 版本 | 行数 |
|---|---:|
| v0.2.0 备份 | 636 |
| v0.3.0 当前 | 1,323 |
| v0.4.0 重构后 | **约 660** |

入口文件不再包含：

- Python 解释器候选；
- HF 缓存候选；
- uv/pip 安装流程；
- memory_search 工具定义；
- pre-step 注入实现。

这些职责全部迁移到独立模块。

---

## 3. runtime_env.js：环境自举

### 3.1 Python 解释器

顺序：

```text
DSH_DREAM_PYTHON
→ UI python_path
→ profile/.dream-venv
→ .venv
→ 系统 python / python3
```

### 3.2 缓存目录

顺序：

```text
DSH_DREAM_HF_CACHE
→ UI hf_home_path
→ profile/.hf-cache
→ plugin/.hf-cache
→ cwd/.hf-cache
```

每个候选目录都会执行真实写入探针。

### 3.3 uv 自举

如果目标环境无法 import：

```text
chromadb
sentence-transformers
```

则执行：

```text
uv venv <profile>/.dream-venv
uv pip install \
  --python <venv-python> \
  chromadb sentence-transformers \
  torch==2.2.1 "numpy<2" \
  --quiet \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.4 全局状态机

`RuntimeEnv` 持有：

```js
this.ready = null;
this.state = {
  status: "idle" | "checking" | "provisioning" | "installed" | "ready" | "error",
  python,
  venv,
  error,
};
```

`ensureReady()` 返回同一个 Promise：

- 安装未完成时，`bridge_process.js` 不会 spawn Python；
- pre-retrieval 在 `provisioning/checking` 状态直接返回，不等待安装；
- 只有 `ready` 或 `installed` 才允许启动桥接。

这从架构上避免了 warmup 与 pre-retrieval 并发初始化 ChromaDB。

---

## 4. bridge_process.js：进程与协议

`PythonBridge` 负责：

- spawn Python；
- 发送 warmup；
- 管理 stdout 行协议；
- 维护 `request_id → Promise` 查询表；
- 提供 `query()` / `writeTurn()` / `close()`。

环境变量由宿主注入：

```text
DSH_DREAM_RESOLVED_PYTHON
DSH_DREAM_HF_CACHE
HF_HOME
DSH_DREAM_CONFIG
DSH_DREAM_HOST_PROVISION=1
```

启动前必须通过 runtime 状态门禁。

---

## 5. memory_tools.js

`memory_search` 工具的 Schema、参数和执行逻辑移入独立模块：

```text
query
top_k
timeline_mode
```

工具只依赖 `queryMemory` 回调，不再知道 Python 进程细节。

---

## 6. pre_retrieval.js

`agent/pre-step` 自动检索逻辑移入独立模块：

- 仅 `step === 1`；
- Top-3；
- `<memory-hint>` 包裹；
- 800ms（首次冷启动放宽到 15000ms）；
- 成功日志：

```text
[Holo] Pre-retrieval success: injected N memories
```

- 失败/无命中日志。

---

## 7. turn_bridge.py 精简

v0.3.0 的 `turn_bridge.py` 同时承担：

- Python 候选探测；
- venv 自愈；
- pip 安装；
- 缓存候选；
- 写缓存；
- 查询；
- 错误上报。

v0.4.0 后改为：

- 信任 `DSH_DREAM_RESOLVED_PYTHON` 与 `HF_HOME`；
- 不再自行寻找 Python；
- 不再自行创建 venv；
- 不再自行 pip install；
- 只保留：
  - warmup；
  - query；
  - holographic write；
  - `memory-hint` 过滤；
  - 清晰的依赖错误回传。

这样 Host 与 Python 不再各维护一套环境发现规则。

---

## 8. 版本升级

```text
package.json                 0.4.0
dsh_plugin/package.json      0.4.0
manifest.json                0.4.0
pyproject.toml               0.4.0
src/agent_memory/__init__.py 0.4.0
README*.md                   v0.4.0 Pre-Alpha
```

`dsh_plugin/package.json` 的 `files` 列表已加入：

```text
runtime_env.js
bridge_process.js
memory_tools.js
pre_retrieval.js
```

---

## 9. 验证

### Python

```text
96 passed, 1 skipped
```

跳过项为可选 `openai` 未安装，与本次重构无关。

### JavaScript

逐个执行：

```text
node --check dsh_plugin/index.js
node --check dsh_plugin/runtime_env.js
node --check dsh_plugin/bridge_process.js
node --check dsh_plugin/memory_tools.js
node --check dsh_plugin/pre_retrieval.js
node --check dsh_plugin/client.js
```

全部通过。

### 说明

本次没有实际触发 `uv venv` / `uv pip install`，避免在开发机上重新下载 torch 等大包；只验证了代码路径、语法和现有测试。

---

## 10. 已知风险

1. `uv` 必须存在于 PATH，或通过 `DSH_DREAM_UV` 指定；缺失时 UI 显示环境错误。
2. `torch==2.2.1` + `numpy<2` 是兼容性锁定，不覆盖所有平台；若某个平台没有对应 wheel，需要调整依赖策略。
3. 首次安装需要网络；离线环境仍需要预置 `.dream-venv`。
4. 模块拆分降低了 `index.js` 复杂度，但 `rag_store.py` 仍然是约 1930 行的大文件，后续仍需独立治理。
5. 真实的 DSH 市场安装环境仍需回归：
   - uv 安装；
   - warmup；
   - 工具调用；
   - 自动预检索；
   - Chroma 单例。

---

## 11. 验收清单

- [x] `index.js` 拆成组装入口
- [x] `runtime_env.js` 独立环境探测与 uv 自举
- [x] `bridge_process.js` 独立进程协议
- [x] `memory_tools.js` 独立工具注册
- [x] `pre_retrieval.js` 独立注入逻辑
- [x] 全局 Promise/状态机阻止并发初始化
- [x] UI 显示“正在初始化记忆引擎（首次运行可能需要几分钟）…”
- [x] Python 端移除重复环境探测
- [x] 版本升级到 0.4.0
- [x] Python 测试 96 passed, 1 skipped
- [x] JS 语法检查通过
- [x] 未执行 push
