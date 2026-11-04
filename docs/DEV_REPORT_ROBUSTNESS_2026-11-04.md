# DSH-DREAM 宿主桥接健壮性修复报告

> 日期：2026-11-04  
> 版本：v0.3.0 稳定性补丁  
> 主题：Python 解释器定位、配置路径优先级与旧配置隔离

---

## 1. 问题背景

纯净 DSH 测试环境暴露了两个相互独立、但会叠加放大问题：

### 1.1 Python 解释器错误

宿主端通过 `spawn` 拉起 Python 桥接进程时，解析到了系统的全局 `python`。

全局 Python 缺少：

```text
chromadb
sentence-transformers
```

桥接进程启动后直接报：

```text
ModuleNotFoundError
```

并退出。结果是：

- 全息入库桥接不可用；
- `memory_search` 工具不可用；
- 自动预检索不可用。

### 1.2 旧配置劫持 RAG 路径

机器上存在多份：

```text
agent_memory_config.json
```

旧的 `AGENT_MEMORY_CONFIG` 环境变量指向旧 checkout 或旧 profile，导致：

- Host 写入的不是当前 profile 的配置；
- Python 读取到旧 `rag_path`；
- 检索到了错误的 Chroma 数据库。

---

## 2. 修复目标

### 2.1 解释器解析优先级

```text
1. DSH_DREAM_PYTHON（显式覆盖）
2. 插件根目录 / 当前工作目录下的 .venv
3. 系统 PATH 中的 python / python3
```

### 2.2 RAG 路径优先级

```text
1. DSH 宿主 UI 写入的当前 agent_memory_config.json
2. AGENT_MEMORY_RAG_PATH
3. 当前用户 DSH 数据目录（~/.dsh/...）
```

### 2.3 配置文件优先级

```text
1. 显式传入的 config_path
2. DSH_DREAM_CONFIG（Host 当前 profile）
3. AGENT_MEMORY_CONFIG（兼容旧调用，仅作 fallback）
4. 插件根目录 agent_memory_config.json
5. manifest 默认值
```

---

## 3. 修改内容

### 3.1 `dsh_plugin/index.js`

#### 新增 `resolvePython()`

解析顺序：

```text
DSH_DREAM_PYTHON
  ↓ 未设置
DSH_DREAM_ROOT / 插件根目录 / process.cwd() 下的 .venv
  ↓ 不存在
python（Windows）或 python3（Unix）
```

Windows 候选：

```text
<root>\.venv\Scripts\python.exe
```

Unix 候选：

```text
<root>/.venv/bin/python
```

好处：

- 插件随仓库的 `.venv` 安装时，优先使用带依赖的解释器；
- 不再默认落到缺少 `chromadb` 的全局 Python；
- 仍保留 `DSH_DREAM_PYTHON` 的显式覆盖能力。

#### 配置文件隔离

```js
const bridgePath = () =>
  resolve(process.env.DSH_DREAM_CONFIG || CONFIG_BRIDGE_PATH);
```

不再使用全局 `AGENT_MEMORY_CONFIG` 决定当前 Host 的桥接目标。

启动 Python 时同时注入：

```js
env: {
  ...process.env,
  AGENT_MEMORY_CONFIG: target,
  DSH_DREAM_CONFIG: target,
}
```

这样当前 Profile 的路径始终优先于历史环境变量。

---

### 3.2 `dsh_plugin/turn_bridge.py`

#### 启动时自愈

新增：

```python
_maybe_reexec_with_venv()
```

逻辑：

1. 检查当前解释器是否能找到：
   - `agent_memory`
   - `chromadb`
   - `sentence_transformers`
2. 如果缺少依赖，按以下顺序寻找：
   - `DSH_DREAM_PYTHON`
   - 插件根目录 `.venv`
   - 当前工作目录 `.venv`
   - 当前工作目录的上级目录 `.venv`
3. 找到后通过 `os.execv()` 重新执行当前脚本；
4. 设置 `DSH_DREAM_REEXEC=1` 防止循环；
5. 找不到可用 venv 时，让原始导入暴露真实错误。

这层兜底保证即使 Host 端误用了全局 Python，桥接进程也能尽可能自动切换到项目 venv。

#### 查询与写入共存

`turn_bridge.py` 继续支持：

```text
全息写入：{"user_msg": "...", "ai_msg": "..."}
检索查询：{"op": "query", "request_id": "...", ...}
```

新增的查询分支复用同一个常驻 Python 进程，避免每次检索重新加载模型。

---

### 3.3 `src/agent_memory/config.py`

#### 配置路径优先级

新增：

```python
DSH_CONFIG_ENV = "DSH_DREAM_CONFIG"
```

`resolve_config_path()` 现在按：

```text
显式 config_path
  → DSH_DREAM_CONFIG
  → AGENT_MEMORY_CONFIG
  → 默认 profile 配置
```

解析。

旧的 `AGENT_MEMORY_CONFIG` 仍可用于直接调用 Python 库，但不会覆盖 Host 当前 profile。

#### RAG 路径优先级

保持并强化：

```python
ui_rag = str(merged.get("rag_path") or "").strip()
env_rag = os.environ.get("AGENT_MEMORY_RAG_PATH", "").strip()
default_rag = ui_rag or env_rag
```

优先级：

```text
UI / 当前桥接配置
  ↓
AGENT_MEMORY_RAG_PATH
  ↓
DEFAULT_RAG_PATH（用户 DSH 数据目录）
```

`DEFAULT_RAG_PATH` 只在后端使用，不向前端 UI 透传。

---

## 4. 行为对比

### 修复前

| 场景 | 行为 |
|---|---|
| 全局 Python 缺少 chromadb | 桥接崩溃 |
| 存在旧 `AGENT_MEMORY_CONFIG` | Host 写入旧配置 |
| 存在旧 `rag_path` | 检索旧库 |
| 插件目录有 `.venv` | 不使用 |
| UI 中自定义路径 | 可能被旧配置覆盖 |

### 修复后

| 场景 | 行为 |
|---|---|
| 设置了 `DSH_DREAM_PYTHON` | 直接使用该解释器 |
| 插件/工作目录有 `.venv` | 优先使用 venv Python |
| venv 不存在 | 回退系统 `python` / `python3` |
| Host 有当前 profile | 使用 `DSH_DREAM_CONFIG` |
| 旧 `AGENT_MEMORY_CONFIG` 存在 | 仅作为直接库调用 fallback |
| UI 填写 `rag_path` | 最高优先级，原样使用 |
| UI 未填写 | 读取 `AGENT_MEMORY_RAG_PATH` |
| 环境变量也没有 | 使用用户 DSH 数据目录 |

---

## 5. 测试

### Python 测试

```text
97 passed
```

新增覆盖：

```text
test_current_host_config_beats_legacy_env
```

验证：

- `DSH_DREAM_CONFIG` 优先于旧 `AGENT_MEMORY_CONFIG`；
- 当前 Host 配置的 `enable_memory` 和 `rag_path` 不会被旧文件覆盖。

### JavaScript 测试

```text
node tests/holographic_events.test.mjs
holographic_events: PASS
```

### 语法检查

```text
node --check dsh_plugin/index.js
python -m py_compile dsh_plugin/turn_bridge.py
```

均通过。

---

## 6. 涉及文件

```text
dsh_plugin/index.js
dsh_plugin/turn_bridge.py
src/agent_memory/config.py
tests/test_config.py
docs/DEV_REPORT_ROBUSTNESS_2026-11-04.md
```

---

## 7. 已知边界

1. `DSH_DREAM_PYTHON` 指向不存在的路径时，Host 端不会静默回退；这是显式配置，应尽快暴露错误。
2. `turn_bridge.py` 的自愈依赖相邻 `.venv` 或当前工作目录 `.venv`；如果用户把虚拟环境放在完全无关的目录，需要设置 `DSH_DREAM_PYTHON`。
3. 自动预检索的 800ms 超时保持不变；首次冷启动可能跳过召回，但不会阻塞聊天。
4. 旧的全局 `AGENT_MEMORY_CONFIG` 不再影响 Host 当前配置，但建议用户手动清理，避免直接调用 Python 库时仍指向旧 checkout。

---

## 8. 验收清单

- [x] `DSH_DREAM_PYTHON` 优先级最高
- [x] 自动发现相邻 `.venv`
- [x] 系统 Python 作为最终回退
- [x] `turn_bridge.py` 启动自愈
- [x] Host 使用 `DSH_DREAM_CONFIG`
- [x] 旧 `AGENT_MEMORY_CONFIG` 不再劫持 Host
- [x] UI `rag_path` 优先于环境变量
- [x] 环境变量优先于用户目录默认值
- [x] Python 测试 97 passed
- [x] Holographic JS 测试通过
- [x] 本地提交，未 Push
