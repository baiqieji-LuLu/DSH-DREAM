# DSH-DREAM v0.2.0 → 当前版本代码审计报告

> 审计日期：2026-11-04  
> 审计对象：`DSH-DREAM-v0.2.0-prealpha.zip` 与当前工作区  
> 审计方式：解压归档 + `difflib` 对比 + `agent_memory_plugin.log` 关键字检查  
> 说明：本报告只做审计，不修改源码，不执行 commit/push。

---

## 0. 结论先行

今天的环境自适应改造整体方向正确，解决了真实的发布阻断问题：

- 去掉了作者本机绝对路径；
- 增加了 Python 解释器、缓存目录的自适应；
- 增加了市场安装后的依赖自举；
- 修掉了 ChromaDB 重复客户端和旧数据库兼容问题；
- 增加了可编辑的 RAG/Python/Cache UI 和隐私保护。

但代码已经出现明显的“职责集中”风险：

- `dsh_plugin/index.js` 从 636 行增长到 1323 行；
- `dsh_plugin/turn_bridge.py` 从 144 行增长到 360 行；
- 同一个“找 Python / 找缓存 / 找配置”的逻辑在 JS 和 Python 各写了一遍；
- `rag_store.py` 本身已有 1900 行左右，今天又叠加了单例与重试逻辑。

**判断：功能上值得保留，结构上必须在下一次大功能前拆分；目前属于“中度偏高的史山风险”，不是不可维护，但继续堆补丁会快速恶化。**

---

## 1. 比对方法与基础数据

### 1.1 归档

```text
DSH-DREAM-v0.2.0-prealpha.zip
大小：4,742,291 bytes
条目：129
解压目录：.tmp_diff_v020/
```

归档是“无顶层目录”的结构，核心文件直接位于：

```text
dsh_plugin/index.js
dsh_plugin/turn_bridge.py
src/agent_memory/rag_store.py
src/agent_memory/config.py
pyproject.toml
```

### 1.2 行数变化

| 文件 | v0.2.0 行数 | 当前行数 | 变化 |
|---|---:|---:|---:|
| `dsh_plugin/index.js` | 636 | 1,323 | **+687（约 2.08 倍）** |
| `dsh_plugin/turn_bridge.py` | 144 | 360 | **+216（约 2.5 倍）** |
| `src/agent_memory/rag_store.py` | 1,889 | 1,932 | +43 |
| `src/agent_memory/config.py` | 295 | 335 | +40 |
| `pyproject.toml` | 118 | 118 | 0（内容有变更） |

### 1.3 差异规模

| 文件 | 变更 hunk | 新增行 | 删除行 |
|---|---:|---:|---:|
| `dsh_plugin/index.js` | 25 | 712 | 25 |
| `dsh_plugin/turn_bridge.py` | 11 | 246 | 30 |
| `src/agent_memory/rag_store.py` | 7 | 70 | 27 |
| `src/agent_memory/config.py` | 9 | 57 | 17 |
| `pyproject.toml` | 2 | 2 | 2 |

### 1.4 MD5 对照

| 文件 | v0.2.0 MD5 | 当前 MD5 |
|---|---|---|
| `dsh_plugin/index.js` | `86d8c50c3908b7d96a97995d0ad7e7c3` | `3a9382da24a2c913f6ce9ee956e153c5` |
| `dsh_plugin/turn_bridge.py` | `2eb230ff3f3efce78b5d1ba9dbcc97dd` | `4575de9b3cf3e9c5b1666779f13793d7` |
| `src/agent_memory/rag_store.py` | `9852477c9fc3cfab12d7ca8e0f444c98` | `f8ab71dc3b78b4941414bc97ab1d97ad` |
| `src/agent_memory/config.py` | `f7e7d3f75243e7f5955d09aa94077228` | `0bb7b400bdd21d40daae7b9ab1cb3c99` |
| `pyproject.toml` | `3257ea6771c38bd0865b55093dbee042` | `ef22531f94faa8750bf7462657830485` |

---

## 2. 优秀保留项

这些改动解决了真实的开源发布问题，**后续重构时不应删除**。

### 2.1 隐私与 UI 脱敏

| 改动 | 价值 | 证据 |
|---|---|---|
| 默认 RAG 路径不在 UI 回显绝对路径 | 避免暴露用户名和本机目录 | `index.js` 的 `normalizePayload` 不再合成默认 `rag_path`；`client.js` 输入框默认空 |
| `rag_path` 可编辑 | 用户可以自定义数据库 | UI 输入框已恢复可编辑，保存链路包含 `scope.mutate` |
| `python_path` UI | 用户可指定带依赖的 Python | `Config`、`manifest.json`、`client.js` 三处均有字段 |
| `hf_home_path` UI | 用户可指定可写模型缓存 | 同上，并参与 `resolveCache` |
| 运行环境检查 | 用户能看到缺依赖/缓存不可写 | `GET /dsh-dream/runtime` + UI 状态 |
| 插件展示名/包名 | 避免 `@local/dsh-dream-settings` 出现在 UI | 包名 `dsh-dream`，`displayName` 与 `meta.title` |

这部分是整个改造中最值得保留的成果：它把“开发机专用插件”变成了“不会泄露作者路径、可让用户自行修正环境”的发布形态。

### 2.2 默认路径不再硬编码

v0.2.0 中仍有：

```js
const PROJECT_ROOT = "G:\\deepseek_ragproject_workspace";
```

当前已替换为：

```js
const PROJECT_ROOT = resolve(
  process.env.DSH_DREAM_ROOT || resolve(PLUGIN_DIR, ".."),
);
```

`turn_bridge.py` 也不再使用 `_CHECKOUT_ROOT`，改为 `Path(__file__).resolve().parent.parent`。

### 2.3 关键防回声逻辑

`<memory-hint>` 被同时加入：

- `holographic_events.js` 的 `INJECTED_MARKERS`
- `turn_bridge.py` 的 `_INJECTED_MARKERS`

这防止“自动召回内容再次入库”的污染循环，必须保留。

---

## 3. 环境自举与路径重构

### 3.1 Python 解释器探测

`dsh_plugin/index.js` 当前顺序：

```text
DSH_DREAM_PYTHON
→ 插件/Profile/工作目录下的 .dream-venv
→ .venv
→ 系统 python / python3
```

`turn_bridge.py` 也有对应的 `_python_candidates()` 与 `os.execv()` 自愈。

### 3.2 首次运行自动装配

当前新增：

```text
<DSH profile>/.dream-venv
```

通过：

```text
python -m venv .dream-venv
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple chromadb sentence-transformers
```

Host 与 Python 桥接各有兜底；Host 通过 `DSH_DREAM_HOST_PROVISION=1` 避免重复装配。

### 3.3 模型缓存多级降级

`index.js` 的 `resolveCache()`：

```text
DSH_DREAM_HF_CACHE
→ UI hf_home_path
→ profile/.hf-cache
→ plugin/.hf-cache
→ cwd/.hf-cache
```

每个候选都做真实写入探针，失败继续降级；Python 端有同样的 `_cache_candidates()`。

### 3.4 配置路径隔离

`config.py` 当前优先：

```text
显式 config_path
→ DSH_DREAM_CONFIG
→ AGENT_MEMORY_CONFIG
→ 默认配置
```

旧 `AGENT_MEMORY_CONFIG` 不再劫持当前 Host Profile。

### 3.5 RAG 路径严格化

`rag_path` 必须由 UI/桥接配置或 `AGENT_MEMORY_RAG_PATH` 显式提供：

```python
if not rag_text:
    raise ConfigError(...)
if not rag_path.is_absolute():
    raise ConfigError(...)
```

这解决了“UI 填了 `.chat_archive_test_chroma`，但旧回退连到 `.chroma`”的问题。

### 3.6 未发现的“自动修 numpy 冲突”

本次审计**没有发现**针对 numpy 冲突的自动修复代码。当前自动安装只覆盖：

```text
chromadb
sentence-transformers
```

如果 numpy 冲突仍存在，需要单独增加依赖解析/版本约束，不能把现有自动装配误认为已经处理。

---

## 4. 死锁与兼容性改动

### 4.1 Chroma PersistentClient 单例化

当前 `rag_store.py` 新增：

```python
_CHROMA_CLIENTS: dict[str, dict[str, Any]] = {}
_CHROMA_CLIENTS_LOCK = threading.RLock()
```

按 `persist_directory` 路径缓存客户端，并用引用计数管理 `close()`。

这解决的是：

```text
An instance of Chroma already exists ... with different settings
```

而不是简单地“不让 RAGStore 实例化两次”。不同数据库路径仍然可以各自拥有客户端。

### 4.2 强制 persistent Settings

首次创建客户端时优先传入：

```python
Settings(allow_reset=True, is_persistent=True)
```

如果首次打开遇到 code 14，会清理 system cache 后用同样的 persistent settings 重试一次。

### 4.3 sentence-transformers 兼容

v0.2.0 的 `rag_store.py` 传入：

```python
SentenceTransformer(..., local_files_only=True)
```

当前已移除，避免新版 `sentence-transformers` 报：

```text
unexpected keyword argument 'local_files_only'
```

如需离线，应使用：

```text
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

### 4.4 ChromaDB 版本锁定

`pyproject.toml`：

```toml
chromadb>=0.4.24,<0.5.0
```

目的是避开新版 Rust/SQLite 后端对旧数据库的 `code: 14 unable to open database file` 兼容问题。

---

## 5. 日志证据

对 `agent_memory_plugin.log` 的检查结果：

| 关键字 | 出现次数 |
|---|---:|
| `code: 14` | 0 |
| `ModuleNotFoundError` | 0 |
| `RAGStoreUnavailable` | 1 |
| `An instance of Chroma already exists` | 0 |
| `local_files_only` | 0 |
| `WinError 5` | 1 |

日志中可确认的早期真实故障是：

```text
Ignored error while caching commit hash for 'sentence-transformers/all-MiniLM-L6-v2':
[WinError 5] 拒绝访问。: 'C:\Users\74451\.cache'
```

随后：

```text
RAGStoreUnavailableError: cannot initialize embedding model 'all-MiniLM-L6-v2'
```

这说明：

- 模型缓存/权限问题确实存在；
- 之后加入 `HF_HOME` 用户目录、可写探针和缓存降级是合理的；
- 但当前日志没有记录最新的 `code: 14` 或 `ModuleNotFoundError`，这些应来自新的测试环境日志，不能把当前日志当作全部证据。

日志还包含：

```text
schemastery resolution failed
UnicodeEncodeError: 'utf-8' codec can't encode ... surrogates
```

这些是另外两类稳定性问题，目前不属于本次审计的主线，但建议保留在待办中。

---

## 6. 代码膨胀与史山风险

### 6.1 `dsh_plugin/index.js`：高风险

当前承担的职责包括：

```text
Config Schema
配置桥接写入
settings 事件同步
HTTP 配置接口
HTTP 运行环境检查
Python 解释器解析
HF 缓存解析/写入探针
.venv/.dream-venv 自动装配
pip 安装
turn bridge 生命周期
warmup / query 协议
memory_search 工具注册
agent/pre-step 自动召回
全息记忆事件处理
诊断日志
```

一个 1300 行的入口文件同时承担 UI 配置、进程管理、依赖安装、工具注册和预检索，已经不属于“单文件小插件”。

**建议拆分：**

```text
runtime_env.js       Python/缓存/venv 解析与装配
bridge_process.js    persistent bridge 生命周期与 query/warmup
memory_tools.js      defineTool + memory_search
pre_retrieval.js     agent/pre-step + memory-hint
holographic.js       现有全息事件
index.js             只做组装
```

### 6.2 `dsh_plugin/turn_bridge.py`：中高风险

从 144 行增长到 360 行，混合了：

```text
全息写入
查询协议
warmup 协议
依赖自检
venv 自动装配
缓存探测
错误上报
日志
```

建议至少拆成：

```text
bridge_protocol.py    stdin/stdout 协议
runtime_provision.py  venv/pip
memory_bridge.py      write/query/warmup
```

### 6.3 重复逻辑

当前 JS 与 Python 各有一套：

- Python 解释器候选；
- 缓存目录候选；
- 可写性探测；
- 环境变量优先级；
- 错误文案。

这类逻辑一旦修改，容易出现“JS 改了 Python 没改”的漂移。建议把候选规则固化为 JSON/配置或单独文档，并增加交叉测试。

### 6.4 `rag_store.py`：体量大但改动相对集中

`rag_store.py` 在 v0.2.0 已经有 1,889 行，本次只增加约 43 行。  
单例和 Settings 逻辑虽然重要，但集中在一个 1900 行模块里，后续继续叠加缓存、重试、重建逻辑会迅速变成高风险区。

**结论：`rag_store.py` 是“原本就大的老山”，`index.js`/`turn_bridge.py` 是“今天新长出来的山”。优先拆后者。**

---

## 7. 最终审计判断

### 值得保留

- 隐私路径脱敏；
- `python_path` / `hf_home_path` UI；
- RAG 路径可编辑；
- `.dream-venv` 自动装配；
- 缓存多级可写降级；
- `DSH_DREAM_CONFIG` 隔离旧配置；
- `<memory-hint>` 防回声；
- Chroma 客户端单例与 Settings 重试；
- sentence-transformers 参数兼容；
- ChromaDB 旧版本锁定。

### 需要警惕

- `index.js` 已经过载，继续加功能前应拆分；
- JS/Python 双份环境规则容易漂移；
- 自动装配涉及网络、venv、pip，失败路径需要真实 DSH 环境回归；
- 当前日志没有覆盖所有新错误关键字，测试证据需要补全；
- 未发现 numpy 冲突自动修复，不要把它算作已完成能力。

### 建议优先级

1. **先拆 `index.js`**（环境、桥接、工具、预检索四块）；
2. **再拆 `turn_bridge.py`**（协议、装配、记忆操作）；
3. 为缓存/Python 候选规则增加一份跨语言测试；
4. 用真实市场安装环境验证一次“空环境首次装配”；
5. 再考虑增加 numpy 依赖冲突的自动解析。
