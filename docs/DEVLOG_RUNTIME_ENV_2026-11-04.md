# DSH-DREAM 运行环境与模型缓存修复日志

> 日期：2026-11-04  
> 版本：v0.3.0 运行环境补丁  
> 主题：市场安装后的 Python 依赖、模型缓存与运行环境可视化

---

## 1. 问题背景

从 DSH 官方市场安装后，插件位于只读的 profile `node_modules` 目录：

```text
~/.dsh/profiles/desktop/node_modules/...
```

开发机上可直接使用的：

```text
.venv
.hf-cache
```

在真实用户环境里并不存在，也不应该假设存在。

由此产生两个死局：

1. **Python 依赖缺失**  
   Host 端 `spawn` 落到系统全局 Python，缺少：
   - `chromadb`
   - `sentence-transformers`
   
   桥接进程直接 `ModuleNotFoundError` 退出。

2. **模型缓存不可写**  
   旧逻辑可能寻找插件目录下的 `.hf-cache`；市场安装目录只读，模型无法下载。

---

## 2. 修复目标

```text
可发现的 Python 解释器
  +
可写的用户级 HF 缓存
  +
清晰的运行环境错误反馈
  +
用户可填写的 Python 路径
```

最终目标是：市场安装后，不依赖作者本机的 `.venv` 或 `.hf-cache`。

---

## 3. Python 解释器定位

### 3.1 Host 端 `dsh_plugin/index.js`

新增 `resolvePython(configured)`：

```text
1. DSH_DREAM_PYTHON 环境变量
2. UI 配置中的 python_path
3. DSH_DREAM_ROOT / 插件根目录 / 当前工作目录下的 .venv
4. 系统 python / python3
```

Windows：

```text
<root>\.venv\Scripts\python.exe
```

Unix：

```text
<root>/.venv/bin/python
```

启动桥接时使用解析后的解释器：

```js
const python = resolvePython(bridgeConfig?.python_path);
spawn(python, ["-u", TURN_BRIDGE], { ... });
```

同时向子进程注入：

```text
DSH_DREAM_PYTHON
DSH_DREAM_HF_CACHE
HF_HOME
HF_HUB_CACHE
```

### 3.2 桥接端 `dsh_plugin/turn_bridge.py`

保留自愈逻辑：

1. 检查当前解释器是否具备：
   - `agent_memory`
   - `chromadb`
   - `sentence_transformers`
2. 缺失时查找 `DSH_DREAM_PYTHON` 或相邻 `.venv`；
3. 找到后 `os.execv()` 重新执行；
4. 使用 `DSH_DREAM_REEXEC=1` 防止循环；
5. 找不到时不再静默退出，而是进入“依赖错误模式”。

---

## 4. 动态模型缓存

### 4.1 缓存位置

Host 端计算：

```js
const HF_HOME = resolve(
  process.env.DSH_DREAM_HF_CACHE || resolve(PROFILE_DIR, ".hf-cache"),
);
```

默认示例：

```text
C:\Users\<user>\.dsh\profiles\desktop\.hf-cache
```

不再使用：

```text
<plugin-dir>/.hf-cache
```

### 4.2 注入给 Python

```js
HF_HOME,
HF_HUB_CACHE: resolve(HF_HOME, "hub"),
HF_HUB_OFFLINE: "0",
TRANSFORMERS_OFFLINE: "0",
```

### 4.3 `turn_bridge.py` 兜底

```python
configured = DSH_DREAM_HF_CACHE or HF_HOME
fallback = ~/.dsh/profiles/desktop/.hf-cache
```

并尝试创建目录：

```python
_HF_CACHE.mkdir(parents=True, exist_ok=True)
```

因此市场安装后模型可以下载到用户目录，而不是写入只读的 `node_modules`。

---

## 5. 依赖错误反馈

### 5.1 桥接端

`turn_bridge.py` 不再因为顶层 import 失败而直接崩溃。

改为：

```python
_PLUGIN_IMPORT_ERROR = None
try:
    from agent_memory.plugin_entry import get_plugin
except Exception as exc:
    _PLUGIN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
```

进入错误模式后，对每个请求返回：

```json
{
  "request_id": "...",
  "ok": false,
  "code": "PYTHON_DEPENDENCY_ERROR",
  "error": "Python environment is missing required DSH-DREAM packages.",
  "detail": "ModuleNotFoundError: ...",
  "python": "C:\\...\\python.exe",
  "hf_home": "C:\\Users\\...\\.dsh\\profiles\\desktop\\.hf-cache",
  "hint": "Set DSH_DREAM_PYTHON ... or run pip install -e \".[dev]\" ..."
}
```

### 5.2 工具层

`memory_search` 工具收到依赖错误时，会把以下信息作为工具结果返回给模型/用户：

```text
Python environment error:
Python environment is missing required DSH-DREAM packages.
ModuleNotFoundError: ...
Python: ...
HF_HOME: ...
Set DSH_DREAM_PYTHON ...
```

自动预检索仍然 fail-open：依赖错误只会记录诊断，不阻塞聊天。

---

## 6. UI 新增：Python 解释器路径

### 6.1 `dsh_plugin/index.js` Config

新增：

```js
python_path: z
  .string()
  .default("")
  .description("Optional path to a Python interpreter with chromadb and sentence-transformers; leave blank to auto-detect an adjacent .venv or PATH.")
  .volatile(),
```

### 6.2 `dsh_plugin/client.js`

新增：

```text
Python 解释器路径
```

输入框：

- 可编辑；
- 留空时自动发现 `.venv` 或 PATH；
- 提示修改后重启 DSH；
- 保存到桥接配置 `python_path`；
- 同步到 DSH Host 设置状态。

### 6.3 `manifest.json`

两处配置 Schema 均新增：

```json
"python_path": {
  "type": "string",
  "title": "Python interpreter path",
  "description": "Optional absolute path ...",
  "default": "",
  "ui:widget": "text"
}
```

并加入 `required` 列表。

---

## 7. UI 新增：运行环境检查

新增只读状态：

```text
运行环境检查
```

Host 端新增：

```text
GET /dsh-dream/runtime
```

检查：

```text
agent_memory
chromadb
sentence_transformers
```

状态显示：

- 正在检查；
- Python 运行环境正常；
- 缺少 Python 依赖：`chromadb, sentence_transformers`；
- 无法检查运行环境。

该接口只返回依赖状态，不在 UI 回显用户绝对路径。

---

## 8. 配置优先级

### 8.1 配置文件

```text
显式 config_path
  → DSH_DREAM_CONFIG
  → AGENT_MEMORY_CONFIG（兼容 fallback）
  → 默认 profile 配置
```

### 8.2 RAG 路径

```text
DSH UI / 当前 agent_memory_config.json
  → AGENT_MEMORY_RAG_PATH
  → 用户 DSH 数据目录
```

### 8.3 Python

```text
DSH_DREAM_PYTHON
  → UI python_path
  → 相邻 .venv
  → PATH python/python3
```

---

## 9. 涉及文件

```text
dsh_plugin/index.js
dsh_plugin/turn_bridge.py
dsh_plugin/client.js
manifest.json
src/agent_memory/config.py
tests/test_config.py
docs/DEVLOG_RUNTIME_ENV_2026-11-04.md
```

---

## 10. 验证

### Python

```text
97 passed
```

### JavaScript

```text
node tests/holographic_events.test.mjs
holographic_events: PASS
```

### 语法

```text
node --check dsh_plugin/index.js
node --check dsh_plugin/client.js
python -m py_compile dsh_plugin/turn_bridge.py
```

均通过。

---

## 11. 已知边界

1. UI 修改 Python 路径后建议重启 DSH，因为已启动的桥接进程不会被热替换。
2. `DSH_DREAM_PYTHON` 指向无效路径时，Host 会如实失败；运行环境检查会显示异常。
3. 自动预检索仍为 800ms fail-open，首次冷启动可能跳过召回。
4. 模型下载仍需网络；离线环境需要预先在用户缓存目录准备模型。

---

## 12. 验收清单

- [x] 不再依赖插件目录 `.hf-cache`
- [x] HF_HOME 指向用户 DSH 数据目录
- [x] Python 解释器支持环境变量、UI、venv、PATH 四级解析
- [x] `turn_bridge.py` 依赖缺失时返回结构化错误
- [x] `memory_search` 工具能返回清晰依赖错误
- [x] UI 增加 Python 解释器路径输入框
- [x] UI 增加只读运行环境检查
- [x] Manifest/Schema 同步 `python_path`
- [x] 配置与 RAG 路径优先级保持
- [x] Python 测试 97 passed
- [x] JS 测试与语法检查通过
