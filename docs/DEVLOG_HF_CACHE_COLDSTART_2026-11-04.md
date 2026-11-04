# DSH-DREAM 模型缓存与冷启动重构日志

> 日期：2026-11-04  
> 版本：v0.3.0 运行环境补丁  
> 主题：可写缓存目录探测、模型缓存降级、首轮冷启动超时与静默预热

---

## 1. 问题背景

在市场安装场景中，插件位于只读的 profile `node_modules` 下：

```text
C:\Users\<user>\.dsh\profiles\desktop\node_modules\...
```

同时，用户 profile 目录在沙箱或受限权限环境中也可能不可写：

```text
WinError 5: Access is denied
```

这会导致：

1. `HF_HOME` 指向不可写目录，模型下载失败；
2. 本地 embedding 模型无法加载，自动召回完全不可用；
3. 冷启动加载模型需要 6–12 秒，超过原先 800ms 的预检索超时；
4. 用户看不到缓存错误原因，只能看到“没有召回”。

---

## 2. 缓存目录多级探测

### 2.1 Host 端 `dsh_plugin/index.js`

新增 `resolveCache(configured)`，按以下顺序寻找 **实际可写** 的目录：

```text
1. DSH_DREAM_HF_CACHE
2. UI 配置 hf_home_path
3. ~/.dsh/profiles/desktop/.hf-cache
4. <plugin-root>/.hf-cache
5. <cwd>/.hf-cache
```

每个候选目录都会执行真实写入测试：

```js
mkdirSync(directory, { recursive: true });
writeFileSync(probe, "ok", "utf8");
unlinkSync(probe);
```

任意一步失败，就继续尝试下一级。

### 2.2 Python 端 `dsh_plugin/turn_bridge.py`

`_cache_candidates()` 使用同样的降级顺序：

```text
DSH_DREAM_HF_CACHE
→ HF_HOME
→ 用户 DSH profile/.hf-cache
→ 插件目录/.hf-cache
→ 当前工作目录/.hf-cache
```

`_writable_cache()` 使用：

```python
path.mkdir(parents=True, exist_ok=True)
probe.write_text("ok", encoding="utf-8")
probe.unlink(missing_ok=True)
```

并捕获 `OSError` / `PermissionError`，包括 Windows `WinError 5`。

### 2.3 全部失败时

- Host 端返回明确中文错误：
  ```text
  模型缓存目录不可写，请手动在设置中指定目录
  ```
- UI 运行环境检查显示：
  ```text
  模型缓存目录不可写，请手动在设置中指定目录
  ```
- Python 端不再假装成功；诊断信息中包含：
  ```json
  {
    "hf_home": "",
    "cache_error": "模型缓存目录不可写，请手动在设置中指定目录"
  }
  ```

---

## 3. 新增 UI 配置

### 3.1 模型缓存目录 `hf_home_path`

`dsh_plugin/index.js` Config 新增：

```js
hf_home_path: z
  .string()
  .default("")
  .description("Optional writable model cache directory...")
  .volatile(),
```

`dsh_plugin/client.js` 新增输入框：

```text
模型缓存目录
```

提示：

```text
留空则依次尝试 DSH Profile、插件目录与当前工作目录下的 .hf-cache
```

保存链路：

```text
client.js 输入
  → HTTP /dsh-dream/config
  → agent_memory_config.json
  → resolveCache(hf_home_path)
  → HF_HOME
```

### 3.2 运行环境检查

新增只读状态：

```text
运行环境检查
```

Host 端 `GET /dsh-dream/runtime` 同时检查：

- Python 依赖：
  - `agent_memory`
  - `chromadb`
  - `sentence_transformers`
- 缓存目录可写性。

状态显示：

- 正在检查；
- Python 运行环境正常；
- 缺少 Python 依赖；
- 模型缓存目录不可写；
- 无法检查。

---

## 4. 首轮冷启动超时

### 4.1 动态超时

新增：

```js
let bridgeColdStart = true;
```

`queryMemory()` 中：

```js
const effectiveTimeout = bridgeColdStart ? 15000 : timeoutMs;
bridgeColdStart = false;
```

行为：

| 阶段 | 超时 |
|---|---:|
| 首次唤醒 | **15000ms** |
| 后续普通查询 | **800ms** |
| 手动 `memory_search` 工具 | 5000ms（首次仍可放宽到 15000ms） |

### 4.2 后台静默预热

插件激活时：

```js
ensureTurnBridge();
```

桥接进程启动后自动发送：

```json
{"op": "warmup", "request_id": "warmup"}
```

`turn_bridge.py` 收到后调用：

```python
get_plugin()
```

提前完成插件导入和本地模型加载，降低用户真正提问时的延迟。

---

## 5. 注入成功日志

`agent/pre-step` 成功注入后打印：

```text
[Holo] Pre-retrieval success: injected 3 memories
```

同时增加：

```text
[Holo] Pre-retrieval skipped: no matching memories
```

以及依赖错误时的诊断信息。

这样可以区分：

- 没有触发预检索；
- 触发了但无命中；
- 依赖/缓存失败；
- 成功注入了几条记忆。

---

## 6. 关键文件

| 文件 | 变更 |
|---|---|
| `dsh_plugin/index.js` | 缓存可写探测、`hf_home_path`、动态冷启动超时、静默预热、成功日志 |
| `dsh_plugin/turn_bridge.py` | 缓存多级降级、写测试、warmup 协议、缓存错误上报 |
| `dsh_plugin/client.js` | 模型缓存目录输入框、运行环境检查、缓存错误显示 |
| `manifest.json` | `hf_home_path` Schema 与 required |
| `docs/DEVLOG_HF_CACHE_COLDSTART_2026-11-04.md` | 本日志 |

---

## 7. 验证

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

### Manifest

确认：

```text
config.properties.hf_home_path
settings_schema.properties.hf_home_path
settings_schema.required 包含 hf_home_path
```

均存在。

---

## 8. 已知边界

1. 如果所有候选缓存目录都不可写，UI 会明确提示；需要用户填写可写目录。
2. 首次冷启动即使放宽到 15 秒，极端慢盘仍可能超时；后续请求恢复 800ms。
3. Python 解释器仍需具备 `chromadb` 与 `sentence-transformers`；可通过 UI 的 Python 路径字段指定。
4. 模型下载遵循 Hugging Face 网络访问；离线环境需要预置模型缓存。

---

## 9. 验收清单

- [x] 缓存目录真实写入探测
- [x] WinError 5 / PermissionError 自动降级
- [x] DSH_DREAM_HF_CACHE 最高优先级
- [x] UI `hf_home_path` 输入框
- [x] Profile → 插件目录 → 工作目录降级
- [x] 全部失败时 UI 明确报错
- [x] 首轮 15000ms / 后续 800ms
- [x] 插件启动后台静默预热
- [x] `[Holo] Pre-retrieval success: injected N memories`
- [x] Python 测试 97 passed
- [x] JS 测试与语法检查通过
