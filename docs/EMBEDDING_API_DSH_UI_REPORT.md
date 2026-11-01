# DSH 外接 Embedding API 与维度防呆报告

- 日期：2026-10-27
- 状态：**PASS**
- 类型检查：**mypy Success**
- 全量测试：**87 passed**

## 一、修改的前端 / DSH 文件

### 1. `manifest.json`

新增并纳入 `config` 与 `settings_schema`：

- `embedding_provider`：`local` / `openai_compatible`
- `embedding_model`：默认空字符串
- `embedding_api_key`：密码型 string，`writeOnly: true`
- `embedding_base_url`：默认 `https://api.openai.com/v1`
- `auto_rebuild_collection`：默认 `true`

### 2. `dsh_plugin/cordis.patch.yml`

默认配置新增：

```yaml
embedding_provider: local
embedding_model: ''
embedding_api_key: ''
embedding_base_url: https://api.openai.com/v1
auto_rebuild_collection: true
```

### 3. `dsh_plugin/client.js`

新增：

- “Embedding 模型（高阶）”配置区块
- 红底/红色加粗警告：

```text
⚠️ 警告：更换 Embedding 模型会改变向量维度！修改此项后，系统将尝试自动新建集合（或要求您手动清空旧 RAG 数据库），否则新记忆将无法与旧记忆一起被检索！
```

- `embedding_provider` 下拉选择
- `embedding_model` 文本输入
- `embedding_api_key` **`type="password"`**，前端脱敏
- `embedding_base_url` 文本输入
- `auto_rebuild_collection` 复选框，默认勾选
- 保存时同步 `scope.mutate` 和 `/dsh-dream/config` HTTP bridge
- 客户端日志中的 API Key 脱敏为 `***`

### 4. `dsh_plugin/index.js`（Host 半）

新增 Config 字段和 bridge 归一化：

- 读取/去除四个 Embedding 字段首尾空格
- 规范化 provider
- 默认 base URL
- 默认自动重建为 true
- `diag`、`console.log`、HTTP bridge 日志全部对 `embedding_api_key` 脱敏，避免写入 `agent_memory_plugin.log`

## 二、后端修改

### `src/agent_memory/config.py`

`MemoryConfig` 新增：

```python
embedding_provider: str = "local"
embedding_model: str = ""
embedding_api_key: str = ""
embedding_base_url: str = "https://api.openai.com/v1"
auto_rebuild_collection: bool = True
```

`load_config()` 对 model / key / base URL 做 `strip()`，provider 只接受 `local` 或 `openai_compatible`。

### `src/agent_memory/plugin_entry.py`

`RAGStore` 构造时传入全部 Embedding 配置和 `auto_rebuild_collection`，面板保存后生效。

### `src/agent_memory/rag_store.py`

新增 `OpenAICompatibleEmbeddingFunction`：

- 标准库 `urllib.request` 调用 `{base_url}/embeddings`
- Bearer API Key
- 校验响应数量和向量维度
- `get_config()` **不包含 API Key**
- 不把 Key 写入 Chroma collection metadata

新增维度防呆：

1. 用 `"dimension_check"` 探测当前模型维度。
2. 读取旧集合 `embedding_dimension` metadata；旧集合没有该字段时读取第一条 embedding 的维度。
3. 维度不一致且勾选自动重建：
   - 打印 `[WARNING] 维度不匹配，正在自动清空旧集合...`
   - `client.delete_collection(name)`
   - 用新 embedding 重建
4. 维度不一致且未勾选：
   - 抛出 `RAGStoreError`：

```text
向量维度从 X 变成了 Y！请手动清空 .chroma 数据库，或在面板勾选自动重建！
```

5. 新集合 metadata 写入 `embedding_dimension`，供下次启动直接比较。

### `src/agent_memory/memory_agent.py`

仅做类型检查修复：将 `_generate()` 的 `str | None` 先存入 `generated_reply`，避免 mypy 将 `reply` 推断为不可为 None。

## 三、测试

### 类型检查

```text
.venv\Scripts\python.exe -m mypy src\agent_memory
Success: no issues found in 10 source files
```

### JS 语法检查

```text
node --check dsh_plugin/client.js  -> 0
node --check dsh_plugin/index.js   -> 0
```

### 全量测试

```text
87 passed in 1.87s
```

新增测试：

- `tests/test_embedding_dimension.py`
  - 维度不匹配且不自动重建：明确报错
  - 维度不匹配且自动重建：旧数据清空，新集合可写
- `tests/test_embedding_http.py`
  - Mock HTTP 请求校验 URL、Bearer、payload、响应排序
  - 验证 API Key 不进入 `get_config()`
- `tests/test_config.py`
  - 新字段默认值和首尾空格处理

### 默认 BAAI 冒烟

```text
model BAAI/bge-small-zh-v1.5
provider local
dim_meta 512
count 1
```

## 四、验证边界

- 当前 DSH profile 中没有检测到已安装的 `@local/dsh-dream-settings` entry，因此本次没有做浏览器实拍；已完成 schema、JS 语法、Host bridge 和后端测试。
- 没有真实 OpenAI/DeepSeek API Key，因此云端请求使用 mock HTTP 单元测试验证；实际切换时由面板填写 Key 和 Base URL。
- 切换 Embedding 模型会改变向量维度；自动重建会删除旧 Chroma collection，README 已提示备份 `.chroma`。

## 五、清理

- `.test-tmp/` 已删除
- pytest 缓存和 `__pycache__` 已清理
- 没有残留临时数据库或测试脚本
- `.venv` 保留，依赖 `pip check` 通过
