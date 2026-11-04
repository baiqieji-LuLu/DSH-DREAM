# v0.5.0 Phase 1/2/3 实施报告：ONNX、Storage 解耦与 sqlite-vec

> 版本：v0.5.0
> 范围：ONNX 默认 embeddings、VectorBackend、sqlite-vec、Chroma 迁移
> 状态：ENV/ONNX/Storage 三项完成；无 torch 干净环境测试通过

---

## 1. Phase 1：ONNX Embedding

### 1.1 新增模块

```text
src/agent_memory/embedding/
  __init__.py
  base.py
  onnx_bge.py
```

### 1.2 `EmbeddingBackend`

定义：

```python
class EmbeddingBackend(Protocol):
    dimension: int
    model_name: str
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
```

### 1.3 `OnnxBgeEmbedding`

实现内容：

- `tokenizers.Tokenizer.from_file()`；
- `onnxruntime.InferenceSession`；
- `input_ids` / `attention_mask` / 可选 `token_type_ids`；
- BGE query instruction；
- 支持 `cls` / `mean` pooling；
- L2 normalize；
- 空输入、2D/3D 输出、不同 provider 的基本处理。

> 备注：BGE 的 sentence-transformers 配置实际使用 **CLS pooling**，不是 mean pooling；因此默认 `pooling="cls"`，同时保留 `mean` 选项。

### 1.4 RAGStore 接入

`RAGStore` 新增可选参数：

```python
embedding_backend: Any | None = None
```

当传入 backend 时：

- `_embedding_function` 使用 `backend.embed_documents`；
- `_model_name` 使用 backend 的模型名；
- `_probe_embedding_dimension()` 直接使用 `backend.dimension`；
- `_search_flat()` 使用 `backend.embed_query()`；
- HNSW 查询使用 `query_embeddings`，避免二次拼接 instruction。

保留原有 `embedding_function` / `SentenceTransformer` 路径作为回退。

### 1.5 验证脚本

新增：

```text
scripts/verify_onnx_embedding.py
```

用法：

```text
python scripts/verify_onnx_embedding.py \
  --onnx <model.onnx> \
  --tokenizer <tokenizer.json> \
  --model-dir <bge-model-dir> \
  --limit 100
```

脚本要求：

- 100 条文本；
- ONNX vs sentence-transformers；
- 余弦相似度全部 > 0.999；
- 不达标时退出码 1。

### 1.6 实际验收结果

本次用本地 BGE 模型导出 ONNX 后运行：

```text
shape: (100, 512) vs (100, 512)
cosine: min=0.99999988 mean=1.00000000 max=1.00000024
threshold 0.999: failures=0/100
```

**结论：100 条长文本全部通过 >0.999 验收。**

根因与修复：

- BGE 的 `tokenizer.json` 中 `lowercase=false`；
- sentence-transformers 加载 BGE 时实际使用 `do_lower_case=True`；
- 原生 tokenizers 因此把 "English" 等大写词切成 `[UNK]`，导致长文本约 0.96 偏差；
- 在 `OnnxBgeEmbedding` 中显式设置：
  ```python
  BertNormalizer(
      clean_text=True,
      handle_chinese_chars=True,
      strip_accents=None,
      lowercase=True,
  )
  ```
- 修复后短文本、长文本、100 条批量全部对齐。

BGE 的 pooling 配置已确认：

```json
{
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false
}
```

因此默认使用 **CLS pooling**；mean pooling 仍作为可选模式保留。

### 1.7 ONNX 默认引擎

`RAGStore` 默认：

```python
embedding_engine="onnx"
```

当以下条件同时成立时，优先加载 ONNX：

- 没有显式传入 `embedding_function`；
- 没有显式传入 Chroma `client`；
- `embedding_provider == "local"`；
- 本地找到 ONNX 资产。

ONNX 资产查找顺序：

```text
DSH_DREAM_ONNX_MODEL + DSH_DREAM_ONNX_TOKENIZER
→ DSH_DREAM_ONNX_DIR
→ ~/.dsh/profiles/desktop/.dream-models/<model>/
→ ~/.dsh/.dream-models/<model>/
→ <project>/models/<model>/
```

资产缺失或加载失败时，才回退 sentence-transformers。

`MemoryConfig.embedding_engine` 默认也是 `"onnx"`，可显式配置：

```text
"onnx"
"sentence-transformers"
```

---

## 2. Phase 2：StorageBackend 抽象

### 2.1 新增模块

```text
src/agent_memory/storage/
  __init__.py
  base.py
  chroma_backend.py
```

### 2.2 `VectorBackend` Protocol

接口包含：

```text
add
query
get
delete
count
set_metadata
close
collection_name
```

### 2.3 `ChromaVectorBackend`

把以下 Chroma 操作封装到 backend：

- `collection.add`；
- `collection.query`；
- `collection.get`；
- `collection.delete`；
- `collection.count`；
- `collection.modify`。

### 2.4 RAGStore 改造

- `RAGStore.__init__` 末尾创建：

```python
self._backend = ChromaVectorBackend(self._collection, self._collection_name)
```

- 新增、批量新增、向量查询、语料读取、按 ID 获取、删除、count 已改为调用 `self._backend`。
- `_collection` 仍保留为兼容别名，供现有测试和迁移工具访问。
- Chroma 的 collection 创建、维度迁移和 auto-rebuild 仍在 `RAGStore` 中，尚未完全迁入 backend。

### 2.5 Phase 3：sqlite-vec 默认后端

新增：

```text
src/agent_memory/storage/sqlite_backend.py
tests/test_sqlite_backend.py
```

实现：

- float32 embedding BLOB；
- metadata JSON（timestamp_ms / source_type）；
- add / upsert / query / search / get / delete / count；
- SQLite WAL + busy_timeout；
- `sqlite_vec.load()` 扩展加载。

默认策略：

```python
RAGStore(storage_backend="sqlite-vec")
```

- 未显式传入 `client` / `embedding_function` 时默认 sqlite-vec；
- 显式传入 Chroma client 或 embedding_function 时继续使用 Chroma；
- `MemoryConfig.storage_backend` 默认 `"sqlite-vec"`，支持显式 `"chroma"`；
- `plugin_entry` 会把该配置传给 RAGStore。

### 2.6 Chroma → sqlite-vec 自动迁移

新增：

```text
src/agent_memory/storage/migrate.py
scripts/migrate_chroma_to_sqlite.py
```

逻辑：

- 检测旧目录下 `chroma.sqlite3`；
- 读取 Chroma `agent_memory` collection；
- 分批读取 `ids / documents / metadatas / embeddings`；
- 以 float32 写入 sqlite-vec；
- 生成 marker：
  ```text
  .agent_memory.chroma-migrated.json
  ```
- marker 存在时跳过迁移；
- 旧 Chroma 数据不删除，失败不会破坏原库。

`RAGStore` 在 sqlite 初始化时会自动调用 `migrate_if_needed()`；也可以手动执行：

```text
python scripts/migrate_chroma_to_sqlite.py \
  --chroma-path <old/.chroma> \
  --sqlite-path <new/agent_memory.sqlite3>
```

### 2.7 测试结果

无 torch 干净虚拟环境：

```text
97 passed, 1 skipped
```

跳过项：

```text
openai not installed
```

与本次改动无关。

---

## 3. 版本与依赖

版本统一为：

```text
0.5.0
```

涉及：

```text
package.json
dsh_plugin/package.json
manifest.json
pyproject.toml
src/agent_memory/__init__.py
README*.md
```

核心依赖已切换为轻量栈：

```toml
dependencies = [
  "numpy>=1.26,<2",
  "onnxruntime>=1.17,<2",
  "tokenizers>=0.15,<1",
  "sqlite-vec>=0.1.6",
  "jinja2>=3.1.0",
  "rank-bm25>=0.2.2",
]
```

旧重依赖移入可选 `legacy`：

```toml
legacy = [
  "chromadb>=0.4.24,<0.5.0",
  "torch>=2.0",
  "sentence-transformers>=2.7.0",
]
```

`runtime_env.js` 自动装配列表已改为：

```text
onnxruntime tokenizers sqlite-vec rank-bm25 jinja2 numpy<2
```

不再自动安装 torch 或 sentence-transformers。

---

## 4. 当前结论

### 已完成

- EmbeddingBackend 接口；
- ONNX BGE backend；
- ONNX 与 sentence-transformers 100 条对齐通过；
- ONNX 为本地默认 embedding engine；
- ONNX 资产缺失时自动回退 sentence-transformers；
- RAGStore 接受注入 embedding backend；
- VectorBackend 接口；
- ChromaVectorBackend；
- SqliteVecBackend；
- RAGStore 默认 sqlite-vec（显式 client/embedding_function 时保留 Chroma）；
- `MemoryConfig.storage_backend` 默认 sqlite-vec；
- `MemoryConfig.embedding_engine` 默认 onnx；
- Chroma → sqlite-vec 自动迁移与 marker；
- runtime_env 不再安装 torch / sentence-transformers；
- 核心依赖不再包含 torch / sentence-transformers；
- 无 torch 干净环境 97 项测试通过；
- 版本升级到 0.5.0。

### 仍需保留的边界

- ONNX 模型资产需要用户自行放置或下载；
- Chroma collection 生命周期尚未完全迁入 ChromaVectorBackend；
- sqlite-vec 的 10k/100k 性能仍需压测；
- 旧 Chroma 迁移依赖可选 `legacy` 依赖执行，迁移本身不删除旧数据。

---

## 5. 建议下一步

1. 为 ONNX 模型资产增加官方下载/校验流程；
2. 对 sqlite-vec 做 10k/100k 召回与延迟压测；
3. 将 Chroma collection 生命周期完全迁入 backend 或彻底移除；
4. 在真实用户旧库上验证迁移 marker 与重复启动幂等性。
