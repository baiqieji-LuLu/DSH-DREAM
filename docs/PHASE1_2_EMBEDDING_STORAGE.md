# v0.5.0 Phase 1/2 实施报告：Embedding 与 Storage 解耦

> 版本：v0.5.0  
> 范围：Phase 1 ONNX embedding + Phase 2 VectorBackend 抽象  
> 状态：代码实现完成；**ONNX 与 sentence-transformers 的 100 条对齐验收未达到 >0.999 阈值，需要继续排查**

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
cosine: min=0.95702600 mean=0.96510100 max=0.97076094
threshold 0.999: failures=100/100
```

**结论：未达到 >0.999 的验收标准。**

已确认：

- ONNX 输出与手动 PyTorch `AutoModel` 的 CLS 输出余弦为 1.0；
- `sentence-transformers` 的 `encode()` 与手动 PyTorch 输出在部分测试中为 1.0；
- 但 ONNX 与 `sentence-transformers.encode()` 的批量结果稳定在约 0.96；
- 该差异需要继续定位（tokenizer padding、ST `encode` 内部预处理或 ONNX 图输入对齐）。

在差异解决前，不能宣称 Phase 1 验收通过，也不应把 ONNX 设为默认 embedding 引擎。

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

### 2.5 测试结果

```text
96 passed, 1 skipped
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

`pyproject.toml` 新增可选 extra：

```toml
onnx = [
  "onnxruntime>=1.17,<2",
  "tokenizers>=0.15,<1",
]
```

核心依赖仍保留当前的 Chroma + sentence-transformers 路径，因为 Phase 1 验收尚未通过。

---

## 4. 当前结论

### 已完成

- EmbeddingBackend 接口；
- ONNX BGE backend；
- RAGStore 接受注入 backend；
- VectorBackend 接口；
- ChromaVectorBackend；
- RAGStore 的常用数据 I/O 走 backend；
- 96 项测试通过；
- 版本升级到 0.5.0。

### 未完成 / 阻塞

- ONNX 与 sentence-transformers 的 100 条对齐 **未达到 >0.999**；
- Chroma collection 生命周期尚未完全迁入 ChromaVectorBackend；
- Phase 1 暂不能作为默认 embedding 引擎发布；
- 需要继续定位 ONNX/ST 差异后重新跑验收。

---

## 5. 建议下一步

1. 对 ONNX 与 ST 分别导出 **同一 tokenizer 输入**的 hidden state，逐层比对；
2. 固定 padding/truncation 参数，排查 `tokenize()` 与 `encode()` 的差异；
3. 用单条文本确认 ST 是否应用了额外 prompt/instruction；
4. 通过后再将 ONNX 设为可选默认，并开始 sqlite-vec Phase 3。
