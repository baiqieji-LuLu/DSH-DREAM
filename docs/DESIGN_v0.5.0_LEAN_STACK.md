# DSH-DREAM v0.5.0 轻量化架构设计

> 状态：设计稿，不写实现代码  
> 目标：去掉 torch / sentence-transformers 重依赖，降低首次安装体积与超时概率  
> 保留：ChromaDB + BGE 业务语义、RRF 混合检索、全息记忆、时间线模式

---

## 1. 结论摘要

方案可行，但不建议在 v0.5.0 一次性同时替换“推理引擎”和“向量数据库”。

推荐分阶段：

```text
v0.5.0：
  ONNX Runtime + tokenizers 替换 sentence-transformers / torch
  + 引入 StorageBackend 抽象
  + sqlite-vec 作为默认轻量后端
  + Chroma 保留为 legacy / 迁移后端

后续：
  根据 10k / 100k 压测决定是否彻底删除 Chroma
```

推荐默认依赖：

```text
onnxruntime
tokenizers
numpy
rank-bm25
jinja2
sqlite-vec
```

不再默认安装：

```text
torch
sentence-transformers
chromadb
```

---

## 2. 为什么替换

当前本地 embedding 链路：

```text
BGE
  → sentence-transformers
  → PyTorch
```

问题：

- torch 体积大，首次安装容易超时；
- sentence-transformers 与 torch 版本耦合；
- 新版 `SentenceTransformer` 参数兼容性不稳定；
- 用户需要为“只做一次 embedding”安装完整训练/推理框架。

目标链路：

```text
BGE tokenizer
  → tokenizers
  → ONNX Runtime CPU
  → mean pooling
  → L2 normalize
```

保留能力：

- BGE query instruction；
- 512 维向量；
- 云端 OpenAI-compatible embedding；
- 维度变化后的无损重嵌入。

---

## 3. Embedding 层设计

### 3.1 新模块

```text
src/agent_memory/embedding/
  base.py          # EmbeddingBackend protocol
  onnx_bge.py      # tokenizers + onnxruntime
  cloud.py         # 现有云端 embedding
  legacy_st.py     # 可选 sentence-transformers 回退
```

### 3.2 接口

```python
class EmbeddingBackend(Protocol):
    dimension: int
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
```

### 3.3 ONNX 流程

1. 加载 `tokenizer.json`；
2. 文本 padding / truncation 到 512；
3. 生成：
   - `input_ids`
   - `attention_mask`
   - 必要时 `token_type_ids`
4. ONNX Runtime CPU 推理；
5. 使用 attention mask 做 mean pooling；
6. L2 normalize；
7. 查询侧加入 BGE instruction，文档侧不加。

### 3.4 模型资产

建议单独下载，不塞进 Python 包：

```text
bge-small-zh-v1.5/
  model.onnx
  tokenizer.json
  config.json
```

优先考虑 int8 量化 ONNX，兼顾体积和精度。

### 3.5 验收标准

```text
同一批 100 条文本：
sentence-transformers embedding
vs
ONNX embedding
```

要求：

- 余弦相似度 > 0.999；
- 检索 Top-K 基本一致；
- 查询 instruction 行为一致；
- 无 torch 时仍可导入和检索。

---

## 4. Storage 层设计

### 4.1 抽象接口

`RAGStore` 不再直接依赖 Chroma，改为：

```python
class VectorBackend(Protocol):
    def upsert(self, records: list[MemoryRecord]) -> None: ...
    def search(self, vector, k, filters=None) -> list[VectorHit]: ...
    def get(self, ids) -> list[MemoryRecord]: ...
    def delete(self, ids) -> None: ...
    def count(self) -> int: ...
    def iter_all(self, batch_size=1000): ...
    def close(self) -> None: ...
```

### 4.2 后端选择

```text
storage_backend = "sqlite-vec" | "lance" | "chroma"
```

推荐：

```text
默认：sqlite-vec
可选：lance（大数据量）
legacy：chroma（迁移旧库）
```

### 4.3 sqlite-vec 方案

#### 优点

- 纯 SQLite，依赖小；
- 一个 `.db` 文件，备份简单；
- 无 Chroma 全局 system cache / 文件锁问题；
- 与当前“20 万以内精确 Flat”策略匹配；
- 元数据、时间戳、来源类型可直接 SQL 管理。

#### 缺点

- 没有成熟 HNSW/IVF；
- 20–50 万以上性能需实测；
- Windows 下需验证扩展加载；
- 并发写入需要 WAL / busy_timeout / 单写者约束。

#### 表结构

```text
memories
  id TEXT PRIMARY KEY
  content TEXT
  source_type TEXT
  created_at TEXT
  timestamp_ms INTEGER
  content_sha256 TEXT
  metadata_json TEXT

vec_memories (sqlite-vec virtual table)
  rowid
  embedding float[512]
```

查询：

```text
vec_memories MATCH ?
  → rowid + distance
  → JOIN memories
  → metadata / source_type / timeline filter
```

### 4.4 LanceDB 方案

#### 优点

- 原生向量表；
- 支持较大规模；
- 支持 metadata filter；
- 无 Chroma 锁问题。

#### 缺点

- 引入 `lance`、`pyarrow` 等依赖；
- 安装体积未必比 Chroma 小很多；
- Windows / Python 3.13 兼容性需验证；
- API 与当前同步式 `RAGStore` 需要适配。

#### 定位

```text
可选高性能后端，不作为默认“极轻”方案
```

---

## 5. RAGStore 改造边界

保留在 `RAGStore`：

- provenance 白名单；
- timestamp 生成；
- state-slot 冲突检测；
- timeline mode；
- RRF 合并；
- BM25；
- collection 维度校验；
- 自动重建逻辑。

移到 StorageBackend：

- 向量写入；
- 向量检索；
- ID / metadata 持久化；
- 删除；
- count；
- 全量遍历。

目标：

```text
RAGStore 负责“记忆语义”
StorageBackend 负责“如何存和查”
EmbeddingBackend 负责“如何向量化”
```

---

## 6. pyproject.toml 设计

### 6.1 核心依赖

```toml
dependencies = [
  "numpy>=1.26,<3",
  "onnxruntime>=1.17,<2",
  "tokenizers>=0.15,<1",
  "rank-bm25>=0.2.2",
  "jinja2>=3.1",
  "sqlite-vec>=0.1.6",
]
```

### 6.2 可选依赖

```toml
[project.optional-dependencies]

legacy-chroma = [
  "chromadb>=0.4.24,<0.5.0",
]

local-torch = [
  "torch>=2.0",
  "sentence-transformers>=2.7",
]

lance = [
  "lancedb>=0.5",
]

llm = [
  "openai>=1.0",
]
```

默认安装不再包含：

```text
torch
sentence-transformers
chromadb
```

---

## 7. 旧数据迁移

### 7.1 触发条件

```text
配置 storage_backend = sqlite-vec
且目标新库为空
且检测到旧 Chroma 目录
```

### 7.2 流程

```text
读取旧 Chroma collection
  → 分批 get(ids, documents, metadatas, embeddings)
  → 校验 source_type / timestamp / content_sha256
  → 写入 sqlite-vec
  → 写入 migration marker
  → 保留旧 Chroma 只读
```

### 7.3 要求

- 支持中断续传；
- 迁移前后 count 校验；
- 维度不一致时重新 embedding；
- 迁移失败自动回退 Chroma；
- 绝不自动删除旧库。

---

## 8. 启动与运行时

### 8.1 首次运行

```text
1. 检测 onnxruntime / tokenizers / sqlite-vec
2. 下载 BGE ONNX 与 tokenizer
3. 打开 sqlite-vec 数据库
4. 若检测到旧 Chroma，执行迁移
5. 启动 Python bridge
```

### 8.2 JS 环境自举

`runtime_env.js` 的 uv 安装列表更新为：

```text
onnxruntime
tokenizers
numpy
rank-bm25
jinja2
sqlite-vec
```

删除：

```text
torch
sentence-transformers
chromadb
```

`torch` / `sentence-transformers` 只在用户显式选择 legacy 本地模型时安装。

### 8.3 UI

新增只读状态：

```text
Embedding engine: ONNX Runtime
Vector backend: sqlite-vec
```

可保留环境检查接口，但不再显示 torch 安装状态。

---

## 9. 测试计划

### 9.1 Embedding 对齐

- ONNX vs sentence-transformers；
- 中文 / 英文 / 混合文本；
- 空文本 / 超长文本；
- 查询 instruction。

### 9.2 检索对齐

- 10k / 100k 数据；
- Flat + BM25 + RRF Top-K；
- timeline mode；
- state-slot 冲突合并；
- source_type 过滤；
- 删除 / 更新 / 重建。

### 9.3 迁移

- 旧 Chroma 记录数一致；
- ID / timestamp / metadata 完整；
- 迁移后检索结果一致；
- 中断恢复；
- 旧库保留。

### 9.4 安装体验

- 无 torch 环境导入成功；
- 模型下载体积与耗时；
- Windows / Python 3.10–3.13；
- 无 Chroma 时无 `code: 14`。

---

## 10. 风险与缓解

| 风险 | 等级 | 缓解 |
|---|---|---|
| ONNX 与 ST 向量偏差 | 中 | 固定 100 条 golden embedding 对齐测试 |
| sqlite-vec 大规模性能 | 中高 | 10k/100k 压测；超过 20 万切 LanceDB |
| sqlite-vec 平台加载 | 中 | Windows 下验证扩展加载与 wheel |
| LanceDB 体积 | 中 | 只作为 optional extra |
| 迁移中断 | 中 | marker、断点续传、旧库只读保留 |
| BGE ONNX 资产 | 中 | SHA256、许可证、模型缓存目录 |
| RAGStore 回归 | 高 | 先做 StorageBackend 抽象，再切默认后端 |

---

## 11. 实施顺序

### Phase 1：Embedding 替换

```text
ONNX + tokenizers
保留 Chroma
移除 torch / sentence-transformers 核心依赖
```

### Phase 2：StorageBackend 抽象

```text
RAGStore 不再直接依赖 Chroma
Chroma 变成 backend 之一
```

### Phase 3：sqlite-vec 默认

```text
新库默认 sqlite-vec
旧 Chroma 迁移
```

### Phase 4：LanceDB 可选

```text
大数据量后端
```

### Phase 5：Chroma 降级为 legacy extra

```text
默认安装不再包含 Chroma
```

---

## 12. 最终建议

v0.5.0 的目标应是：

```text
核心安装：
  onnxruntime + tokenizers + sqlite-vec
  + numpy + rank-bm25 + jinja2

可选：
  legacy-chroma（迁移旧库）
  lance（大数据量）
  local-torch（旧本地模型回退）
```

这样既能实现“首次安装从数百 MB 降到几十 MB”，又能保留旧用户迁移和回退路径。

**不建议在 v0.5.0 同时删除全部 Chroma 代码；先完成 ONNX + StorageBackend 抽象 + sqlite-vec 默认，再在后续版本彻底移除 Chroma。**
