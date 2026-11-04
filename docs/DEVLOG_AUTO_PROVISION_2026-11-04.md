# DSH-DREAM 首次运行自动装配日志

> 日期：2026-11-04  
> 版本：v0.3.0 开箱即用补丁  
> 主题：自动创建 `.dream-venv`、安装 Python 依赖与 UI 装配状态

---

## 1. 目标

普通用户从 DSH 市场安装插件后，不应被要求提前准备：

- Python 虚拟环境；
- `chromadb`；
- `sentence-transformers`；
- 网络镜像配置；
- 手动设置解释器路径。

v0.3.0 补丁增加首次运行自动装配：

```text
检测依赖
  ↓ 缺失
创建 ~/.dsh/profiles/desktop/.dream-venv
  ↓
pip install chromadb sentence-transformers
  ↓
UI 提示重启 DSH
```

---

## 2. Host 端自动装配

修改文件：

```text
dsh_plugin/index.js
```

### 2.1 新增装配状态

```js
let provisionState = {
  status: "idle",       // idle | checking | provisioning | installed | ready | error
  python: "",
  venv: DREAM_VENV,
  error: "",
};
```

### 2.2 依赖检测

`ensureProvisioned()`：

1. 解析可写模型缓存目录；
2. 使用当前选中的 Python 执行 `RUNTIME_PROBE`；
3. 检查：
   - `agent_memory`
   - `chromadb`
   - `sentence_transformers`

### 2.3 自动创建 venv

默认路径：

```text
<DSH profile>/.dream-venv
```

例如：

```text
C:\Users\<user>\.dsh\profiles\desktop\.dream-venv
```

通过：

```text
python -m venv <DREAM_VENV>
```

创建。

### 2.4 自动安装依赖

通过新环境的 Python 执行：

```text
python -m pip install \
  --disable-pip-version-check \
  --no-input \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  chromadb sentence-transformers
```

使用清华 PyPI 镜像，降低国内网络超时概率。

### 2.5 装配完成后的行为

- 状态变为 `installed`；
- 结束当前桥接进程；
- 下一次启动优先使用 `.dream-venv`；
- UI 提示用户重启 DSH。

### 2.6 失败处理

- 如果所有缓存目录不可写：保留已有缓存错误提示；
- 如果 venv 创建失败或 pip 安装失败：
  - 记录 `provisionState.error`；
  - 不阻塞 DSH 主进程；
  - 保留手动指定 Python 路径的入口。

---

## 3. 桥接端兜底

修改文件：

```text
dsh_plugin/turn_bridge.py
```

### 3.1 解释器自愈

`_python_candidates()` 现在优先寻找：

```text
<root>/.dream-venv/Scripts/python.exe
<root>/.venv/Scripts/python.exe
```

Unix：

```text
<root>/.dream-venv/bin/python
<root>/.venv/bin/python
```

### 3.2 独立于 Host 的装配兜底

新增：

```python
_auto_provision_if_needed()
```

当桥接进程没有被 Host 装配、且当前解释器缺库时：

1. 检查 `DSH_DREAM_HOST_PROVISION`，Host 已负责时跳过；
2. 检查 `DSH_DREAM_AUTO_PROVISION=0` 退出开关；
3. 创建 `~/.dsh/profiles/desktop/.dream-venv`；
4. 使用清华镜像安装：
   - `chromadb`
   - `sentence-transformers`
5. 安装完成后 `os.execv()` 进入新环境。

---

## 4. UI 状态

修改文件：

```text
dsh_plugin/client.js
```

运行环境检查现在支持：

```text
正在为首次运行装配依赖...
```

以及：

```text
依赖装配完成，请重启 DSH。
```

状态接口：

```text
GET /dsh-dream/runtime
```

返回：

```json
{
  "ok": false,
  "missing": ["chromadb", "sentence_transformers"],
  "provision": {
    "status": "provisioning",
    "venv": "C:\\Users\\...\\.dsh\\profiles\\desktop\\.dream-venv"
  }
}
```

前端在 `provisioning` 状态期间会每 3 秒刷新，直到：

- `installed`：提示重启；
- `ready`：显示运行正常；
- `error`：显示依赖/缓存错误。

---

## 5. UI 配置字段

已有字段继续保持：

```text
Python 解释器路径
模型缓存目录
```

自动装配失败时，用户仍可手动指定：

```text
DSH_DREAM_PYTHON
hf_home_path
```

无需重新安装插件。

---

## 6. 关键文件

```text
dsh_plugin/index.js
dsh_plugin/turn_bridge.py
dsh_plugin/client.js
manifest.json
docs/DEVLOG_AUTO_PROVISION_2026-11-04.md
```

---

## 7. 验证

### Python

```text
97 passed
```

### JavaScript

```text
node --check dsh_plugin/index.js
node --check dsh_plugin/client.js
node tests/holographic_events.test.mjs
```

均通过。

### 自动装配标记检查

```text
dream-venv
pypi.tuna.tsinghua.edu.cn
正在为首次运行装配依赖...
runtimeInstalled
DSH_DREAM_HOST_PROVISION
```

均存在。

---

## 8. 已知边界

1. 自动装配需要网络访问清华 PyPI 镜像；离线环境仍需要手动准备依赖。
2. pip 安装耗时取决于网络和包缓存，首次可能数分钟。
3. 装配完成后必须重启 DSH，让 Host 使用 `.dream-venv` 作为桥接解释器。
4. 如果用户显式指定了不可用的 Python 路径，自动装配可能失败；UI 会保留手动修正入口。
5. 自动装配不会删除或覆盖已有的 `.venv`、`.dream-venv` 中用户自行安装的包。

---

## 9. 验收清单

- [x] 检测 `chromadb` / `sentence-transformers`
- [x] 缺失时创建用户级 `.dream-venv`
- [x] 使用清华镜像自动安装依赖
- [x] Host 端后台装配
- [x] `turn_bridge.py` 独立兜底
- [x] UI 显示“正在为首次运行装配依赖...”
- [x] 装配完成提示重启 DSH
- [x] 失败不阻塞主进程
- [x] Python 测试 97 passed
- [x] JS 语法与 Holographic 测试通过
