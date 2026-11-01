# DSH-DREAM 1000 轮压力测试报告

- 生成时间：2026-10-26T16:09:38+08:00
- 开始时间：2026-10-26T16:09:23+08:00
- 测试规模：1000 轮对话 / 第 1001 轮查询
- 向量后端：real chromadb + sentence-transformers all-MiniLM-L6-v2
- 总体结论：**PASS**

## 1. 执行时间与耗时

| 项目 | 值 |
| --- | --- |
| 总耗时 | 14.273 秒 |
| 第 1001 轮检索耗时 | 9.906 ms |
| 写入 RAG 的记忆数量 | 1001 |
| 初始对话历史 | 2000 条 |
| 最终窗口 | 4 条 |

## 2. 窗口截断验证

| 检查项 | 结果 |
| --- | --- |
| 初始历史 | 2000 条（1000 轮 × 2 条消息） |
| 窗口 Token 预算 | 80 |
| 最终窗口保留 | 4 条 |
| 第 50 轮标记 `WINDOW-MARKER-0050` 在窗口内 | **False** |
| 第 50 轮敏感事实在窗口内 | **False** |
| 第 50 轮记忆是否在窗口内（应为 False） | **False** |
| 最新一轮（第 1000 轮）是否保留 | **True** |

## 3. 极限召回验证

| 项目 | 值 |
| --- | --- |
| 查询轮次 | 第 1001 轮 |
| 查询文本 | `请调用 memory_search 帮我查一下我的银行卡密码是多少？` |
| 检索耗时 | 9.906 ms |
| 是否成功召回 `888888` | **True** |
| 命中记忆 ID | `mem-8a0c0faa4c6846058c4475231edd82ed` |
| 命中来源 | `system_fact` |
| 检索结果 ID 列表 | `['mem-8a0c0faa4c6846058c4475231edd82ed', 'mem-ae8787463c8945aba0b0824331174589', 'mem-cb4cd5f42c5d4a8a83598fd4bd9680fb']` |

召回的完整文本：

> 我的银行卡密码是 888888，绝对不能忘记

Prompt 注入验证：

| 检查项 | 结果 |
| --- | --- |
| 敏感事实进入最终 Prompt | **True** |
| RAG Hint 位于用户查询之后 | **True** |
| 审计快照 ID | `1` |
| 审计记录数量 | 1 |

## 4. 最终 Prompt 体积

| 项目 | 值 |
| --- | --- |
| 最终 Prompt Token 估算 | **401** |
| 最终 Prompt 字符数 | 1504 |
| 窗口 Token 预算 | 80 |
| 是否保持可控（≤ 1000 tokens） | **True** |

即使已经历 1000 轮对话，最终 Prompt 仍然只包含窗口内的近期消息和少量召回结果。

## 5. 清理确认

| 检查项 | 结果 |
| --- | --- |
| 临时目录 | `G:\deepseek_ragproject_workspace\.stress-tmp-85c46c8bd61f4c5dabf9f24af7c274ed` |
| 临时目录已删除 | **True** |
| 根目录测试数据库残留 | **无** |
| 所有数据库位于临时目录 | **True** |

本次测试的 `anchors.json`、`audit.db` 以及 Chroma 目录均位于临时目录中；清理后未在项目根目录留下
测试数据库。

## 6. 测试日志

- 临时目录创建：G:\deepseek_ragproject_workspace\.stress-tmp-85c46c8bd61f4c5dabf9f24af7c274ed
- 第 1 轮：继续收集日常闲聊
- 第 50 轮：写入关键 system_fact（888888 已隐藏在大海中）
- 第 100 轮：继续收集日常闲聊
- 第 500 轮：继续收集日常闲聊
- 第 1000 轮：继续收集日常闲聊
- 批量写入 1000 条日常闲聊，耗时 1.277 秒
- 第 1001 轮：检索完成，耗时 9.906 ms
- 第 1001 轮：Prompt 注入完成，Token 数 401
- 审计快照写入：1
- 释放内存组件，准备清理临时目录

## 7. 检查项汇总

| 检查项 | 结果 |
| --- | --- |
| 1000 轮共 2000 条历史 | PASS |
| 窗口只保留近期消息 | PASS |
| 第 50 轮标记不在窗口 | PASS |
| 第 50 轮事实不在窗口 | PASS |
| 最新一轮被保留 | PASS |
| RAG 中包含 1001 条记忆 | PASS |
| 成功召回 888888 | PASS |
| 敏感事实注入 Prompt | PASS |
| RAG Hint 位于查询之后 | PASS |
| Prompt 体积可控 | PASS |
| 临时目录已清理 | PASS |
| 根目录无数据库残留 | PASS |

## 8. 复现方式

```bash
python examples/stress_test.py
```

真实 Chroma 模式（真实 PersistentClient + sentence-transformers all-MiniLM-L6-v2）：

```bash
python examples/stress_test.py --backend chroma
```
