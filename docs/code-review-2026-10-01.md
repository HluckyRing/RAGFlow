# 代码审查与整改清单（2026-10-01）

> 这是 2026-10-01 那次代码审查的**原始条目**。审查产出原本只存在于一次会话的转录里，2026-10-02 从转录完整还原后归档到仓库，以免再次丢失。
>
> ⚠️ **编号是连续的**：P0 段为 1–7，P1 段为 8–17，P2 段为 18–21。**不存在 P1-1 ~ P1-7，也不存在 P1-18** —— 早期多份进度记录误把这三个区段当成「缺失的 P1 编号」，反复写下了一个并不存在的缺口。

## 来源

- 审查产出：会话 `session-6823cec8-6d3e-4bfd-9901-bb0721fd40e4`（workspace 为本仓库，2026-10-02 15:21–16:09 活动，即完成批次 A/B 的那个会话）
- 转录位置：`~/.dsh/sessions/--D-Python-ai_rag_project--/<session-id>/session.v4.jsonl.zstd`
  （**多帧 zstd**，必须用流式解压；直接 `decompress()` 会报 `could not determine content size in frame header`）
- 本清单的 P1/P2 条目为转录中的**逐字原文**；P0 条目的措辞由转录中的整改讨论还原，语义对应。

## P0 段（1–7）：必修，已全部完成

| 编号 | 问题 | 修复落点 |
|---|---|---|
| P0-1 | 每个请求都全量重载状态，并用磁盘内容覆盖内存（3 次请求触发 6 次 `init_vector_store`） | `e0636fc` |
| P0-2 | 并发保存共用同一个 `.tmp` 互相覆盖，导致丢数据（200 次并发保存失败 135 次且只记 warning） | `e0636fc` |
| P0-3 | `session_id` 没有隔离作用：任何 sid 都能读写全部对话 | `e0636fc` |
| P0-4 | 指代消解取错轮次（off-by-one），且被「这 / 那 / 其他」这类词误触发 | `3c35409` |
| P0-5 | 关键词降级检索失效：打分退化成「问题原句必须逐字出现」，永远返回前 2 块 | `3c35409` |
| P0-6 | CSV 编码：GBK 文件直接抛 `UnicodeDecodeError` | `3c35409` |
| P0-7 | 删除对话不删 collection，`chroma_db` 只增不减 | `e0636fc` |

## P1 段（8–17）：已全部完成

| 编号 | 原始措辞 | 修复落点 |
|---|---|---|
| P1-8 | 异步接口里的同步阻塞调用 | `915dd40` |
| P1-9 | state 文件冗余/膨胀（`full_text` 双份存储 + 每次全量重写 + 无上传大小限制） | `81db123`（`full_text` 不再落盘、加 `MAX_UPLOAD_MB`） |
| P1-10 | HyDE 在降级路径下仍调用 LLM | `3c35409`（改为只在 `use_vector` 为真时调用） |
| P1-11 | 向量结果无相关性阈值 | `96fabeb`（`MAX_DISTANCE=0.55`） |
| P1-12 | `stream_answer` 健壮性（`choices[0]`、错误文本入历史） | `96fabeb`（`LLMStreamError`、逐级判空） |
| P1-13 | `create_conversation` 提前建空 collection | `915dd40`（改为首次上传时按需补建） |
| P1-14 | 相对路径 `STATE_FILE` / `VECTOR_DB_PATH` | `81db123`（`PROJECT_ROOT` + `resolve_path()`） |
| P1-15 | 前端 XSS（`esc` 不转义单引号 + 内联 `onclick`） | `2216a97`（`textContent`/`dataset` + `addEventListener`） |
| P1-16 | 配置缺失时启动报错不友好 / 无超时 | `81db123`（缺 `API_KEY` 中文提示并降级、`LLM_TIMEOUT`） |
| P1-17 | 单文件上传、无法单独删除已上传文件 | `2216a97`（批量 `files` + `DELETE .../files/{name}`） |

## P2 段（18–21）：参考项

| 编号 | 原始措辞 | 状态 |
|---|---|---|
| P2-18 | 无测试/CI | 测试已补（103 个用例）；CI 见 `.github/workflows/ci.yml` |
| P2-19 | legacy 关闭 TLS 校验 | 已修（`02b8873`，移除 `ssl._create_unverified_context` 全局替换） |
| P2-20 | 死代码/笔误清单 | 四处已全部处理，见下 |
| P2-21 | `0.0.0.0` 无鉴权 | 已修（`02b8873`，默认只绑 `127.0.0.1`） |

P2-20 原始点名的四处：

| 位置 | 处理 |
|---|---|
| `SUPPORTED_TYPES`（`src/loaders.py`，全仓无引用） | 已删除 |
| `deleteConv`（`src/templates/index.html`，从未被调用） | 已删除（`2216a97`） |
| `GET /api/conversations/{conv_id}/messages`（`src/server.py`，前端从不调用） | 已删除 |
| `.gitignore` 的 `!?.env.example`（笔误） | 已修正为 `!.env.example` |

## 审查中被证伪的两个怀疑（不用改）

- **Chroma EmbeddingFunction 冲突**：A/B/C 三条路径都正常（chromadb 1.5.9 会从持久化配置重建 EF）
- **`.env` 带 BOM**：实际无 BOM，`API_KEY` 读取正常

## 教训

编号对不上时，**先怀疑编号体系本身**（分段连续编号：P0 1–7 / P1 8–17 / P2 18–21），而不是假设存在丢失项。本次就是因为把 P0 段与 P2 段的编号当成了「缺失的 P1 项」，在多个文档里写下了不存在的缺口。
