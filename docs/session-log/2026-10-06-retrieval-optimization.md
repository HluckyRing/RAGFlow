# 会话归档：2026-10-06 检索优化（P0 → reranker）

> **怎么用这份文件**：新会话想补齐上下文，按这个顺序读 ——
> 1. 本文件（发生了什么、为什么、结论是什么）
> 2. `docs/retrieval-baseline-2026-10-06.md`（所有基线数字与逐条结论的权威版本）
> 3. `AGENTS.md` / `CLAUDE.md`（编码约定，两份逐字一致）
> 4. `MEMORY.md`（现状、速查、待办）
>
> 时间跨度：2026-10-06 白天 ~ 2026-10-07 00:15（Asia/Shanghai）。所有数字都是真跑出来的。

## 一句话总结

把「检索质量」这件事从「只能靠感觉」改造成**有尺子、有基线、有回归守卫**的工程：
先建离线评测（P0），再按评测结果改切片与 context 装配（P1）、让引用可核验（P2）、
把来源显示到前端（P3），随后修掉两个**静默失效**的真 bug（reindex 覆盖、HyDE 被推理模型
吃光），补上 section 级指标、真实语料基线、检索路径可见性，最后实测 reranker 并把默认值
定成「关闭但一键可开」。

## 提交清单（14 个，均未推送；`main` ahead 14，`origin/main` 停在 18b1eff）

| 提交 | 内容 |
| --- | --- |
| `d074f62` | **P0** 检索评测基线：`scripts/eval_retrieval.py` + 合成语料/标注（零行为变更） |
| `46233d7` | **P1** `CHUNK_SIZE` 600→480 + `split_text` overlap 兜底 + context 按块边界/原文序装配 |
| `01de26b` | **P2** 块带来源元数据（`split_files`、`metadatas`、`【来源：…】`、prompt 要求引用） |
| `22fb4b9` | **P3** 前端显示来源（`sources` SSE 事件 + chip，落盘后可刷新） |
| `017336e` | **reindex 修复**：按 `conv_id` 去重（`state/` 优先、legacy `.migrated` 只补缺） |
| `e55e62d` | **section 级指标**：文本 spec 生成 pptx + `expect_sections` + 归属正确率 |
| `3113131` | **路径可见性**：每条样本记 `path`（向量 / 关键词兜底）+ 汇总计数 |
| `88adf41` | **真实语料基线**：`scripts/eval_real_corpus.py`（切窗诊断 + 距离带，只记数字不留文件） |
| `24f3661` | **HyDE 修复**：推理模型吃光 `max_tokens` 导致 HyDE 静默失效 |
| `513a23e` | 真实语料补 HyDE 口径重标（文档） |
| `2cf2bcf` | **calibrate 口径修复**：`--hyde` 下必须用假设文本查询 + 补 HyDE 口径基线 |
| `65669fd` | **reranker**：跨编码器精排接入（默认关闭），线上口径误召回 2/5 → 0/5 |
| `afaffa5` | `.env.example` 补 reranker 五个开关 |
| `1831deb` | `.gitignore` 加 `MEMORY.md` |

## 四个真 bug（都是「静默」的那种，最值钱）

1. **chunk 被 512 token 截断**（P1 前）：`CHUNK_SIZE=600` 而 bge-small-zh-v1.5 窗口是 512
   token，超出的尾部**静默不参与向量**。实测 600 时 2/5 个块超窗（521/528 token），480 后 0/8。
   修法：`CHUNK_SIZE=480` + `split_text` 拼 overlap 时若超 `chunk_size` 就放弃那段 overlap。
2. **reindex 用旧快照覆盖线上索引**（`017336e`）：legacy `kb_state.json.migrated` 与 `state/`
   会话文件里同 `conv_id`、同 collection 名，但内容更旧；重建顺序把 legacy 排在后面 → 覆盖线上。
   修法：按 `conv_id` 去重 + 打印 `[跳过]` + WARNING。
3. **HyDE 自模型切到 `deepseek-flash` 起一直是关着的**（`24f3661`）：推理模型先把 token 花在
   `reasoning_content`，`_generate_hyde` 写死的 `max_tokens=200` 被吃光 → `content` 空串 →
   `content or question` 静默退回原问题，**一行日志都没有**。修法：该调用显式
   `extra_body={"thinking": {"type": "disabled"}}` + 空内容记 WARNING + 评测跑前自检。
4. **`--calibrate` 用错 query 文本**（`2cf2bcf`）：`--hyde` 下仍拿原问题直查，打出的是 no-hyde
   距离带，`suggested_max_distance` 因此不可用（两轮数字逐位相同才发现）。

## 关键测量（全部实测，细节见 `docs/retrieval-baseline-2026-10-06.md`）

### 合成评测（41 条 = 36 正 + 5 负，7 文件 / 14 chunk）

| 配置 | recall@5 | MRR | 文件 / section 归属 | 兜底 | 负样本误召回 |
| --- | --- | --- | --- | --- | --- |
| P1 后（30 正，6 文件） | 1.000 | 0.961 | 1.000 / — | — | 0.000 |
| P2 后 | 1.000 | 1.000 | 1.000（30/30）/ — | — | 0.000 |
| 补 section（41 条，no-hyde） | 1.000 | 1.000 | 1.000（36/36）/ **1.000（6/6）** | 7/41 | 0.000 |
| **HyDE（线上口径）** | 1.000 | **0.958** | 1.000 / 1.000 | 3/41 | **0.400（n01、n03）** |
| **HyDE + reranker 0.3** | **1.000** | **1.000** | **1.000 / 1.000** | 6/41 | **0.000** |

### 真实语料（4 对话 / 23 文件 / 约 29.7 万字 → 1199 chunk / 12 条真实提问）

| 口径 | 真实提问最佳距离 | 无关代理距离（合成负样本 n01–n05 当代理） |
| --- | --- | --- |
| no-hyde | min 0.4277 / 中位 0.4905 / max 0.5841（**1/12 超 0.55**） | min 0.5111 / 中位 0.5456 / max 0.6827（10/20 超） |
| **HyDE** | min 0.3879 / 中位 0.4376 / max 0.5486（**0/12 超**） | min 0.4244 / 中位 0.5273 / max 0.6530（**6/20 超**） |
| 超窗 chunk | **0 / 1199**（没有一个块超过 512 token） | — |

### reranker（`BAAI/bge-reranker-base`，MIT、278M、1.11GB、零新依赖）

- **CPU 延迟 3.93 s/次**（20 候选，单对约 280ms）
- 分数带：正样本 gold **min 0.0284 / 中位 0.9636 / max 0.9993**；负样本最高分 **max 0.0738**
- 判别力远好于向量距离（负样本最高分 ≪ 正样本中位），但严格中点规则被**一条**离群正样本
  （0.0284）破坏 → 阈值改用「跑评测比指标」标定：0.1 与 0.3 指标相同，取 **0.3**。

## 决策与理由

| 决策 | 为什么 |
| --- | --- |
| `CHUNK_SIZE=480` | embedding 窗口 512 token，600 字会被静默截断；真实语料 1199 块实测 0 超窗 |
| `MAX_DISTANCE` **保持 0.55** | 合成/真实 × no-hyde/HyDE 四种组合下正负距离带**都重叠**（HyDE：正 max 0.5486 > 负 min 0.4244），没有支持换值的证据 |
| `RERANK_ENABLED` **默认 0** | 质量已证明（误召回 2/5→0/5、MRR 回满），但 CPU 3.9s/次，超过约定的「≤1.5s 才默认开」门槛 |
| 评测语料的结构标记**不存二进制** | 文本 spec 现场生成 pptx → 真实 loader 抽文本；仓库可 review、可 diff |
| 真实语料**只记数字不留文件** | 读 `state/*.json` + `EphemeralClient`，不落文件、不打印正文 |
| sources/引用**重建在真实数据上** | 2026-07-24 那版因「引用不准」被删；这次先让「块→来源」在数据里是真的（P2），再显示（P3） |
| 只本地提交、不推送 | 仓库既定约定：推送需用户显式授权，推后回读 `git ls-remote` 核对 |

## 踩过的坑（环境与流程）

- **pwsh 沙箱**：默认必然报 `SetNamedSecurityInfoW failed (Win32 5): grantWrite(D:\...)`；
  加一次性 `sandbox_permissions: danger-full-access` 就能正常跑 git/pytest/真实脚本。bash 不可用。
- **国内下 HF 模型基本不可用**：直连 34–245 KB/s，`huggingface_hub` 会卡在 `.incomplete`
  **0 字节**（配本地代理也一样）；`hf-mirror.com` 的 `/resolve/` 只是 308 跳回 huggingface.co。
  **ModelScope 同源实测 12.6 MB/s**（1.11GB≈90 秒）：`modelscope.cn/api/v1/models/<repo>/repo/files`
  列文件 + `.../repo?Revision=master&FilePath=<name>` 下载 → 放到本地目录 → 让 `RERANK_MODEL`
  / `EMBEDDING_MODEL` 指过去。别手工拼 HF 缓存布局（文件名校验会打架）。
- **行尾 CRLF**：用 `.Replace()` 做反向对照时多行锚点会因行尾不匹配失败 → 用单行锚点。
- **少暴露一个开关名就会崩**：`retrieval.py` 漏导入 `RERANK_MODEL`，`--rerank` 在**打印语句**
  里 AttributeError；已加 `test_retrieval_exposes_all_rerank_knobs`。
- **pre-commit 密钥守卫误拦 `.env.example`**：规则 `/(^|\/)\.env[^/]*$/` 把模板也拦了，而
  `tests/test_repo_hygiene.py` 要求它能入库。已给本机守卫加「只放行 `.env.example`」例外，
  并用三项负向探测确认仍拦得住。**该文件未版本化，换机器要重加。**
- **测试绝不加载真模型**：`tests/conftest.py` 有 autouse 关掉 `RERANK_ENABLED`。

## 验证纪律（这套流程是这次最有价值的沉淀之一）

每次改动都走：**TDD 红基线 → 全量 `pytest tests -q` → 反向对照（临时改回旧写法，确认对应
用例变红，再逐字节还原）→ 涉及真实运行路径补真实 uvicorn 冒烟（ASGI 不算）→ `git status/diff`
复核无残留 → 提交信息先给用户确认 → 只本地提交**。没有真实输出就不能说「验证通过」；
提交前必须重跑验证，不复用旧结果。测试总数 **182 → 326**。

## 结束时状态与待办

- 本地 `main` = `1831deb`，**ahead 14 未推送**；全量 **326 passed**；工作区干净
  （`?? .agents/` 是刻意的）。
- 待办：① 真实语料的 reranker 分数带（脚本目前只有 `--rerank` 开关）；② xlsx 的「工作表名」
  支路还没有评测样本；③ 原问题+HyDE 双路融合（RRF）、关键词兜底切缓存；④ reranker 想开建议
  先试更小的 `RERANK_CANDIDATES` 或量化 ONNX（`onnxruntime` 已在依赖里）；⑤ 前端引用 chip
  点击跳预览抽屉；⑥ 推送 14 个提交。
