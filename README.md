# 🚀 RAGFlow — 多格式知识库问答系统

[![CI](https://github.com/HluckyRing/RAGFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/HluckyRing/RAGFlow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**本地优先（local-first）的知识库智能问答**：把 PDF / Word / Excel / PPT / CSV / Markdown 丢进自己的机器，
用自然语言提问，得到带**可核验来源**的流式回答。检索侧用 **HyDE + 双路融合 + 问句拆解 + 每问分路检索 +
范围词文件过滤**，默认值全部由实测决定（见下方「📊 实测数据」一节），不是拍脑袋的。

> **Local-first RAG (retrieval-augmented generation) question answering over your own documents.**
> Multi-format ingestion (PDF / Word / Excel / PPT / CSV / Markdown), verifiable citations on every
> answer, streaming responses over SSE. Retrieval combines HyDE, dual-probe fusion, question
> decomposition, per-sub-question retrieval and chapter-scoped file filtering — compound questions
> keep **81.1%** of the sub-questions in context vs **45.9%** for single-query retrieval at the same
> context budget. FastAPI + ChromaDB backend, dependency-free vanilla HTML/CSS/JS frontend, fully
> self-hostable and offline-friendly. Optimized for Chinese documents (bge-small-zh embeddings);
> the LLM endpoint is OpenAI-compatible.

**为什么不是又一个 RAG demo**：同名项目很多（比如 `infiniflow/ragflow`）。这个仓库的差异在于
**每一个默认值都有可复现的实测依据** —— [`docs/retrieval-baseline-2026-10-06.md`](docs/retrieval-baseline-2026-10-06.md)
里是 <!-- AUTO:docs-lines -->2691<!-- /AUTO:docs-lines --> 行逐条基线，`tests/` 里是 <!-- AUTO:test-count -->768<!-- /AUTO:test-count --> 条测试，`docs/session-log/` 里连「哪些想法被证伪了」都留着
（例如「调大 `TOP_K`」实测 +0.0pp，被直接否掉）。想直接看数字，往下翻到「📊 实测数据」一节。

## ✨ 核心亮点

- 🧩 **复合问句拆解 + 每问分路检索**（默认开）：一句话里问了几件事时，自动拆成子问题**各自检索**再按名额合并 context —— 同一份 5 块预算下，命中的子问题从 **45.9% → 81.1%**；带省略指代的追问（"那这个规定还有效吗？"）也会先补全主语
- 🎯 **问到哪一章就只查那一章**（`SCOPE_FILTER`，默认开）：问句点名章号（"结合第2章课件，…"）且唯一匹配到某份文件时，检索自动加 `where=file_name` 并把"结合第2章课件"这类**范围短语从探针里删掉** —— 同一条提问从"只捞到 1 块、老实拒答"变成命中第 2 章 p91 并答对（事实核 **11/14 → 13/14**，负样本误召回不变，代价 +11 ms）；没有章号的提问一个字都不改，范围过滤把命中砍光时还会自动退回全库
- 📚 **整章枚举 + 覆盖说明**（默认开）：问"帮我整理第 N 章的内容"时把那一份文件**整份枚举**再按节均匀取样，事实核覆盖 PPT **29/30（97%）**、机器学习 PDF **14/15（93%）**；摘不全时往 context 插一句【覆盖说明】并通过 SSE 的 `coverage` 事件让前端显示"本次只摘录了 43/110 段"——**模型和用户看到的是同一本账**
- 🙅 **会明确拒答，而不是硬编**：材料答不了时推结构化 `refused` 事件（前端显示「⚠️ 知识库里没有能回答这个问题的内容（已检索 N 个文件）」），**拒答不挂来源**；只答得了一部分时正常作答并单独点明缺口
- 📄 **多格式支持**：上传 PDF、Word、Excel、PPT、CSV、TXT、Markdown 等文件，自动提取文本
- 🔍 **HyDE 检索**：先让 AI 生成"假设性答案"再去向量库匹配，显著提升复杂问题命中率
- 💬 **多轮对话记忆**：短问题里的指代词（"它"、"这个"）自动接上上一轮的问题，避免检索时语义丢失
- 💾 **状态持久化**：对话和文件数据自动保存，重启不丢失
- 🔒 **会话隔离**：每个浏览器会话独立存储，互不可见、互不覆盖
- ✅ **测试覆盖**：`pytest tests -v` 覆盖状态隔离、并发保存、旧数据迁移、指代消解、关键词降级、多编码导入、路径锚定、多文件上传/单文件删除、历史截断、拆问与分路检索、拒答契约，以及前端注入/主题/交互守卫；CI 在 push/PR 上自动跑
- ⚡ **工程降级**：向量检索不可用时自动切换关键词匹配（中文 2/3-gram 打分），且降级时不再浪费一次 HyDE 调用；缺 `API_KEY` 时服务照常启动，只降级并给中文提示
- 🧱 **入口防护**：单次上传默认上限 20 MB（按整批总字节数判断，超限在入口直接拒绝、不进解析），大模型调用默认 60 秒超时；服务默认只监听 `127.0.0.1`，不对外暴露
- 🔐 **对外监听默认拒绝启动**：绑非回环地址却没配 `AUTH_TOKEN` 时服务**直接退出**（退出码 2），不会悄悄暴露一个没有护栏的实例；`/api/chat` 按客户端 IP 限流（默认 20 次/60 s，超限 429 + `Retry-After`）。仍**不是身份体系**：`session_id` 由客户端生成
- 🗂 **多文件知识库**：一次可选或拖拽多个文件（一次请求提交，只重建一次索引），文件列表里可单独移除某个文件并自动重建索引
- 🎛 **交互与主题**：深色模式（跟随系统，手动切换后记住）、移动端抽屉侧边栏、消息级操作（重新生成 / 编辑重发 / 停止生成 / 复制）、对话搜索（Ctrl/Cmd+K）、导出 Markdown、表格列表代码块渲染与消息时间戳
- 👁 **原文件预览**：上传原件会持久化，右侧抽屉默认看原文件、可切换抽取文本；PDF 用浏览器原生渲染，Markdown 用内置渲染器渲染出标题/加粗/引用/表格，DOCX/XLSX/PPTX 用本地内置的渲染库（docx-preview / SheetJS / @aiden0z-pptx-renderer），渲染失败回退下载原件

## 📊 实测数据（不是拍脑袋的默认值）

每个默认值都是量出来的，逐条表格在 [`docs/retrieval-baseline-2026-10-06.md`](docs/retrieval-baseline-2026-10-06.md)（<!-- AUTO:docs-lines -->2691<!-- /AUTO:docs-lines --> 行）：

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| 一句问了几件事，该怎么检索 | **每问分路检索**：同一份 5 块 context 下覆盖 **45.9% → 81.1%**；分路只用 2 块就打赢整句用 5 块 | 真实语料 35 条样本 / 37 个可用子问，2/3/5/10 块预算扫描 |
| 是不是把 `TOP_K` 调大就行 | **不是**：覆盖@5 = 覆盖@10（**+0.0pp**） | 未覆盖子问的目标片名次是双峰的（第 1 附近或 20 名开外，6–10 名一条都没有） |
| 拆问（query 分解）值不值那一次调用 | **值**：端到端只差 0–2.7pp，首字延迟反而降 **0.55 s**（省掉的 HyDE 比拆问贵） | 判据先写死、ON/OFF 两个服务交替发 52 次请求配对比较 |
| 多轮追问的主语怎么补 | 拆出来的**每条子问再过一遍指代消解**：省略式那一半 12.5% → **75.0%** | 四臂对照（含「不消解」控制臂） |
| 问句里带「第2章课件」这类范围词会不会更准 | **不会，反而更差**：那 5 个字把答案页从第 8 名挤到第 16 名（距离 0.40 ≪ 阈值 0.55，**不是阈值问题**）→ 改成「章号当文件过滤器」后事实核 **11/14 → 13/14**、整条答到 **10/12 → 11/12**，逐条无回归 | 12 条带范围词提问 / 257 块语料 / 真 HyDE |
| 「整理第 N 章」怎么保证不漏章尾 | **整章枚举 + 块数按字符预算反推**：事实核 PPT **29/30（97%）**、机器学习 PDF **14/15（93%）**，整章答全 0/6→5/6、0/3→2/3 | 6 条「整理第N章」gold / 30 个事实核，真 HyDE |
| HyDE 该不该开 | 开，但**误召回会涨**（远域 11/15），靠 `intersect` 闸门压回 2/15 | 23 条负样本 |
| 要不要再加一道独立的可答性判断 | **多余**：它每条固定 **+0.78 s**，而生成那次调用自己就会拒答（域内负样本 8/8、远域 2/2 正确拒答，正样本 0/36 误拒） | 59 条语料对照，零额外调用 |
| reranker 为什么默认关 | CPU **3.9 s/问**，没过「中位 ≤1.5 s」这条线 | 实测 |
| 并发扛得住吗 | embedding 是**排队不是并行**（8 并发只 ×1.3 吞吐），但它不是瓶颈；20 并发请求零错误 | 进程内基准 + 真实 uvicorn 压测 |

`pytest tests -q` → **<!-- AUTO:test-count -->768<!-- /AUTO:test-count --> passed**（整条约 20 s）。

## 📋 支持格式

<!-- AUTO:formats -->
| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` | openpyxl |
| PPT | `.pptx` `.pptm` | python-pptx |
| CSV | `.csv` | 标准库 |
| 文本 | `.txt` `.md` | 无 |
<!-- /AUTO:formats -->

## 🛠 技术栈

- **语言**：Python <!-- AUTO:python-min -->3.11<!-- /AUTO:python-min -->+
- **大模型**：DeepSeek API（兼容 OpenAI 格式）
- **向量引擎**：ChromaDB + BAAI/bge-small-zh-v1.5（中文 Embedding）
- **后端**：FastAPI + SSE 流式输出
- **前端**：原生 HTML/CSS/JS（无框架、无 CDN；Office 预览用本地 vendor 的 docx-preview / SheetJS / @aiden0z-pptx-renderer，仍可离线；CSS 变量主题，含深色模式与移动端响应式）

## 🚀 快速开始

### 配置

在项目根目录创建 `.env` 文件（**此文件不会被上传到 GitHub**）：

```
API_KEY="sk-你的密钥"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-flash"

# 下面只是最常调的几项；**完整清单与每个默认值的理由见 .env.example**（那里是权威版本）
# CHUNK_SIZE=480            # 文本切片长度（字），受 bge 512 token 窗口约束
# MAX_DISTANCE=0.55         # 向量相关性阈值（l2 距离，越小越严格）
# DECOMPOSE_ENABLED=1       # 复合问句拆解 + 每问分路检索（=0 回到整句一次检索，不花 API）
# SCOPE_FILTER=1            # 问句点名章号时把检索限定到那份文件（"结合第2章课件，…"）
# COVERAGE_INTRA_FILE=1     # "整理第N章"：整章枚举 + 覆盖说明
# FUSION_ENABLED=1          # 原问题 + HyDE 双路融合（FUSION_GATE=intersect）
# FUSION_CANDIDATES=0       # >0 时放大每路候选池再取交集（召回优先，context 会变大）
# RERANK_ENABLED=0          # 跨编码器精排（CPU 3.9 s/问，默认关）
# ANSWERABILITY_CHECK=0     # 独立可答性判断（默认由生成侧自己拒答）
# MAX_UPLOAD_MB=20          # 单次上传大小上限（MB）
# LLM_TIMEOUT=60            # 大模型调用超时（秒）
# HOST="127.0.0.1"          # 监听地址；改成 0.0.0.0 时必须配 AUTH_TOKEN，否则拒绝启动
# AUTH_TOKEN=""             # 对外监听时必填；/api/chat 另有 CHAT_RATE_LIMIT=20 次/60 s
# PORT=8080
```

### 1. 安装依赖

```bash
git clone https://github.com/HluckyRing/RAGFlow
cd RAGFlow
pip install -r requirements.txt
```

### 2. 启动

```bash
python server.py
```

打开浏览器访问 `http://localhost:8080`。

> **别用 `python src/server.py`**：以脚本方式运行时 `sys.path[0]` 是 `src/` 而不是项目根，
> `import src.xxx` 会报 `ModuleNotFoundError`。要在 `src` 里跑入口请改用 `python -m src.server`。

默认只监听回环地址 `127.0.0.1`，同网段的其它机器访问不到。确实需要局域网 / 容器访问时，在 `.env`
里加 `HOST="0.0.0.0"`，**并且必须配好 `AUTH_TOKEN`** —— 绑非回环地址却没有 token 时服务会
**拒绝启动**（退出码 2），不会悄悄把一个没有护栏的实例暴露到网络上。`/api/chat` 还按客户端 IP
限流（默认 20 次/60 s，超限 429 + `Retry-After`）。

需要说清楚的边界：这仍**不是身份体系**。`session_id` 由客户端生成，换一个 id 就是另一个"用户"，
真正的账号、会话归属校验、CSRF 都还没有（见 `docs/session-log/INDEX.md` 的待办）。

## 📁 项目结构

<!-- AUTO:structure -->
```text
RAGFlow/
├── src/
│   ├── answerability.py    # 可答性判断（严格模式；默认由生成侧自己拒答）
│   ├── config.py           # 环境配置、日志、OpenAI 客户端
│   ├── coverage.py         # 覆盖取样（跨文件 / 文件内跨节）+ 覆盖账与【覆盖说明】
│   ├── decompose.py        # 问句拆解（廉价闸门 + JSON 解析 + 任一故障退回原问句）
│   ├── hits_merge.py       # 多路检索结果的合并口径（线上与评测共用同一份实现）
│   ├── llm.py              # 指代消解 + 多轮对话 + 每问分路编排 + 流式答案生成
│   ├── loaders.py          # 多格式文件加载器
│   ├── pdf_ingestion.py    # 文本切片
│   ├── prompts.py          # Prompt 模板（HyDE / 拆问 / 可答性 / QA）
│   ├── rerank.py           # 跨编码器精排（默认关：CPU 3.9 s/问）
│   ├── retrieval.py        # 向量检索 + 关键词回退 + HyDE + 双路融合 + 候选池
│   ├── scope.py            # 范围词 → 文件过滤器（"结合第2章课件，…" → where=file_name）
│   ├── security.py         # 对外监听 fail-closed（无 AUTH_TOKEN 拒绝启动）+ /api/chat 限流
│   ├── server.py           # FastAPI 应用定义（用 python server.py 启动）
│   ├── state.py            # 会话状态持久化（按 session_id 隔离）
│   ├── text_utils.py       # 疑问词剥离 + 2/3-gram 抽词（检索与指代消解共用）
│   ├── gates/              # 闸门措辞表（GATE_PACK 可插拔，失效可见）
│   │   ├── __init__.py    # 闸门措辞包的选择入口（GATE_PACK 决定用哪一套）
│   │   └── zh.py          # 闸门措辞表：中文包（默认）
│   └── templates/          # 前端界面（vendor/ 为本地内置的 Office 预览库）
├── scripts/
│   ├── analyze_real_questions.py        # 盘真实提问（只读 state/）：闸门触发率、复合问占比
│   ├── cleanup_orphan_collections.py    # 孤儿向量集合清理（默认干跑）
│   ├── eval_compound_retrieval.py       # 复合问句检索覆盖 + 分路对照（含模型拆问臂）
│   ├── eval_decomposition.py            # 拆问本身的结构测量（闸门/延迟/条数/误拆）
│   ├── eval_real_corpus.py              # 真实语料基线（只读、只记数字：切窗 + 距离带）
│   ├── eval_retrieval.py                # 合成语料检索评测（recall/MRR/归属，默认不花额度）
│   ├── eval_scope_filter.py             # 范围词 / 候选池四臂对照（事实核离线校验）
│   ├── measure_inline_abstention.py     # 拒答与部分可答的端到端测量
│   ├── reindex_all.py                   # 按当前切片重建所有对话索引（默认干跑）
│   ├── rejudge_abstention.py            # 用存下来的回答离线重算判定口径
│   └── sync_readme.py                   # README 的机械部分生成 / 校验（本文件）
├── tests/                # pytest 测试（pytest -q 全绿才算完）
├── docs/
│   ├── retrieval-baseline-2026-10-06.md # 逐条实测基线，所有默认值的依据
│   ├── session-log/      # 每一轮开发的复盘（数字、决策、踩过的坑）
│   └── code-review-2026-10-01.md       # 2026-10-01 代码审查原始清单
├── .github/workflows/    # CI（pytest + README 同步校验）与 Release
├── state/                # 会话状态，每个会话一个 JSON 文件（gitignore）
├── uploads/              # 上传原件，按会话/对话分目录存放（gitignore）
├── legacy/               # 历史版本归档
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```
<!-- /AUTO:structure -->

## 🧹 维护

### 文档同步（README 的数字不是手写的）

README 里「能从仓库算出来」的部分由脚本生成：测试条数与基线行数、Python 版本、项目结构树、
支持格式表。改了代码/加了测试之后跑：

```bash
python scripts/sync_readme.py --write    # 重写 README 里的自动块，然后看 git diff 再提交
python scripts/sync_readme.py --check    # 只比对，不一致退出码 1（CI 已经这么用）
```

脚本只改 `<!-- AUTO:xxx -->…<!-- /AUTO:xxx -->` 之间的内容，**块外的散文一个字都不碰**；
新增 `src/` 或 `scripts/` 下的文件却没写说明时，脚本会直接报错 —— 逼着人补一句，而不是让 README
悄悄漏掉它。

### 孤儿向量集合

删对话会级联删掉它的向量集合，但如果状态文件丢过、换过 `session_id`、或手工删过 `state/`，
`chroma_db` 里就会留下没人引用的「孤儿集合」，只增不减。对账并清理：

```bash
python scripts/cleanup_orphan_collections.py            # 干跑：只打印计划，不删任何东西
python scripts/cleanup_orphan_collections.py --apply    # 确认计划后再执行
```

脚本只删名字以 `kb_conv_` 开头、且不被任何状态文件引用的集合；如果它一个被引用的集合都没
扫到（通常是路径指错了），会直接中止而不是把库删空。

上传原件在删除单个文件/整个对话时会一并清理，不会只增不减；若手工删过 `state/`，`uploads/`
也可能留下无人引用的孤儿目录，可按目录名与状态文件对照后清理。
