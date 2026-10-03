# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

RAGFlow — 基于 RAG 架构的本地知识库问答系统，支持 PDF/Word/Excel/CSV/TXT/Markdown 多格式文件上传，通过 HyDE（假设性文档检索）提升召回率，FastAPI + SSE 流式输出，前端为零依赖原生 HTML/CSS/JS。

## 启动与开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（从项目根执行）
python server.py                # 根目录快捷入口（推荐）
python -m src.server            # 等价写法
# 注意：python src/server.py 不可用 —— 以脚本方式运行时 sys.path[0] 是 src/，
# import src.xxx 会因找不到项目根而报 ModuleNotFoundError
# 服务运行在 http://localhost:8080
```

测试：`pip install -r requirements-dev.txt` 后执行 `pytest tests -v`（覆盖状态隔离、并发保存、旧数据迁移、路径穿越防护）。无 lint/format 配置。
仍建议手动回归：启动服务 → 浏览器打开 `http://localhost:8080` → 上传文件 → 提问。

## 架构

**数据流**: 上传文件 → `loaders.py` 提取文本 → `pdf_ingestion.py` 按段落切片 → ChromaDB 向量索引 → 用户提问 → `retrieval.py` HyDE 检索（LLM 生成假设答案再查向量库）→ `llm.py` 流式生成答案（SSE）

**其他目录**:

- `legacy/` — 历史版本归档（早期控制台版 RAG 实现）
- `data/` — 示例文档（中文考试题库 PDF）
- `chroma_db/` — ChromaDB 持久化存储（gitignore）

**模块职责**:

| 文件 | 职责 |
| --- | --- |
| `src/config.py` | 环境变量加载、OpenAI 客户端初始化、日志配置、全局常量（CHUNK_SIZE, TOP_K 等） |
| `src/loaders.py` | 多格式文件加载器，通过 `LOADERS` 字典按扩展名分发，`load_file()` 为统一入口；`_decode_bytes()` 做多编码回退（utf-8-sig/utf-8/gbk/gb2312/big5/latin-1） |
| `src/pdf_ingestion.py` | `split_text()` — 按段落切分文本，支持 chunk_size/overlap 可配 |
| `src/text_utils.py` | 中文文本轻量处理：`strip_question_words()` 剥离疑问词，`content_phrases()` 抽实义片段（指代消解用），`extract_terms()` 生成 2/3-gram（关键词检索用） |
| `src/retrieval.py` | ChromaDB 向量存储管理 + `hyde_retrieve()` 核心检索（向量检索 → 失败降级关键词匹配） |
| `src/llm.py` | `resolve_query()` 指代消解（短问题 + 代词时拼接上文实体）+ `stream_answer()` SSE 流式生成 |
| `src/prompts.py` | HyDE 和 QA 两套 system prompt 模板 |
| `src/server.py` | FastAPI 应用：会话/对话 CRUD、文件上传、SSE 聊天接口 |
| `src/state.py` | 会话状态持久化：每个 session_id 一个 JSON 文件（`state/`），进程内缓存 + 每会话锁 + 原子写 + 旧数据迁移 |

**关键设计决策**:
- **HyDE 检索**: 先让 LLM 生成"假设性答案"，用这个答案文本去做向量匹配，比直接用问题检索召回率更高
- **工程降级链**: 向量检索失败 → 自动切换关键词匹配（中文 2/3-gram + 英文词打分，`sqrt(长度)` 归一化）。一条都没命中时返回空，由上层提示「未找到相关内容」，不拿无关段落喂模型。降级时**不再**调用 HyDE
- **HyDE 按需调用**: 只有 `use_vector` 为真时才生成假设性答案；关键词检索用原始问题（HyDE 生成的长段落不适合做词频匹配）
- **状态持久化**: 会话和对话数据按 session_id 隔离，存于 `state/<sha1(sid)>.json`。文件名由 sid 哈希派生，客户端传什么都无法写出 STATE_DIR（防路径穿越）。使用 atomic write（写唯一临时文件 → fsync → rename）
- **会话隔离**: 状态只在会话首次访问时从磁盘加载一次，之后以内存为准；同一会话的「改内存 + 落盘」由每会话 RLock 串行化。缓存和锁都在进程内，因此**只支持单 worker 部署**
- **旧数据迁移**: 首次遇到「客户端自带、服务端未见过的 sid」时，把旧版 `kb_state.json` 原子改名认领给它（保留为 `kb_state.json.migrated` 便于回滚）；`/api/session` 新发的 sid 不参与认领
- **多轮对话**: `resolve_query()` 在问题 < 15 字且含指代词（它/它们/这个/该/上述…，刻意不含裸「这」「那」；含「其他」「其中」等也不算）时，从上一轮用户消息提取实义片段拼到当前问题前。注意 `history` 是**不含当前提问**的快照，所以上一轮就是 `user_msgs[-1]`
- **ChromaDB**: 使用 PersistentClient 持久化到项目根下的 `chroma_db/`（`VECTOR_DB_PATH`，相对路径锚定项目根），每个对话独立 collection（命名 `kb_{name}_{hash}`）
- **不阻塞事件循环**: 所有可能阻塞的调用（文件解析、embedding 推理、LLM 网络请求、写盘）一律经 `run_in_threadpool` 执行。SSE 的 `_chat_stream()` 必须是**同步**生成器 —— Starlette 只对非 AsyncIterable 用 `iterate_in_threadpool` 迭代，写成 `async def` 反而会在事件循环上迭代、把全服务卡死
- **按需建向量库**: 建对话时不创建 Chroma collection，推迟到首次上传时在 `_commit_upload` 里补建（不论对话是 API 建的还是上传时顺带建的），避免「建了对话没传文件」留下空集合
- **Embedding 模型**: BAAI/bge-small-zh-v1.5，通过 `HF_ENDPOINT` 环境变量支持 HuggingFace 镜像
- **相关性阈值**: 向量结果按 l2 距离过滤（collection 未指定 `hnsw:space`，走 Chroma 默认的 l2；BGE 是归一化向量，距离 = 2 − 2cos）。默认 `MAX_DISTANCE=0.55` —— 实测本项目语料上相关查询 max≈0.47、无关查询 min≈0.62，取分离带中点。超阈值的块不进 context；全部超阈值则走关键词兜底。**换语料或换 embedding 模型必须重新标定**
- **流式失败不入历史**: `stream_answer` 失败时抛 `LLMStreamError`，而不是把错误文案当回答 yield 出去（否则会被写进 `messages`，下一轮又被当上下文喂回模型）。server 捕获后仍把错误推给前端，但只落盘**已生成的部分回答**，错误文案绝不入库
- **路径锚定项目根**: `config.PROJECT_ROOT` 是唯一的项目根定义，`resolve_path()` 把相对路径（含默认值）锚定到它。`VECTOR_DB_PATH` / `STATE_DIR` / `LEGACY_STATE_FILE` 都走它 —— 从任何目录启动都不会各自生出一份向量库或状态目录
- **状态文件不存派生字段**: 磁盘上只存 `files[].file_text`；`full_text` 由 `server._conv_full_text()` 在运行时拼接、加载时重新派生。旧文件里带 `full_text` 也能无损读，但新写入不再把正文存两遍
- **缺 API_KEY 只降级不崩**: `config.build_client()` 缺 key 时记一条中文 ERROR 并返回 `None`（不再让 openai SDK 抛英文异常把 import 阶段带崩）。此时向量/关键词检索照常，HyDE 跳过，问答抛 `LLMStreamError` 给前端中文提示
- **上传大小上限**: `MAX_UPLOAD_MB`（默认 20）在 `/api/upload` 入口按 `_upload_size()` 判断，**按整批总字节数**算，超限直接 413、不进入解析 —— 解析大 PDF/DOCX 是全流程最贵的一步
- **多文件上传与单文件删除**: 上传用复数 `files` 字段一次提交整批（兼容旧的单数 `file`），`_commit_upload(sid, conv_id, [(name, text)])` 一次只重建一次索引 —— `_rebuild_collection` 是全量重写，逐文件提交会退化成 O(N²) 次重复嵌入。`DELETE /api/conversations/{conv_id}/files/{file_name}` 删除单个文件并重建索引；**删光文件时派生 `full_text` 必须为空**，否则索引里会残留已删正文
- **前端动态数据不进内联处理器**: 对话名/文件名一律经 `textContent` / `dataset` 落地，事件用 `addEventListener` 绑定。内联事件处理器是 HTML 属性、会被当代码解析，动态数据里一个单引号就能闭合字符串执行任意 JS（P1-15 的老写法）；`esc()` 也补了单引号转义作为纵深防御
- **前端主题与响应式**: 配色收敛为 CSS 变量，深色值挂在 `html[data-theme="dark"]`。`<head>` 里的内联脚本在 CSS 生效前定主题（避免首屏闪白）：没手动选过就跟随 `prefers-color-scheme`，手动切换后写入 `localStorage.ragflow_theme` 并长期优先。≤820px 时侧边栏改为固定抽屉 + 遮罩，`syncSidebar()` 在断点变化时同步开合
- **重新生成/编辑重发靠 truncate_to 覆盖旧轮次**: `/api/chat` 多了可选整数 `truncate_to`（`0..len(messages)`），语义是「先把 `messages` 截到该长度，再追加本次提问」；不传时行为与改造前完全一致，越界/非整数/布尔一律 400 中文提示。截断与追加在 `_append_message` 的**同一把会话锁内**完成，不会被并发写截到一半。重新生成不需要新接口：截到这条提问处 + 用同样的文字重发即可，服务端因此不会留下重复轮次
- **停止生成要落半截回答**: SSE 生成器 `_chat_stream()` 用 `try/finally` 落盘已生成的部分，客户端中断（生成器被 `close()` → `GeneratorExit`）时给消息标 `stopped=true`，前端显示「⏹ 已停止生成」；错误文案仍然绝不进 `messages`。抽成模块级函数是为了能直接测「客户端中途断开」这条路径
- **消息时间戳由服务端落盘**: `_append_message()` 给缺 `time` 的消息补服务器时间。时间戳只活在前端内存里的话，刷新就没了
- **默认只监听回环**: `HOST`（默认 `127.0.0.1`）/ `PORT`（默认 8080）统一定义在 `config.py`，两个可用入口（`python server.py`、`python -m src.server`）都复用它。**`python src/server.py` 不能当脚本跑**（`sys.path[0]` 会变成 `src/`，`import src.xxx` 直接 ModuleNotFoundError，只能作为模块被导入）。这个服务没有鉴权，要对外提供必须显式设 `HOST=0.0.0.0`，不要在代码里写死对外地址
- **归档脚本不关 TLS 校验**: `legacy/` 里曾用 `ssl._create_default_https_context = ssl._create_unverified_context` 全局关掉证书校验，已移除；`tests/test_security.py` 有静态守卫防止被写回来
- **孤儿 collection 清理**: `scripts/cleanup_orphan_collections.py` —— 默认干跑，要 `--apply` 才真删；只删名字以 `kb_conv_` 开头且不被任何状态文件引用的集合；如果一个被引用的集合都没扫到（通常是路径指错）就中止，只有 `--force` 能越过。状态来源含 `state/*.json` 与旧版 `kb_state.json(.migrated)`
- **审查清单与 CI**: 2026-10-01 那次代码审查的原始条目归档在 `docs/code-review-2026-10-01.md`（**编号连续**：P0 1–7 / P1 8–17 / P2 18–21；**不存在 P1-1~7 与 P1-18**），整改状态随代码更新；`.github/workflows/ci.yml` 在 push/PR 上跑 `pytest -q`

## 配置

项目根目录 `.env` 文件（不入 git，参考 `.env.example`）:

```ini
API_KEY="sk-xxx"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-flash"
# 可选
EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=10
MAX_CONTEXT_LENGTH=8000
MAX_DISTANCE=0.55
MAX_UPLOAD_MB=20
LLM_TIMEOUT=60
# 相对路径一律锚定项目根，可填绝对路径
VECTOR_DB_PATH="./chroma_db"
STATE_DIR="./state"
# 监听地址：默认只绑回环；要对外提供才改成 0.0.0.0
HOST="127.0.0.1"
PORT=8080
HF_ENDPOINT="https://hf-mirror.com"
```

LLM API 使用 OpenAI 兼容接口，`BASE_URL` + `API_KEY` 决定实际调用哪个服务。

<!-- superpowers-zh:begin (do not edit between these markers) -->
# Superpowers-ZH 中文增强版

本项目已安装 superpowers-zh 技能框架（20 个 skills），仅限本地使用，未入库。其他开发者 clone 后需单独安装。

## 核心规则

1. **收到任务时，先检查是否有匹配的 skill** — 哪怕只有 1% 的可能性也要检查
2. **设计先于编码** — 收到功能需求时，先用 brainstorming skill 做需求分析
3. **测试先于实现** — 写代码前先写测试（TDD）
4. **验证先于完成** — 声称完成前必须运行验证命令

## 可用 Skills

Skills 位于 `.claude/skills/` 目录（不入 git），每个 skill 有独立的 `SKILL.md` 文件。

- **brainstorming**: 在任何创造性工作之前必须使用此技能——创建功能、构建组件、添加功能或修改行为。在实现之前先探索用户意图、需求和设计。
- **chinese-code-review**: 中文 review 沟通参考——话术模板、分级标注（必须修复/建议修改/仅供参考）、国内团队常见反模式应对。仅在用户显式 /chinese-code-review 时调用，不要根据上下文自动触发。
- **chinese-commit-conventions**: 中文 commit 与 changelog 配置参考——Conventional Commits 中文适配、commitlint/husky/commitizen 中文模板、conventional-changelog 中文配置。仅在用户显式 /chinese-commit-conventions 时调用，不要根据上下文自动触发。
- **chinese-documentation**: 中文文档排版参考——中英文空格、全半角标点、术语保留、链接格式、中文文案排版指北约定。仅在用户显式 /chinese-documentation 时调用，不要根据上下文自动触发。
- **chinese-git-workflow**: 国内 Git 平台配置参考——Gitee、Coding.net、极狐 GitLab、CNB 的 SSH/HTTPS/凭据/CI 接入差异与镜像同步配置。仅在用户显式 /chinese-git-workflow 时调用，不要根据上下文自动触发。
- **dispatching-parallel-agents**: 当面对 2 个以上可以独立进行、无共享状态或顺序依赖的任务时使用
- **executing-plans**: 当你有一份书面实现计划需要在单独的会话中执行，并设有审查检查点时使用
- **finishing-a-development-branch**: 当实现完成、所有测试通过、需要决定如何集成这份工作时使用
- **mcp-builder**: MCP 服务器构建方法论 — 系统化构建生产级 MCP 工具，让 AI 助手连接外部能力
- **receiving-code-review**: 收到代码审查反馈后、实施建议之前使用，尤其当反馈不明确或技术上有疑问时——需要技术严谨性和验证，而非敷衍附和或盲目执行
- **requesting-code-review**: 完成任务、实现重要功能或合并前使用，用于验证工作成果是否符合要求
- **subagent-driven-development**: 当在当前会话中执行包含独立任务的实现计划时使用
- **systematic-debugging**: 遇到任何 bug、测试失败或异常行为时使用，在提出修复方案之前执行
- **test-driven-development**: 在实现任何功能或修复 bug 时使用，在编写实现代码之前
- **using-git-worktrees**: 当需要开始与当前工作区隔离的功能开发，或在执行实现计划之前使用——通过原生工具或 git worktree 回退机制确保隔离工作区存在
- **using-superpowers**: 在开始任何对话时使用——确立如何查找和使用技能，要求在任何响应（包括澄清性问题）之前调用 Skill 工具
- **verification-before-completion**: 在宣称工作完成、已修复或测试通过之前使用，在提交或创建 PR 之前——必须运行验证命令并确认输出后才能声称成功；始终用证据支撑断言
- **workflow-runner**: 在 Claude Code / OpenClaw / Cursor 中直接运行 agency-orchestrator YAML 工作流——无需 API key，使用当前会话的 LLM 作为执行引擎。当用户提供 .yaml 工作流文件或要求多角色协作完成任务时触发。
- **writing-plans**: 当你有规格说明或需求用于多步骤任务时使用，在动手写代码之前
- **writing-skills**: 当创建新技能、编辑现有技能或在部署前验证技能是否有效时使用

## 如何使用

当任务匹配某个 skill 时，使用 `Skill` 工具加载对应 skill 并严格遵循其流程。绝不要用 Read 工具读取 SKILL.md 文件。

如果你认为哪怕只有 1% 的可能性某个 skill 适用于你正在做的事情，你必须调用该 skill 检查。
<!-- superpowers-zh:end -->
