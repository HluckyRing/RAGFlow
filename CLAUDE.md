# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

RAGFlow — 基于 RAG 架构的本地知识库问答系统，支持 PDF/Word/Excel/CSV/TXT/Markdown 多格式文件上传，通过 HyDE（假设性文档检索）提升召回率，FastAPI + SSE 流式输出，前端为零依赖原生 HTML/CSS/JS。

## 启动与开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（两种方式等效）
python server.py                # 根目录快捷入口
python src/server.py            # 直接启动
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
| `src/loaders.py` | 多格式文件加载器，通过 `LOADERS` 字典按扩展名分发，`load_file()` 为统一入口 |
| `src/pdf_ingestion.py` | `split_text()` — 按段落切分文本，支持 chunk_size/overlap 可配 |
| `src/retrieval.py` | ChromaDB 向量存储管理 + `hyde_retrieve()` 核心检索（向量检索 → 失败降级关键词匹配） |
| `src/llm.py` | `resolve_query()` 指代消解（短问题 + 代词时拼接上文实体）+ `stream_answer()` SSE 流式生成 |
| `src/prompts.py` | HyDE 和 QA 两套 system prompt 模板 |
| `src/server.py` | FastAPI 应用：会话/对话 CRUD、文件上传、SSE 聊天接口 |
| `src/state.py` | 会话状态持久化：每个 session_id 一个 JSON 文件（`state/`），进程内缓存 + 每会话锁 + 原子写 + 旧数据迁移 |

**关键设计决策**:
- **HyDE 检索**: 先让 LLM 生成"假设性答案"，用这个答案文本去做向量匹配，比直接用问题检索召回率更高
- **工程降级链**: 向量检索失败 → 自动切换关键词匹配（正则提取 2+ 字中英文词做词频打分），保证服务不中断
- **状态持久化**: 会话和对话数据按 session_id 隔离，存于 `state/<sha1(sid)>.json`。文件名由 sid 哈希派生，客户端传什么都无法写出 STATE_DIR（防路径穿越）。使用 atomic write（写唯一临时文件 → fsync → rename）
- **会话隔离**: 状态只在会话首次访问时从磁盘加载一次，之后以内存为准；同一会话的「改内存 + 落盘」由每会话 RLock 串行化。缓存和锁都在进程内，因此**只支持单 worker 部署**
- **旧数据迁移**: 首次遇到「客户端自带、服务端未见过的 sid」时，把旧版 `kb_state.json` 原子改名认领给它（保留为 `kb_state.json.migrated` 便于回滚）；`/api/session` 新发的 sid 不参与认领
- **多轮对话**: `resolve_query()` 检测代词（它/这/那/其/她/他）且问题 < 15 字符时，从上一轮用户消息提取实体拼接到当前问题
- **ChromaDB**: 使用 PersistentClient 持久化到 `./chroma_db/`，每个对话独立 collection（命名 `kb_{name}_{hash}`）
- **Embedding 模型**: BAAI/bge-small-zh-v1.5，通过 `HF_ENDPOINT` 环境变量支持 HuggingFace 镜像

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
VECTOR_DB_PATH="./chroma_db"
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
