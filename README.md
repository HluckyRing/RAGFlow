# 🚀 RAGFlow — 多格式知识库问答系统

[![CI](https://github.com/HluckyRing/RAGFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/HluckyRing/RAGFlow/actions/workflows/ci.yml)

基于 **RAG（检索增强生成）** 架构的本地知识库智能问答工具。支持 **PDF / TXT / Markdown / Word / Excel / CSV** 多种文件格式，通过 **HyDE（假设性文档检索）** 提升召回率。

## ✨ 核心亮点

- 📄 **多格式支持**：上传 PDF、Word、Excel、PPT、CSV、TXT、Markdown 等文件，自动提取文本
- 🔍 **HyDE 检索**：先让 AI 生成"假设性答案"再去向量库匹配，显著提升复杂问题命中率
- 💬 **多轮对话记忆**：短问题里的指代词（"它"、"这个"）自动接上上一轮的问题，避免检索时语义丢失
- 💾 **状态持久化**：对话和文件数据自动保存，重启不丢失
- 🔒 **会话隔离**：每个浏览器会话独立存储，互不可见、互不覆盖
- ✅ **测试覆盖**：`pytest tests -v` 覆盖状态隔离、并发保存、旧数据迁移、指代消解、关键词降级、多编码导入、路径锚定、多文件上传/单文件删除、历史截断，以及前端注入/主题/交互守卫；CI 在 push/PR 上自动跑
- ⚡ **工程降级**：向量检索不可用时自动切换关键词匹配（中文 2/3-gram 打分），且降级时不再浪费一次 HyDE 调用；缺 `API_KEY` 时服务照常启动，只降级并给中文提示
- 🧱 **入口防护**：单次上传默认上限 20 MB（按整批总字节数判断，超限在入口直接拒绝、不进解析），大模型调用默认 60 秒超时；服务默认只监听 `127.0.0.1`，不对外暴露
- 🗂 **多文件知识库**：一次可选或拖拽多个文件（一次请求提交，只重建一次索引），文件列表里可单独移除某个文件并自动重建索引
- 🎛 **交互与主题**：深色模式（跟随系统，手动切换后记住）、移动端抽屉侧边栏、消息级操作（重新生成 / 编辑重发 / 停止生成 / 复制）、对话搜索（Ctrl/Cmd+K）、导出 Markdown、表格列表代码块渲染与消息时间戳

## 📋 支持格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` | openpyxl |
| PPT | `.pptx` `.pptm` | python-pptx |
| CSV | `.csv` | 标准库 |
| 文本 | `.txt` `.md` | 无 |

## 🛠 技术栈

- **语言**：Python 3.11+
- **大模型**：DeepSeek API（兼容 OpenAI 格式）
- **向量引擎**：ChromaDB + BAAI/bge-small-zh-v1.5（中文 Embedding）
- **后端**：FastAPI + SSE 流式输出
- **前端**：原生 HTML/CSS/JS（零依赖、无 CDN；CSS 变量主题，含深色模式与移动端响应式）

## 🚀 快速开始

### 配置

在项目根目录创建 `.env` 文件（**此文件不会被上传到 GitHub**）：

```
API_KEY="sk-你的密钥"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-flash"
# 可选：向量相关性阈值（l2 距离，越小越严格），默认 0.55
# MAX_DISTANCE=0.55
# 可选：单次上传大小上限（MB），默认 20
# MAX_UPLOAD_MB=20
# 可选：大模型调用超时（秒），默认 60
# LLM_TIMEOUT=60
# 可选：监听地址，默认只绑回环 127.0.0.1；要对外提供才改成 0.0.0.0
# HOST="127.0.0.1"
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

默认只监听回环地址 `127.0.0.1`，同网段的其它机器访问不到 —— 这个服务没有鉴权，所以默认不暴露。
确实需要局域网访问时，在 `.env` 里加 `HOST="0.0.0.0"` 后重启。

## 📁 项目结构

```
RAGFlow/
├── src/
│   ├── server.py         # FastAPI 应用定义（用 python server.py 启动）
│   ├── state.py          # 会话状态持久化（按 session_id 隔离）
│   ├── config.py         # 环境配置、日志、OpenAI 客户端
│   ├── loaders.py        # 多格式文件加载器
│   ├── pdf_ingestion.py  # 文本切片
│   ├── text_utils.py     # 疑问词剥离 + 2/3-gram 抽词（检索与指代消解共用）
│   ├── retrieval.py      # 向量检索 + 关键词回退 + HyDE
│   ├── llm.py            # 指代消解 + 多轮对话 + 流式答案生成
│   ├── prompts.py        # Prompt 模板
│   └── templates/        # 前端界面
├── tests/                # pytest 测试
├── scripts/
│   └── cleanup_orphan_collections.py   # 孤儿向量集合清理（默认干跑）
├── docs/
│   └── code-review-2026-10-01.md       # 2026-10-01 代码审查原始清单（P0/P1/P2 连续编号）
├── .github/workflows/    # CI（pytest）与 Release
├── state/                # 会话状态，每个会话一个 JSON 文件（gitignore）
├── legacy/               # 历史版本归档
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## 🧹 维护

删对话会级联删掉它的向量集合，但如果状态文件丢过、换过 `session_id`、或手工删过 `state/`，
`chroma_db` 里就会留下没人引用的「孤儿集合」，只增不减。对账并清理：

```bash
python scripts/cleanup_orphan_collections.py            # 干跑：只打印计划，不删任何东西
python scripts/cleanup_orphan_collections.py --apply    # 确认计划后再执行
```

脚本只删名字以 `kb_conv_` 开头、且不被任何状态文件引用的集合；如果它一个被引用的集合都没
扫到（通常是路径指错了），会直接中止而不是把库删空。
