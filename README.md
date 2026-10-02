# 🚀 RAGFlow — 多格式知识库问答系统

基于 **RAG（检索增强生成）** 架构的本地知识库智能问答工具。支持 **PDF / TXT / Markdown / Word / Excel / CSV** 多种文件格式，通过 **HyDE（假设性文档检索）** 提升召回率。

## ✨ 核心亮点

- 📄 **多格式支持**：上传 PDF、Word、Excel、CSV、TXT、Markdown 等文件，自动提取文本
- 🔍 **HyDE 检索**：先让 AI 生成"假设性答案"再去向量库匹配，显著提升复杂问题命中率
- 💬 **多轮对话记忆**：短问题里的指代词（"它"、"这个"）自动接上上一轮的问题，避免检索时语义丢失
- 💾 **状态持久化**：对话和文件数据自动保存，重启不丢失
- 🔒 **会话隔离**：每个浏览器会话独立存储，互不可见、互不覆盖
- ✅ **测试覆盖**：`pytest tests -v` 覆盖状态隔离、并发保存、旧数据迁移、指代消解、关键词降级、多编码导入、路径锚定、多文件上传/单文件删除与前端注入守卫
- ⚡ **工程降级**：向量检索不可用时自动切换关键词匹配（中文 2/3-gram 打分），且降级时不再浪费一次 HyDE 调用；缺 `API_KEY` 时服务照常启动，只降级并给中文提示
- 🧱 **入口防护**：单次上传默认上限 20 MB（按整批总字节数判断，超限在入口直接拒绝、不进解析），大模型调用默认 60 秒超时
- 🗂 **多文件知识库**：一次可选或拖拽多个文件（一次请求提交，只重建一次索引），文件列表里可单独移除某个文件并自动重建索引

## 📋 支持格式

| 格式 | 扩展名 | 依赖 |
|------|--------|------|
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` | openpyxl |
| CSV | `.csv` | 标准库 |
| 文本 | `.txt` `.md` | 无 |

## 🛠 技术栈

- **语言**：Python 3.11+
- **大模型**：DeepSeek API（兼容 OpenAI 格式）
- **向量引擎**：ChromaDB + BAAI/bge-small-zh-v1.5（中文 Embedding）
- **后端**：FastAPI + SSE 流式输出
- **前端**：原生 HTML/CSS/JS（零依赖）

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
```

### 1. 安装依赖

```bash
git clone https://github.com/HluckyRing/RAGFlow
cd RAGFlow
pip install -r requirements.txt
```

### 2. 启动

```bash
python src/server.py
```

打开浏览器访问 `http://localhost:8080`。

## 📁 项目结构

```
RAGFlow/
├── src/
│   ├── server.py         # FastAPI 服务端（启动入口）
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
├── state/                # 会话状态，每个会话一个 JSON 文件（gitignore）
├── legacy/               # 历史版本归档
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```
