# 🚀 RAGFlow — 多格式知识库问答系统

基于 **RAG（检索增强生成）** 架构的本地知识库智能问答工具。支持 **PDF / TXT / Markdown / Word / Excel / CSV** 多种文件格式，通过 **HyDE（假设性文档检索）** 提升召回率，强制大模型**引用原文**，确保回答可追溯、可验证。

## ✨ 核心亮点

- 📄 **多格式支持**：上传 PDF、Word、Excel、CSV、TXT、Markdown 等文件，自动提取文本
- 🔍 **HyDE 检索**：先让 AI 生成"假设性答案"再去向量库匹配，显著提升复杂问题命中率
- 💬 **多轮对话记忆**：结合历史上下文理解代词（"它"、"这"），避免语义丢失
- 📖 **精准溯源**：回答强制引用原文片段，点击"查看引用来源"即可核对
- ⚡ **工程降级**：向量检索不可用时自动切换关键词匹配，保证服务不中断

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
- **前端界面**：Streamlit

## 🚀 快速开始

### 配置 API 密钥

在项目根目录创建 `.env` 文件（**此文件不会被上传到 GitHub**）：

```
API_KEY="sk-你的密钥"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-v4-flash"
```

### 1. 克隆仓库

```bash
git clone https://github.com/HluckyRing/RAGFlow
cd RAGFlow
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
streamlit run app.py
```

打开浏览器，上传文件即可开始问答。

## 📁 项目结构

```
RAGFlow/
├── app.py                # Streamlit 主界面
├── src/
│   ├── config.py         # 环境配置、日志、OpenAI 客户端
│   ├── loaders.py        # 多格式文件加载器
│   ├── pdf_ingestion.py  # 文本切片
│   ├── retrieval.py      # 向量检索 + 关键词回退 + HyDE
│   ├── llm.py            # 指代消解 + 多轮对话 + 答案生成
│   └── prompts.py        # Prompt 模板
├── legacy/               # 历史版本归档
├── requirements.txt
└── pyproject.toml
```
