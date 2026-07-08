# 📚 Local PDF RAG Q&A System（本地PDF知识库问答系统）

基于 **RAG（检索增强生成）** 架构，实现了一个完全本地化、支持多轮对话的 PDF 智能问答工具。系统通过 **HyDE（假设性文档检索）** 提升召回率，并强制大模型**引用原文**，确保回答可追溯、可验证。

## ✨ 核心亮点
- 📄 **本地知识库**：上传 PDF，自动切片、向量化，数据不出本地，保护隐私。
- 🔍 **HyDE 检索**：先让 AI 生成“假设性答案”再去向量库匹配，显著提升复杂问题命中率。
- 💬 **多轮对话记忆**：结合历史上下文理解代词（“它”、“这”），避免语义丢失。
- 📖 **精准溯源**：回答强制引用原文片段，点击“查看引用来源”即可核对。
- ⚡ **工程降级**：向量检索不可用时自动切换关键词匹配，保证服务不中断。

## 🛠 技术栈
- **语言**：Python 3.11
- **大模型**：DeepSeek API（兼容 OpenAI 格式）
- **向量引擎**：ChromaDB + BAAI/bge-small-zh-v1.5（中文轻量级 Embedding 模型）
- **前端界面**：Streamlit
- **部署**：Docker（支持一键运行）

## 🚀 快速开始

### 配置 API 密钥
本项目需要 DeepSeek API Key。请将密钥填入项目根目录的 `.env` 文件中（**注意：此文件不会被上传到 GitHub**）：
API_KEY="sk-你的密钥"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-v4-flash"

### 1. 克隆仓库
```bash
git clone https://github.com/HluckyRing/ai_rag_project
cd ai_rag_project