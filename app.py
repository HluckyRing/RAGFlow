# -*- coding: utf-8 -*-
import os
import ssl
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# ---------- 环境设置 ----------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


# ---------- 1. 读取 PDF ----------
def load_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ---------- 2. 智能切分（保留段落结构） ----------
def split_text(text, chunk_size=600, overlap=100):
    # 先按段落分割（保留更自然的语义单元）
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # 重叠部分取上一段的后 overlap 个字符
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ---------- 3. 关键词检索（备用） ----------
def retrieve_keyword(question, chunks):
    stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到",
                 "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她", "们",
                 "与", "或", "等", "但", "而", "因", "为", "对", "从", "把", "被", "让", "给", "跟", "比", "更", "最",
                 "太", "非常", "十分", "特别", "相当", "比较", "挺", "蛮", "可", "以", "能", "够", "得", "地", "也"}
    question_words = re.findall(r'[\u4e00-\u9fa5]{2,}', question) + re.findall(r'[a-zA-Z]{2,}', question.lower())
    question_words = [w for w in question_words if w not in stopwords]
    if not question_words:
        return chunks[:2]
    scored = []
    for chunk in chunks:
        score = sum(chunk.lower().count(w.lower()) for w in question_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [chunk for _, chunk in scored[:3]]  # 取前3个
    return top if top else chunks[:2]


# ---------- 4. 初始化向量（缓存） ----------
@st.cache_resource
def init_vector_store():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        import warnings
        warnings.filterwarnings("ignore")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5"
        )
        chroma_client = chromadb.PersistentClient(path="./chroma_db_web")
        try:
            collection = chroma_client.get_collection("knowledge_base")
            if collection.count() > 0:
                return collection, True
            else:
                raise Exception("空库")
        except:
            collection = chroma_client.create_collection(
                name="knowledge_base",
                embedding_function=embedding_fn
            )
            return collection, True
    except Exception as e:
        st.warning(f"向量检索不可用，将使用关键词匹配: {str(e)[:50]}...")
        return None, False


# ---------- 5. HyDE 检索 ----------
def hyde_retrieve(question, collection, full_text, use_vector, top_k=5):
    # 5.1 生成假设性答案（使用轻量级提示）
    try:
        hyde_response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
            messages=[
                {"role": "system",
                 "content": "你是一个知识渊博的助手，请针对用户问题，写一段可能包含答案的假设性文档片段。不必太详细，但要包含关键概念。"},
                {"role": "user", "content": f"问题：{question}\n请生成一段假设性答案："}
            ],
            temperature=0.5,
            max_tokens=200
        )
        hyde_answer = hyde_response.choices[0].message.content
        # 用假设答案作为检索词
        search_query = hyde_answer
    except:
        search_query = question  # 降级为原问题

    # 5.2 执行检索
    context_chunks = []
    if use_vector and collection is not None:
        try:
            count = collection.count()
            if count == 0:
                # 构建索引
                chunks = split_text(full_text)
                for i, chunk in enumerate(chunks):
                    collection.add(documents=[chunk], ids=[f"chunk_{i}"])
                count = collection.count()
            if count > 0:
                results = collection.query(
                    query_texts=[search_query],
                    n_results=min(top_k, count)
                )
                docs = results["documents"][0] if results["documents"] else []
                if docs:
                    context_chunks = docs
        except:
            pass

    # 如果向量没搜到，用关键词补
    if not context_chunks:
        chunks = split_text(full_text)
        context_chunks = retrieve_keyword(search_query, chunks)

    return context_chunks


# ---------- 6. 核心问答函数（增强版） ----------
def get_answer(question, full_text, collection, use_vector, history):
    # ----- 6.1 指代消解：从历史中提取实体 -----
    # 简单方法：取最近一轮用户问题中的关键词（超过3个字的名词）
    # 如果当前问题包含“它”、“这”、“其”等代词，则用上一轮问题的关键词替换
    def extract_entities(text):
        # 提取名词（2字以上中文词）
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        # 过滤停用词
        stop = {"什么", "如何", "为什么", "怎样", "哪个", "哪些", "一个", "这个", "那个", "这些", "那些", "它们",
                "他们", "她们", "自己", "大家", "咱们"}
        return [w for w in words if w not in stop]

    if len(question) < 15 and any(w in question for w in ["它", "这", "其", "那", "她", "他"]):
        # 取最近一条用户消息（如果有）
        user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
        if len(user_msgs) >= 2:
            prev_q = user_msgs[-2]
            entities = extract_entities(prev_q)
            if entities:
                # 将代词替换为实体的组合
                search_query = " ".join(entities) + " " + question
            else:
                search_query = question
        else:
            search_query = question
    else:
        search_query = question

    # ----- 6.2 检索（使用 HyDE）-----
    chunks = hyde_retrieve(search_query, collection, full_text, use_vector, top_k=5)
    if not chunks:
        return "（未找到任何相关内容）", []

    # 合并上下文（限制总长度，避免超出模型上下文）
    context = "\n\n".join(chunks)
    if len(context) > 3000:  # 保留前3000字符
        context = context[:3000] + "\n...(截断)"

    # 用于显示的引用片段（取前3个）
    source_chunks = chunks[:3]

    # ----- 6.3 构建带记忆的提示词 -----
    # 系统提示
    system_prompt = f"""你是一个严谨的知识库助手，必须严格基于【参考背景】回答用户问题。
要求：
1. 如果【参考背景】中没有明确答案，请直接说“背景中没有提到相关内容”，不要编造。
2. 回答时尽量引用原文的具体表述，并注明引用的位置（如“根据背景第X段”）。
3. 如果多轮对话中有指代，结合历史信息理解。

【参考背景】：
{context}
"""
    # 构建历史消息（保留最近的5轮）
    history_messages = []
    for msg in history[-5:]:
        if msg["role"] in ["user", "assistant"]:
            history_messages.append(msg)

    llm_messages = [
        {"role": "system", "content": system_prompt},
        *history_messages,
        {"role": "user", "content": question}
    ]

    # ----- 6.4 调用大模型 -----
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
            messages=llm_messages,
            temperature=0.2
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"调用大模型失败：{str(e)}"

    return answer, source_chunks


# ---------- 7. Streamlit UI ----------
st.set_page_config(page_title="📚 本地知识库问答（增强版）", layout="wide")
st.title("📚 本地 PDF 知识库问答系统（HyDE + 记忆增强）")
st.caption("支持上传 PDF，使用 HyDE 检索和多轮对话记忆，提高回答准确率。")

with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🔄 加载/刷新知识库"):
            with st.spinner("正在解析 PDF 并建立索引..."):
                full_text = load_pdf_text(uploaded_file)
                if not full_text.strip():
                    st.error("PDF 内容为空或无法提取文字。")
                else:
                    collection, use_vector = init_vector_store()
                    if collection and collection.count() == 0:
                        chunks = split_text(full_text)
                        for i, chunk in enumerate(chunks):
                            collection.add(documents=[chunk], ids=[f"chunk_{i}"])
                    st.session_state['full_text'] = full_text
                    st.session_state['collection'] = collection
                    st.session_state['use_vector'] = use_vector
                    st.session_state['messages'] = []
                    st.success(f"✅ 加载成功！共 {len(full_text)} 字符，已分块索引。")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for msg in st.session_state['messages']:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的问题："):
    if 'full_text' not in st.session_state or not st.session_state.get('full_text'):
        st.warning("⚠️ 请先在左侧上传 PDF 文件并点击加载。")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在 HyDE 检索并思考..."):
            full_text = st.session_state['full_text']
            collection = st.session_state.get('collection')
            use_vector = st.session_state.get('use_vector', False)

            answer, sources = get_answer(
                prompt,
                full_text,
                collection,
                use_vector,
                st.session_state['messages']
            )
            st.markdown(answer)
            if sources:
                with st.expander("📖 查看引用来源"):
                    for i, src in enumerate(sources):
                        st.caption(f"片段 {i + 1}:")
                        st.text(src[:300] + ("..." if len(src) > 300 else ""))
                        st.divider()
            st.session_state['messages'].append({"role": "assistant", "content": answer})