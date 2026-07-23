import streamlit as st
from src.loaders import load_file, SUPPORTED_TYPES
from src.pdf_ingestion import split_text
from src.retrieval import init_vector_store
from src.llm import get_answer


@st.cache_resource
def cached_init_vector_store():
    return init_vector_store()


st.set_page_config(page_title="📚 RAGFlow 知识库问答", layout="wide")
st.title("📚 RAGFlow 知识库问答系统（HyDE + 记忆增强）")
st.caption(f"支持上传 PDF / TXT / MD / DOCX / XLSX / CSV，使用 HyDE 检索和多轮对话记忆。")

with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_file = st.file_uploader("上传文件", type=SUPPORTED_TYPES)

    if uploaded_file is not None:
        if st.button("🔄 加载/刷新知识库"):
            with st.spinner("正在解析文件并建立索引..."):
                try:
                    full_text = load_file(uploaded_file)
                except Exception as e:
                    st.error(f"文件加载失败: {e}")
                    st.stop()
                if not full_text.strip():
                    st.error("文件内容为空或无法提取文字。")
                else:
                    collection, use_vector = cached_init_vector_store()
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
        st.warning("⚠️ 请先在左侧上传文件并点击加载。")
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
