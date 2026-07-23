import streamlit as st
from src.loaders import load_file, SUPPORTED_TYPES
from src.pdf_ingestion import split_text
from src.retrieval import init_vector_store, sanitize_collection_name
from src.llm import get_answer

st.set_page_config(page_title="RAGFlow", page_icon="📚", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="stSidebar"] { background-color: #f7f7f8; }

    [data-testid="stSidebarUserContent"] { padding: 1rem 0.5rem; }

    .stChatMessage { border-radius: 14px !important; }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem; line-height: 1.7;
    }

    div[data-testid="stExpander"] {
        border: none !important; box-shadow: none !important;
    }

    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


def _add_kb(uploaded_file):
    full_text = load_file(uploaded_file)
    if not full_text.strip():
        return False, "文件内容为空"
    kb_name = uploaded_file.name
    cname = sanitize_collection_name(kb_name)
    collection, use_vector = init_vector_store(cname)
    if collection and collection.count() == 0:
        chunks = split_text(full_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])
    st.session_state.kbs[kb_name] = {
        'full_text': full_text,
        'collection': collection,
        'use_vector': use_vector,
        'messages': [],
    }
    st.session_state.active_kb = kb_name
    return True, f"加载成功，{len(full_text)} 字符"


if 'kbs' not in st.session_state:
    st.session_state.kbs = {}
if 'active_kb' not in st.session_state:
    st.session_state.active_kb = None

# ── Sidebar ──
with st.sidebar:
    c1, c2 = st.columns([4, 1])
    c1.markdown("### 🚀 RAGFlow")
    if c2.button("＋", help="新建知识库", use_container_width=True):
        st.session_state._nav_upload = True
        st.rerun()

    st.divider()

    if st.session_state.kbs:
        st.caption("知识库列表")
        for kb_name in sorted(st.session_state.kbs.keys()):
            is_active = st.session_state.active_kb == kb_name
            dc1, dc2 = st.columns([7, 1])
            label = f"{'●' if is_active else '○'}  {kb_name[:28]}"
            if dc1.button(label, key=f"kb_{kb_name}", use_container_width=True,
                          type="primary" if is_active else "secondary"):
                st.session_state.active_kb = kb_name
                st.rerun()
            if dc2.button("✕", key=f"rm_{kb_name}", help="删除此知识库"):
                del st.session_state.kbs[kb_name]
                if st.session_state.active_kb == kb_name:
                    st.session_state.active_kb = next(iter(st.session_state.kbs), None) if st.session_state.kbs else None
                st.rerun()
    else:
        st.caption("暂无知识库")
        st.caption("点击左上角 ＋ 上传文件")

# ── Main ──
active_kb = st.session_state.active_kb
active = st.session_state.kbs.get(active_kb) if active_kb else None

if active is None:
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("""
        <div style="text-align:center; margin-bottom:24px;">
            <h2 style="font-weight:700; color:#222; margin-bottom:6px;">有什么可以帮助你的？</h2>
            <p style="color:#999; font-size:0.95rem;">上传 PDF / Word / Excel / TXT / Markdown，开始智能问答</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "上传文件", type=SUPPORTED_TYPES,
            label_visibility="collapsed", key="home_upload"
        )
        if uploaded_file and uploaded_file.name not in st.session_state.kbs:
            with st.spinner("正在解析文件..."):
                try:
                    ok, msg = _add_kb(uploaded_file)
                except Exception as e:
                    st.error(f"加载失败: {e}")
                else:
                    if ok:
                        st.rerun()
                    else:
                        st.error(msg)
        elif uploaded_file:
            st.session_state.active_kb = uploaded_file.name
            st.rerun()
else:
    st.caption(f"📄 {active_kb}")

    for msg in active['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.expander("📎 上传新文件"):
        new_file = st.file_uploader(
            "选择文件", type=SUPPORTED_TYPES,
            label_visibility="collapsed", key="inline_upload"
        )
        if new_file and new_file.name not in st.session_state.kbs:
            with st.spinner("正在解析..."):
                try:
                    ok, msg = _add_kb(new_file)
                except Exception as e:
                    st.error(f"加载失败: {e}")
                else:
                    if ok:
                        st.rerun()
        elif new_file:
            st.session_state.active_kb = new_file.name
            st.rerun()

    if prompt := st.chat_input("输入你的问题...", key="chat_input"):
        with st.chat_message("user"):
            st.markdown(prompt)
        active['messages'].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer, sources = get_answer(
                    prompt,
                    active['full_text'],
                    active['collection'],
                    active['use_vector'],
                    active['messages'],
                )
            st.markdown(answer)
            if sources:
                with st.expander("📖 引用来源"):
                    for i, src in enumerate(sources):
                        st.caption(f"片段 {i + 1}")
                        st.text(src[:300] + ("..." if len(src) > 300 else ""))
                        st.divider()
            active['messages'].append({"role": "assistant", "content": answer})
