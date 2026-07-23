# -*- coding: utf-8 -*-
import os
import ssl
import re
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# ---------- 设置环境 ----------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'      # 隐藏模型加载警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'      # 避免并行警告

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# ------------------- 1. 读取 PDF -------------------
def load_pdf_text():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None, "错误：没有找到 data 文件夹"
    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None, "错误：data 文件夹里没有 PDF 文件"
    try:
        reader = PdfReader(os.path.join(data_dir, pdf_files[0]))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            return None, "错误：PDF 内容为空或扫描版"
        return text, None
    except Exception as e:
        return None, f"错误：读取 PDF 失败 - {e}"

# ------------------- 2. 文本切分 -------------------
def split_text(text, chunk_size=500, overlap=50):
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + sent
            else:
                current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ------------------- 3. 关键词检索（备用） -------------------
def retrieve_keyword(question, chunks):
    stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她", "们", "与", "或", "等", "但", "而", "因", "为", "对", "从", "把", "被", "让", "给", "跟", "比", "更", "最", "太", "非常", "十分", "特别", "相当", "比较", "挺", "蛮", "可", "以", "能", "够", "得", "地", "也"}
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
    top = [chunk for _, chunk in scored[:2]]
    return top if top else chunks[:2]

# ------------------- 4. 初始化向量（静默） -------------------
USE_VECTOR = False
vector_collection = None

print("🔍 正在初始化检索引擎...")
try:
    import chromadb
    from chromadb.utils import embedding_functions
    import warnings
    warnings.filterwarnings("ignore")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        vector_collection = chroma_client.get_collection("knowledge_base")
        if vector_collection.count() > 0:
            USE_VECTOR = True
            print("✅ 向量检索已就绪（已缓存）")
        else:
            raise Exception("空库")
    except:
        vector_collection = chroma_client.create_collection(
            name="knowledge_base",
            embedding_function=embedding_fn
        )
        USE_VECTOR = True
        print("✅ 向量检索已就绪（新库）")
except Exception as e:
    print(f"⚠️ 向量检索不可用，将使用关键词匹配（原因：{str(e)[:30]}...）")
    USE_VECTOR = False
    vector_collection = None

# ------------------- 5. 问答函数（无调试打印） -------------------
def answer_question(question, full_text):
    global vector_collection, USE_VECTOR
    context = ""

    if USE_VECTOR and vector_collection is not None:
        try:
            count = vector_collection.count()
            if count == 0:
                chunks = split_text(full_text)
                for i, chunk in enumerate(chunks):
                    vector_collection.add(documents=[chunk], ids=[f"chunk_{i}"])
                count = vector_collection.count()

            if count > 0:
                results = vector_collection.query(
                    query_texts=[question],
                    n_results=min(2, count)
                )
                docs = results["documents"][0] if results["documents"] else []
                if docs:
                    context = "\n---\n".join(docs)
        except:
            pass  # 向量失败则自动降级

    # 如果 context 为空，则使用关键词
    if not context:
        chunks = split_text(full_text)
        context_chunks = retrieve_keyword(question, chunks)
        context = "\n---\n".join(context_chunks)

    if not context:
        context = "（未找到任何相关内容）"

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
        messages=[
            {"role": "system", "content": "请严格根据【参考背景】回答问题。如果背景中没有相关信息，请直接说'背景中没有提到相关内容'，不要编造。"},
            {"role": "user", "content": f"【参考背景】：\n{context}\n\n用户问题：{question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ------------------- 6. 主程序 -------------------
if __name__ == "__main__":
    print("=" * 50)
    print("📚 本地 PDF 知识库问答系统")
    print("=" * 50)

    full_text, error = load_pdf_text()
    if error:
        print(f"❌ {error}")
        input("\n按回车键退出...")
        exit()

    print(f"✅ PDF 加载成功（共 {len(full_text)} 字符）")
    print("💡 输入问题开始问答，输入 exit 退出\n")

    while True:
        try:
            question = input("💬 你的问题：").strip()
            if question.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break
            if not question:
                continue
            answer = answer_question(question, full_text)
            print(f"🤖 回答：{answer}\n")
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误：{e}\n")

    input("\n按回车键退出...")