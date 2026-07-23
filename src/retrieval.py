import re
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from src.config import logger, client, MODEL_NAME, EMBEDDING_MODEL, COLLECTION_NAME, VECTOR_DB_PATH, TOP_K
from src.pdf_ingestion import split_text

STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "它", "他", "她", "们", "与", "或", "等",
    "但", "而", "因", "为", "对", "从", "把", "被", "让", "给", "跟", "比",
    "更", "最", "太", "非常", "十分", "特别", "相当", "比较", "挺", "蛮",
    "可", "以", "能", "够", "得", "地", "也"
}

_embedding_fn = None


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def sanitize_collection_name(filename):
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    clean = re.sub(r'_+', '_', clean).strip('_') or "kb"
    suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"kb_{clean[:40]}_{suffix}"


def init_vector_store(collection_name=None):
    if collection_name is None:
        collection_name = COLLECTION_NAME
    try:
        embedding_fn = _get_embedding_fn()
        chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        try:
            collection = chroma_client.get_collection(collection_name)
            if collection.count() > 0:
                logger.info("向量检索已就绪（已缓存）: %s", collection_name)
                return collection, True
            raise Exception("空库")
        except Exception:
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            logger.info("向量检索已就绪（新库）: %s", collection_name)
            return collection, True
    except Exception as e:
        logger.warning("向量检索不可用，将使用关键词匹配: %s", str(e)[:80])
        return None, False


def retrieve_keyword(question, chunks):
    question_words = re.findall(r'[\u4e00-\u9fa5]{2,}', question) + \
                     re.findall(r'[a-zA-Z]{2,}', question.lower())
    question_words = [w for w in question_words if w not in STOPWORDS]
    if not question_words:
        return chunks[:2]
    scored = []
    for chunk in chunks:
        score = sum(chunk.lower().count(w.lower()) for w in question_words)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = [chunk for _, chunk in scored[:3]]
    return top if top else chunks[:2]


def hyde_retrieve(question, collection, full_text, use_vector, top_k=None):
    if top_k is None:
        top_k = TOP_K

    try:
        hyde_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "你是一个知识渊博的助手，请针对用户问题，写一段可能包含答案的假设性文档片段。不必太详细，但要包含关键概念。"},
                {"role": "user", "content": f"问题：{question}\n请生成一段假设性答案："}
            ],
            temperature=0.5,
            max_tokens=200
        )
        search_query = hyde_response.choices[0].message.content
    except Exception as e:
        logger.warning("HyDE 生成失败，降级为原问题: %s", str(e)[:80])
        search_query = question

    context_chunks = []
    if use_vector and collection is not None:
        try:
            count = collection.count()
            if count == 0:
                chunks = split_text(full_text)
                for i, chunk in enumerate(chunks):
                    collection.add(documents=[chunk], ids=[f"chunk_{i}"])
                count = collection.count()
                logger.info("已为新文档建立向量索引，共 %d 块", count)
            if count > 0:
                results = collection.query(
                    query_texts=[search_query],
                    n_results=min(top_k, count)
                )
                docs = results["documents"][0] if results["documents"] else []
                if docs:
                    context_chunks = docs
        except Exception as e:
            logger.warning("向量检索失败，降级为关键词检索: %s", str(e)[:80])

    if not context_chunks:
        logger.info("使用关键词检索作为备用方案")
        chunks = split_text(full_text)
        context_chunks = retrieve_keyword(search_query, chunks)

    return context_chunks
