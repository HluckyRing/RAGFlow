import re
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from src.config import logger, client, MODEL_NAME, EMBEDDING_MODEL, COLLECTION_NAME, VECTOR_DB_PATH, TOP_K
from src.pdf_ingestion import split_text
from src.prompts import HYDE_SYSTEM_PROMPT
from src.text_utils import extract_terms

_embedding_fn = None
_chroma_client = None


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    return _chroma_client


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
        chroma_client = _get_client()
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


def drop_collection(collection_name):
    """删除整个向量集合（删除对话时调用，避免集合只增不减）。"""
    if not collection_name:
        return False
    try:
        _get_client().delete_collection(collection_name)
        logger.info("已删除向量集合: %s", collection_name)
        return True
    except Exception as e:
        logger.warning("删除向量集合失败（%s）: %s", collection_name, str(e)[:80])
        return False


def retrieve_keyword(question, chunks, top_n=3):
    """关键词降级检索：中文 2/3-gram + 英文单词打分。

    返回按相关度排序的 chunk；一条都没沾上时返回空列表，由上层提示
    「未找到相关内容」，而不是拿不相关的段落去喂模型。
    """
    terms = extract_terms(question)
    if not terms or not chunks:
        return []
    scored = []
    for chunk in chunks:
        low = chunk.lower()
        score = sum(low.count(t) * len(t) for t in terms)
        if score > 0:
            # 除以 sqrt(长度) 归一化，避免长段落仅因为字多就排到前面
            scored.append((score / (len(chunk) ** 0.5), chunk))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


def _generate_hyde(question):
    """让 LLM 生成一段假设性答案，用于向量检索；失败则退回原问题。"""
    try:
        hyde_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{question}\n请生成一段假设性答案："}
            ],
            temperature=0.5,
            max_tokens=200
        )
        return (hyde_response.choices[0].message.content or "").strip() or question
    except Exception as e:
        logger.warning("HyDE 生成失败，降级为原问题: %s", str(e)[:80])
        return question


def hyde_retrieve(question, collection, full_text, use_vector, top_k=None):
    if top_k is None:
        top_k = TOP_K

    context_chunks = []
    if use_vector and collection is not None:
        # 只有真要查向量库时才值得花一次 API 调用来生成 HyDE
        search_query = _generate_hyde(question)
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
        # 关键词检索用原始问题：search_query 是 HyDE 生成的长段落，不适合做词频匹配
        context_chunks = retrieve_keyword(question, chunks)

    return context_chunks
