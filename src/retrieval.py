import re
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from src.config import (logger, client, MODEL_NAME, EMBEDDING_MODEL, COLLECTION_NAME,
                        VECTOR_DB_PATH, TOP_K, MAX_DISTANCE)
from src.pdf_ingestion import split_files, split_text
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


def _keyword_scores(question, chunks):
    """关键词打分：中文 2/3-gram + 英文单词。返回 [(分数, 下标)]，按分数降序。

    除以 sqrt(长度) 归一化，避免长段落仅因为字多就排到前面。
    """
    terms = extract_terms(question)
    if not terms or not chunks:
        return []
    scored = []
    for index, chunk in enumerate(chunks):
        low = chunk.lower()
        score = sum(low.count(t) * len(t) for t in terms)
        if score > 0:
            scored.append((score / (len(chunk) ** 0.5), index))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve_keyword(question, chunks, top_n=3):
    """关键词降级检索：中文 2/3-gram + 英文单词打分。

    返回按相关度排序的 chunk；一条都没沾上时返回空列表，由上层提示
    「未找到相关内容」，而不是拿不相关的段落去喂模型。
    """
    return [chunks[index] for _, index in _keyword_scores(question, chunks)[:top_n]]


def _generate_hyde(question):
    """让 LLM 生成一段假设性答案，用于向量检索；失败则退回原问题。"""
    if client is None:
        logger.warning("未配置 API_KEY，跳过 HyDE，直接用原问题做向量检索")
        return question
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


def _kept_indices(distances, max_distance):
    """距离阈值过滤：返回保留下来的下标；拿不到距离时返回 None（调用方按全保留处理）。

    这些 collection 建的时候没指定 hnsw:space，Chroma 用的是默认度量 l2（平方欧氏）；
    BGE 输出是归一化向量，所以距离 = 2 - 2*cos，取值范围 [0, 4]。
    """
    if not distances:
        return None
    kept, dropped = [], 0
    for index, dist in enumerate(distances):
        if dist <= max_distance:
            kept.append(index)
        else:
            dropped += 1

    if dropped:
        best = min(distances)
        if kept:
            logger.info("向量检索丢弃 %d 块（距离 > %.2f），保留 %d 块；最佳距离 %.4f",
                        dropped, max_distance, len(kept), best)
        else:
            logger.info("向量检索结果全部超过阈值 %.2f（最佳距离 %.4f），判定为未命中",
                        max_distance, best)
    return kept


def _filter_by_distance(docs, distances, max_distance=None):
    """丢掉距离过大（与问题不相关）的块，避免噪音进 prompt。

    阈值默认 0.55，可用环境变量 MAX_DISTANCE 覆盖。
    """
    if max_distance is None:
        max_distance = MAX_DISTANCE
    kept = _kept_indices(distances, max_distance)
    if kept is None:
        return list(docs)          # 拿不到距离就不敢乱丢
    return [docs[i] for i in kept if i < len(docs)]


def chroma_meta(meta):
    """Chroma 的 metadata 只收非 None 的基本类型，所以把值为 None 的字段删掉。"""
    cleaned = {k: v for k, v in (meta or {}).items() if v is not None}
    cleaned.setdefault("file_name", "未知文件")
    return cleaned


def _pairs_with_meta(files, full_text):
    """要检索的 [(块, 来源), ...]。

    给了 files 就按文件分别切片（块不跨文件，来源才唯一）；没给就退回旧的整篇切片、
    来源标成 None —— 老调用方、老测试与老索引都能继续工作。
    """
    if files:
        return split_files(files)
    return [(chunk, None) for chunk in split_text(full_text)]


def _kept_hits(docs, distances, metadatas, max_distance):
    """按距离过滤，连来源一起返回 [(文本, meta), ...]。"""
    kept = _kept_indices(distances, max_distance)
    if kept is None:
        kept = range(len(docs))
    hits = []
    for index in kept:
        if index >= len(docs):
            break
        meta = metadatas[index] if index < len(metadatas) else None
        hits.append((docs[index], meta if isinstance(meta, dict) else None))
    return hits


def hyde_retrieve_hits(question, collection, full_text, use_vector, top_k=None, files=None):
    """检索并带上来源：返回 [(chunk_text, meta), ...]，meta 可能是 None。

    meta 来自建索引时写进 Chroma 的 metadatas；老索引（P2 之前建的）没有 metadatas，
    这时 meta 是 None，上层会标注成「未标注」，绝不编造来源。
    """
    if top_k is None:
        top_k = TOP_K

    hits = []
    if use_vector and collection is not None:
        # 只有真要查向量库时才值得花一次 API 调用来生成 HyDE
        search_query = _generate_hyde(question)
        try:
            count = collection.count()
            if count == 0:
                pairs = _pairs_with_meta(files, full_text)
                for i, (chunk, meta) in enumerate(pairs):
                    collection.add(documents=[chunk], ids=[f"chunk_{i}"],
                                   metadatas=[chroma_meta(meta)])
                count = collection.count()
                logger.info("已为新文档建立向量索引，共 %d 块", count)
            if count > 0:
                results = collection.query(
                    query_texts=[search_query],
                    n_results=min(top_k, count)
                )
                docs = results["documents"][0] if results.get("documents") else []
                dists = results["distances"][0] if results.get("distances") else []
                metas = results["metadatas"][0] if results.get("metadatas") else []
                if docs:
                    hits = _kept_hits(docs, dists, metas, MAX_DISTANCE)
        except Exception as e:
            logger.warning("向量检索失败，降级为关键词检索: %s", str(e)[:80])

    if not hits:
        logger.info("使用关键词检索作为备用方案")
        # 关键词检索用原始问题：search_query 是 HyDE 生成的长段落，不适合做词频匹配
        pairs = _pairs_with_meta(files, full_text)
        texts = [text for text, _ in pairs]
        tops = _keyword_scores(question, texts)[:3]
        hits = [pairs[index] for _, index in tops]

    return hits


def hyde_retrieve(question, collection, full_text, use_vector, top_k=None, files=None):
    """兼容旧签名：只返回文本列表（老调用方与老测试不用改）。"""
    return [text for text, _ in hyde_retrieve_hits(
        question, collection, full_text, use_vector, top_k=top_k, files=files)]
