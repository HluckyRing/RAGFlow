import re
import hashlib
from collections import OrderedDict

import chromadb
from chromadb.utils import embedding_functions

import src.pdf_ingestion as pdf_ingestion
from src import rerank as rerank_mod
from src.config import (logger, client, MODEL_NAME, EMBEDDING_MODEL, COLLECTION_NAME,
                        VECTOR_DB_PATH, TOP_K, MAX_DISTANCE, RERANK_ENABLED, RERANK_MODEL,
                        RERANK_CANDIDATES, RERANK_TOP_N, RERANK_MIN_SCORE,
                        FUSION_ENABLED, RRF_K, FUSION_GATE)
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


def _keyword_scores(question, chunks, lowered=None):
    """关键词打分：中文 2/3-gram + 英文单词。返回 [(分数, 下标)]，按分数降序。

    除以 sqrt(长度) 归一化，避免长段落仅因为字多就排到前面。
    `lowered` 是预先算好的小写正文（切片缓存里存着），省掉每次把上千块重新 lower 一遍；
    归一化仍用原文长度，所以传不传它算出来的分数逐位一致。
    """
    terms = extract_terms(question)
    if not terms or not chunks:
        return []
    scored = []
    for index, chunk in enumerate(chunks):
        low = chunk.lower() if lowered is None else lowered[index]
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
            max_tokens=200,
            # deepseek-flash 是推理模型：它先把预算花在 reasoning_content 上。200 的 cap
            # 会被推理吃光、content 返回空串（2026-10-06 实测：finish_reason=length、
            # reasoning_tokens=200、content 长度 0），于是下面的兜底把 HyDE 悄悄变成了
            # 「原问题直查」，线上一直如此而没有任何告警。HyDE 只要一段像文档的片段、
            # 不需要推理链，所以这一次调用明确关掉思考（成本与延迟反而更低）。
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (hyde_response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("HyDE 返回空内容（finish_reason=%s），降级为原问题做向量检索",
                           getattr(hyde_response.choices[0], "finish_reason", "?"))
            return question
        return content
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
    return _cached_pairs(files, full_text)[0]


# ── 切片缓存 ──
# 关键词兜底每问一次都要把整篇正文重新切一遍：真实语料 297KB / 1199 块实测 8.3 ms，
# 而同一份对话的正文几乎不变。缓存键是「正文内容 + 当前切片参数」的 sha1，所以上传/
# 删除文件后天然失效（不需要谁去主动 clear）；容量有上限，免得长驻服务把多轮对话的
# 正文全留在内存里。
PAIRS_CACHE_MAX = 8
_split_cache = OrderedDict()


def clear_pairs_cache():
    """清空切片缓存。评测脚本换切片参数、或测试用例之间要互不干扰时用。"""
    _split_cache.clear()


def _content_key(files, full_text):
    """缓存键：内容 + 切片参数。用的是 pdf_ingestion 当前的常量（评测会临时改它）。"""
    digest = hashlib.sha1()
    digest.update(f"{pdf_ingestion.CHUNK_SIZE}:{pdf_ingestion.CHUNK_OVERLAP}".encode())
    if files:
        for name, text in files:
            digest.update(b"\x01")
            digest.update((name or "").encode())
            digest.update(b"\x02")
            digest.update((text or "").encode())
    else:
        digest.update(b"\x03")
        digest.update((full_text or "").encode())
    return digest.hexdigest()


def _cached_pairs(files, full_text):
    """返回 `([(块, meta), ...], [小写块, ...])`；同一份内容只切一次。"""
    key = _content_key(files, full_text)
    entry = _split_cache.get(key)
    if entry is not None:
        _split_cache.move_to_end(key)          # LRU：常用的留到最后
        return entry
    if files:
        pairs = split_files(files)
    else:
        pairs = [(chunk, None) for chunk in split_text(full_text)]
    entry = (pairs, [chunk.lower() for chunk, _ in pairs])
    _split_cache[key] = entry
    while len(_split_cache) > PAIRS_CACHE_MAX:
        _split_cache.popitem(last=False)
    return entry


def _probe_hits(docs, distances, metadatas):
    """一路向量召回 → [(文本, meta, 距离)]，保持 Chroma 的距离序（近的在前）。"""
    hits = []
    for index, text in enumerate(docs or []):
        meta = metadatas[index] if index < len(metadatas or []) else None
        dist = distances[index] if index < len(distances or []) else None
        hits.append((text, meta if isinstance(meta, dict) else None,
                     None if dist is None else float(dist)))
    return hits


def _rrf_fuse(ranked_lists, k=None):
    """倒数排名融合（RRF）：每路第 rank 位的文本得 1/(k+rank) 分，跨路相加。

    返回 [(文本, 融合分)]，分数降序（同分按文本排，保证结果可重复）。
    用名次而不是距离：两路的距离虽是同一把尺子，但「这路排第 1」比「0.42 vs 0.44」
    稳得多，而且 RRF 天然把**两路都认可**的块顶到前面。
    """
    if k is None:
        k = RRF_K
    scores = {}
    for ranked in ranked_lists:
        for rank, text in enumerate(ranked, 1):
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _fuse_probes(probes, max_distance=None, gate=None):
    """多路向量召回 → `(全部候选, 过闸门的候选)`，都按融合分降序。

    元素是 `(文本, meta, 两路里最近的距离)`。闸门口径由 `FUSION_GATE` 决定：

    - `union`（任一路近就算近）：HyDE 侧漂了、原问题侧还贴着答案的提问能被救回来；
    - `intersect`（**两路都得近**）：无关提问往往只是某一侧「像」，两路都贴上去才进
      context —— 这是给「无关问题被强行作答」用的精度闸门。

    两种口径都保留在代码里是因为它们是两个不同的假设，只能靠实测选（见
    docs/retrieval-baseline-2026-10-06.md「双路融合」一节）。
    """
    if max_distance is None:
        max_distance = MAX_DISTANCE
    if gate is None:
        gate = FUSION_GATE
    active = [probe for probe in probes if probe]
    entries = {}                                   # 文本 -> 记账
    for probe_index, probe in enumerate(probes):
        for rank, (text, meta, dist) in enumerate(probe, 1):
            entry = entries.get(text)
            if entry is None:
                entry = {"score": 0.0, "meta": meta, "min_dist": dist,
                         "order": len(entries), "seen": set(), "far": 0}
                entries[text] = entry
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["seen"].add(probe_index)
            if entry["meta"] is None and meta is not None:
                entry["meta"] = meta
            near = dist is None or dist <= max_distance
            if dist is not None and (entry["min_dist"] is None or dist < entry["min_dist"]):
                entry["min_dist"] = dist
            if not near:
                entry["far"] += 1
    ranked = sorted(entries.items(),
                    key=lambda kv: (-kv[1]["score"],
                                    kv[1]["min_dist"] if kv[1]["min_dist"] is not None
                                    else float("inf"),
                                    kv[1]["order"]))
    filled = [(text, entry["meta"], entry["min_dist"]) for text, entry in ranked]

    def _kept(text):
        entry = entries[text]
        if gate == GATE_INTERSECT:
            # 两路都召回了它、而且每一路都在阈值内
            return bool(active) and len(entry["seen"]) >= len(active) and entry["far"] == 0
        return entry["far"] < len(entry["seen"]) or entry["min_dist"] is None

    return filled, [hit for hit in filled if _kept(hit[0])]


# 检索路径标签：向量召回与「阈值全丢后的关键词兜底」的分数含义完全不同 ——
# 前者是 l2 距离、后者是 2-gram 词面打分。评测必须能把两者分开统计，
# 否则会把兜底捞回来的样本当成「向量检索质量」，从而误判阈值的好坏。
PATH_VECTOR = "vector"
PATH_KEYWORD = "keyword"

# 双路融合的闸门口径
GATE_UNION = "union"          # 任一路近就算近（提高召回）
GATE_INTERSECT = "intersect"  # 两路都得近（提高精度）


def _probe_queries(question, search_query):
    """要拿哪几路文本去查向量库。开了融合就是 [HyDE, 原问题]，否则只有 HyDE 一路。

    HyDE 退化（生成失败/空内容）时假设文本就是原问题，这时**不**再查第二遍 ——
    重复探针只会白花一次 embedding，融合结果也不会变。
    """
    if FUSION_ENABLED and search_query.strip() != question.strip():
        return [search_query, question]
    return [search_query]


def hyde_retrieve_hits_with_path(question, collection, full_text, use_vector, top_k=None, files=None,
                                 use_hyde=True):
    """同 `hyde_retrieve_hits`，但把「走的哪条路」一起返回：`(PATH_*, hits)`。

    负样本的样子（向量全丢、兜底也没命中）返回的仍然是 `PATH_KEYWORD` —— 那条路确实
    跑了，只是没捞出东西。**不要**用这个标签判断「有没有检索到内容」，那要看 hits 是否为空。

    `use_hyde=False` 是**单探针**口径：原问题直接当向量探针，不生成假设文本、也就没有
    双路融合。给「每问分路检索」用（2026-11-08）—— 复合问句拆成 n 个子问题后，如果每问
    都再调一次 HyDE，延迟会变成 n 倍（实测一次 HyDE 0.7–1.3 s）；单探针的代价是丢掉
    `intersect` 闸门的第二票，但它仍然过 `MAX_DISTANCE` 距离闸门，且**只有被拆成 ≥2 问时
    才走这条路**（单问提问的行为一个字都没变）。
    """
    if top_k is None:
        top_k = TOP_K

    hits = []
    if use_vector and collection is not None:
        # 只有真要查向量库时才值得花一次 API 调用来生成 HyDE
        search_query = _generate_hyde(question) if use_hyde else question
        probes_wanted = _probe_queries(question, search_query)
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
                # 开了重排就多召回一些候选：距离在这条路上不再当最终裁判，别先用它砍人
                limit = max(top_k, RERANK_CANDIDATES) if RERANK_ENABLED else top_k
                # 两路探针合成**一次** query 调用：Chroma 一次算两条 embedding 再各自检索，
                # 比两次往返省一次进程内开销。
                results = collection.query(
                    query_texts=probes_wanted,
                    n_results=min(limit, count)
                )
                docs_all = results.get("documents") or []
                dists_all = results.get("distances") or []
                metas_all = results.get("metadatas") or []
                # 老替身可能只回一条列表：缺的那一路当作没召回，绝不按错位去读
                probes = []
                for i, docs in enumerate(docs_all):
                    probes.append(_probe_hits(
                        docs,
                        dists_all[i] if i < len(dists_all) else [],
                        metas_all[i] if i < len(metas_all) else []))
                if probes and any(probes):
                    ordered, kept = _fuse_probes(probes, MAX_DISTANCE)
                    if len(probes) > 1:
                        logger.info(
                            "双路融合（%s）：HyDE 探针 %d 块 / 原问题探针 %d 块 → 候选 %d 块，"
                            "过距离闸门 %d 块", FUSION_GATE, len(probes[0]), len(probes[1]),
                            len(ordered), len(kept))
                    if RERANK_ENABLED:
                        # 打分用**原问题**：跨编码器的训练分布是 question → passage，
                        # HyDE 文本（像文档的片段）当 query 属于分布外。
                        # 候选池是**融合排序后的全部候选**，不经距离/闸门预砍 —— 开了重排就由
                        # 它当最终裁判（闸门口径只在没有重排时才决定取舍）。
                        candidates = [(text, meta) for text, meta, _dist in ordered][:max(RERANK_CANDIDATES, 1)]
                        reranked = rerank_mod.rerank(question, candidates,
                                                     RERANK_TOP_N, RERANK_MIN_SCORE)
                        if reranked is None:
                            # 重排不可用：退回距离闸门，绝不把一堆远块原样塞给模型
                            hits = [(text, meta) for text, meta, _dist in kept]
                        else:
                            hits = [(text, meta) for text, meta, _score in reranked]
                            logger.info("重排 %d 个候选 → 保留 %d 块（分数下限 %.2f）",
                                        len(candidates), len(hits), RERANK_MIN_SCORE)
                    else:
                        hits = [(text, meta) for text, meta, _dist in kept]
        except Exception as e:
            logger.warning("向量检索失败，降级为关键词检索: %s", str(e)[:80])

    if not hits:
        logger.info("使用关键词检索作为备用方案")
        # 关键词检索用原始问题：search_query 是 HyDE 生成的长段落，不适合做词频匹配
        pairs, lowered = _cached_pairs(files, full_text)
        texts = [text for text, _ in pairs]
        tops = _keyword_scores(question, texts, lowered=lowered)[:3]
        hits = [pairs[index] for _, index in tops]
        return PATH_KEYWORD, hits

    return PATH_VECTOR, hits


def hyde_retrieve_hits(question, collection, full_text, use_vector, top_k=None, files=None,
                       use_hyde=True):
    """检索并带上来源：返回 [(chunk_text, meta), ...]，meta 可能是 None。

    meta 来自建索引时写进 Chroma 的 metadatas；老索引（P2 之前建的）没有 metadatas，
    这时 meta 是 None，上层会标注成「未标注」，绝不编造来源。

    `use_hyde=False` = 单探针口径（见 `hyde_retrieve_hits_with_path`）。
    """
    return hyde_retrieve_hits_with_path(
        question, collection, full_text, use_vector, top_k=top_k, files=files,
        use_hyde=use_hyde)[1]


def per_file_best(collection, query_text, file_name, top_k=1):
    """在**指定文件内部**取最相关的块 —— 跨文件取样的补位用（2026-11-12 第十二轮）。

    与主检索刻意不同的两点，都是为「覆盖」而不是「精度」：

    - 用 `where={"file_name": ...}` 把候选限定到这一份文件，所以取回来的必定是它的块；
    - **不过 `MAX_DISTANCE` 距离闸门**。枚举 / 汇总型提问里，「离问句远」的那份文件正是要被
      补进来的那一个；距离闸门会继续把边缘但必要的那份挡在外面，补位就白做了。

    任何失败都返回 `[]` 并记 WARNING：补位是锦上添花，绝不许把一次正常问答搞崩。
    """
    if collection is None or not file_name:
        return []
    try:
        results = collection.query(query_texts=[str(query_text)],
                                   n_results=max(1, int(top_k)),
                                   where={"file_name": file_name})
    except Exception as e:                                       # noqa: BLE001
        logger.warning("跨文件取样补位查询失败（%s）：%s", file_name, str(e)[:80])
        return []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    return [(doc, metas[index] if index < len(metas) else None)
            for index, doc in enumerate(docs or [])]


def hyde_retrieve(question, collection, full_text, use_vector, top_k=None, files=None):
    """兼容旧签名：只返回文本列表（老调用方与老测试不用改）。"""
    return [text for text, _ in hyde_retrieve_hits(
        question, collection, full_text, use_vector, top_k=top_k, files=files)]
