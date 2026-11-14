from src.config import (logger, client, MODEL_NAME, MAX_CONTEXT_LENGTH,
                        COVERAGE_CROSS_FILE, COVERAGE_INTRA_FILE, COVERAGE_BACKFILL_MAX,
                        COVERAGE_PER_FILE_CANDIDATES, COVERAGE_INTRA_FILE_BLOCKS,
                        COVERAGE_INTRA_FILE_CANDIDATES)
from src.coverage import (coverage_report, is_navigational, looks_enumerative, missing_files,
                          blocks_that_fit, pick_target_file, prefer_content, sample_evenly,
                          sample_per_file)
from src.hits_merge import interleave_hits
from src.prompts import QA_SYSTEM_PROMPT_TEMPLATE, REFUSAL_MARKER
from src.retrieval import hyde_retrieve_hits, per_file_best
from src.text_utils import content_phrases

# 指代词。刻意不收裸「这」「那」：它们太常见（「这是什么」并不需要指代消解），
# 收进来只会把无关的上一轮问题拼进检索词。
PRONOUNS = (
    "它", "它们", "他", "他们", "她", "她们", "其", "该", "此",
    "上述", "前面", "刚才", "这个", "那个", "这些", "那些",
)
# 这些词里含有指代词，但本身不是指代（「其他指标有哪些」不该触发消解）
NON_PRONOUN_WORDS = ("其他", "其它", "其余", "其实", "其中", "尤其", "与其")
MAX_COREF_LENGTH = 15   # 太长的问题一般自带完整上下文


def extract_entities(text):
    """从上一轮问题里抽出实义片段（「什么是市盈率」→ [「市盈率」]）。"""
    return content_phrases(text)


def resolve_query(question, history):
    """短问题里出现指代词时，把上一轮问题的实体拼进来，避免语义丢失。

    注意：history 是【尚未包含当前提问】的历史快照（见 server.chat），
    所以「上一轮用户消息」就是 user_msgs[-1]。
    """
    if len(question) >= MAX_COREF_LENGTH:
        return question

    probe = question
    for word in NON_PRONOUN_WORDS:
        probe = probe.replace(word, "")
    if not any(p in probe for p in PRONOUNS):
        return question

    user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
    if not user_msgs:
        return question

    entities = extract_entities(user_msgs[-1])
    if entities:
        return " ".join(entities) + " " + question
    return question


def _order_pairs_by_position(hits, full_text):
    """把 (块, 来源) 按它们在原文里的先后排序。

    检索返回的是相关度序。跨相邻两块的答案如果按相关度序摆进 prompt，读起来是断的；
    按原文序更接近人阅读的顺序。

    位置取自 full_text.find(块文本)。索引是历史遗留（切片尺寸改过还没重建、或 P2 之前
    建的老索引）时可能找不到某个块，这种块排到最后并保持原有相对顺序 —— 只影响顺序，
    不丢内容。
    """
    if len(hits) < 2 or not full_text:
        return list(hits)
    positioned = []
    for i, (text, meta) in enumerate(hits):
        pos = full_text.find(text)
        positioned.append((pos, i, text, meta))
    # 找不到位置的（pos < 0）排最后，并按原有顺序保持稳定
    positioned.sort(key=lambda item: (item[0] < 0, item[0] if item[0] >= 0 else item[1]))
    return [(text, meta) for _, _, text, meta in positioned]


def _order_by_position(chunks, full_text):
    """（保持既有签名）只关心块文本顺序时用这个。"""
    return [text for text, _ in _order_pairs_by_position([(c, None) for c in chunks], full_text)]


def _source_label(meta):
    """把块的来源渲染成给模型看的标签。

    没有来源（老索引没有 metadatas、或调用方没给 files）就如实写「未标注」，
    绝不编造 —— 2026-07-24 那次引用功能被删，根因就是引用不准。
    """
    if not meta:
        return "未标注"
    name = meta.get("file_name") or "未知文件"
    section = meta.get("section")
    return f"{name} · {section}" if section else name


def _format_source_block(text, meta):
    return f"【来源：{_source_label(meta)}】\n{text}"


def ensure_file_coverage(question, collection, files, hits, top_k, backfill=None):
    """枚举 / 汇总型提问：在**同一个块预算**内保证每份文件都有代表块。

    为什么需要它（2026-11-12 第十二轮实测）：13 份文件的语料里，真实用户问「该知识库的
    主要内容是什么」时，装配后的 context 只覆盖 12 份 —— 缺的那一份是「按相关度排序 +
    `MAX_DISTANCE` 距离闸门」筛掉的，而且**没有任何提示**。这类提问要的是跨文件的代表性
    取样，不是「最相关的 k 块」。

    **两个开关，两套行为**（2026-11-14 拆开，之前共用一个 `COVERAGE_SAMPLING`）：

    - `COVERAGE_CROSS_FILE`：跨文件取样（按文件轮流取样 + 补位缺失文件）；
    - `COVERAGE_INTRA_FILE`：同文件跨节取样（问句点名章号时把那一份文件按节铺开）。
      两条路互斥：认出了目标文件就走文件内，否则才跨文件。只开文件内时，没点名章号的
      枚举型提问**原样返回**（那是另一套行为，不该顺手带上）。

    四条边界（都有测试钉住）：

    - 两个开关都关着 / 没有文件元数据 / 问句不像枚举 → **原样返回**，一次查询都不多花；
    - **块预算不变**：补位块排最前，总数仍截到 `top_k` —— 靠取样取胜，不靠塞更多 token；
    - 补位只针对**缺的那几份**，且最多 `COVERAGE_BACKFILL_MAX` 份 —— 不能让一次提问变成
      n 次检索；
    - 跨文件那条要求 ≥2 份文件（单文件没有「跨」可言），文件内跨节**一条文件就能用**。

    返回顺序：先补位块（保证装配阶段不会先被字符上限截掉），再是按文件轮流取样的结果。
    """
    if not files or not (COVERAGE_CROSS_FILE or COVERAGE_INTRA_FILE):
        return hits
    if not looks_enumerative(question):
        logger.info("闸门 enumerative=miss")
        return hits
    logger.info("闸门 enumerative=hit cross=%d intra=%d",
                bool(COVERAGE_CROSS_FILE), bool(COVERAGE_INTRA_FILE))

    budget = int(top_k) if top_k else len(hits or [])
    names = [name for name, _text in files if name]
    picker = backfill or per_file_best

    # 先看这句话是不是「要某一章/某一份文件的内容」——那种需求要的是**文件内部跨节**铺开，
    # 而不是跨文件代表性。两条路的判据互斥：认出了目标文件就走文件内取样。
    target = pick_target_file(question, names) if COVERAGE_INTRA_FILE else None
    if target:
        inside = prefer_content([(text, meta) for text, meta in hits or []
                                 if (meta or {}).get("file_name") == target])
        # 块数按**字符预算**反推（短页多取、长块少取），否则写死的块数要么浪费预算、
        # 要么被装配截断后只覆盖前半章。
        blocks = blocks_that_fit(inside, COVERAGE_INTRA_FILE_BLOCKS, MAX_CONTEXT_LENGTH)
        picked = sample_evenly(inside, blocks)
        if not inside or len(inside) < blocks:
            # 命中里这份文件的块太少（甚至一块都没有）：在**这份文件内部**多取一批候选再按节铺开。
            #
            # 「一块都没有」是真实会发生的（2026-11-14 扩 gold 时踩到）：整句检索走
            # **关键词兜底**时回来的块**没有 metadatas**，于是 `inside` 为空、`blocks_that_fit([])`
            # 又是 0，`len(inside) < blocks` 这个判断（0 < 0）不成立 → 直接返回空列表 → 模型拿到
            # 空 context → 空回答。所以这里必须显式判 `not inside`。
            extra = picker(collection, question, target,
                           top_k=COVERAGE_INTRA_FILE_CANDIDATES) or []
            merged = list(inside)
            seen = {(text, (meta or {}).get("chunk_index")) for text, meta in merged}
            for text, meta in extra:
                key = (text, (meta or {}).get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    merged.append((text, meta))
            pool = prefer_content(merged)
            picked = sample_evenly(pool, blocks_that_fit(pool, COVERAGE_INTRA_FILE_BLOCKS,
                                                         MAX_CONTEXT_LENGTH))
        logger.info("同文件跨节取样：目标文件「%s」，命中 %d 块 → 按节取 %d 块",
                    target, len(hits or []), len(picked))
        return picked

    if not COVERAGE_CROSS_FILE:
        logger.info("只开了同文件跨节取样（COVERAGE_CROSS_FILE=0）且问句没点名章号 → 原样返回")
        return hits
    if len(names) < 2:
        return hits

    # 否则按文件轮流取样：枚举 / 汇总型提问要的是跨文件的代表性
    # 先做「内容页在前、导航页在后」的稳定分区，否则每份文件的代表块很可能就是它的知识导图页。
    ordered = prefer_content(hits)
    sampled = sample_per_file(ordered, budget)
    fresh = missing_files(sampled, names)

    filled = []
    for name in fresh[:max(0, COVERAGE_BACKFILL_MAX)]:
        candidates = picker(collection, question, name, top_k=COVERAGE_PER_FILE_CANDIDATES)
        # 补位也要挑内容页：一份文件里最像「内容是什么」的那块往往正是它的目录页。
        content = [item for item in candidates or [] if not is_navigational(item[0])]
        filled.extend((content or candidates or [])[:1])

    if filled or fresh:
        report = coverage_report(list(filled) + list(sampled), names)
        logger.info("跨文件取样：%d 份文件里覆盖 %d 份（补位 %d 块%s）",
                    len(names), report["covered"], len(filled),
                    ("，仍缺 " + "、".join(report["missing"])) if report["missing"] else "")

    merged, seen = [], set()
    for text, meta in list(filled) + list(sampled):
        key = (text, (meta or {}).get("file_name"), (meta or {}).get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        merged.append((text, meta))
        if budget and len(merged) >= budget:
            break
    return merged


def _retrieve_hits(question, collection, full_text, use_vector, top_k, files, decompose,
                   history=None):
    """取检索块：**单问就照旧走整句**，复合问句才每问分路。

    三条边界都是刻意的：
    - 拆问只拆出 1 条（绝大多数提问）→ 与改造前逐位一致，不多花一次 embedding，也不掉
      HyDE 与 `intersect` 闸门；
    - 分路各问用的是**单探针**口径（不调 HyDE）—— 每问都调 HyDE 会让延迟变成 n 倍；
      代价是丢掉 `intersect` 的第二票，所以这条路只对**已经拆成 ≥2 问**的提问生效；
    - 分路一条都没捞到 → 退回整句检索（和今天一样），绝不让一次拆问把问答搞空。

    **多轮 + 复合**（2026-11-08 第七轮）：拆问看的是**消解后**的问句，而且拆出来的每一条
    还要再过一次 `resolve_query` —— 省略式追问被拆成单条时，主语往往还在上一轮里
    （「那这个规定还有效吗？」）。拆问模型拿到历史会自己补全，这里再兜一层规则，
    两层都便宜（`resolve_query` 是纯 Python，不花 API）。
    """
    questions = []
    if decompose is not None:
        try:
            questions = [str(text).strip() for text in (decompose(question, history) or [])]
        except TypeError:
            # 兼容只接受一个参数的拆问函数（评测里的替身、以及历史版本）
            try:
                questions = [str(text).strip() for text in (decompose(question) or [])]
            except Exception as e:
                logger.warning("拆问抛错，按整句检索: %s", str(e)[:80])
                questions = []
        except Exception as e:                                  # 拆问自己已经 fail-open 了，这里是双保险
            logger.warning("拆问抛错，按整句检索: %s", str(e)[:80])
            questions = []
        questions = [text for text in questions if text]
    if len(questions) <= 1:
        hits = hyde_retrieve_hits(question, collection, full_text, use_vector,
                                  top_k=top_k, files=files)
        # 跨文件取样只作用在「整句检索」这条路上：拆问已经按子问分路，各自的预算由
        # `interleave_hits` 管，两套取样叠在一起会把预算切碎两次。
        return ensure_file_coverage(question, collection, files, hits, top_k)

    resolved = [resolve_query(text, history or []) for text in questions]
    if resolved != questions:
        logger.info("多轮 + 复合：%d 条子问里有 %d 条做了指代消解",
                    len(questions), sum(1 for before, after in zip(questions, resolved)
                                        if before != after))
    part_hits = []
    for sub_question in resolved:
        part_hits.append(hyde_retrieve_hits(sub_question, collection, full_text, use_vector,
                                            top_k=top_k, files=files, use_hyde=False))
    merged = interleave_hits(part_hits, top_k)
    if merged:
        logger.info("每问分路检索：拆成 %d 问，单探针召回 %s 块 → 合并后 %d 块（预算 %d）",
                    len(resolved), [len(hits) for hits in part_hits], len(merged), top_k)
        return merged
    logger.warning("每问分路检索一条都没捞到，退回整句检索（拆成 %d 问）", len(resolved))
    return hyde_retrieve_hits(question, collection, full_text, use_vector,
                              top_k=top_k, files=files)


def retrieve_and_build_context(question, full_text, collection, use_vector, top_k=None,
                               files=None, decompose=None, history=None):
    """检索，并按「块边界」装配参考背景，每块带上来源。

    两件事叠在一起：
    1. 按块边界装配（P1）：先按相关度决定要哪些块（预算优先给更相关的），再把选中的块
       按原文顺序拼起来；不再 `context[:MAX]` 从中间硬切，宁可比预算短也不给半截块。
    2. 带上来源（P2）：每块前面加 `【来源：文件名 · 第 N 页】`，引用才可核验。

    返回 (context, used_blocks)。used_blocks 是 context 里真正用到的块**及其来源**、与 context
    同序：`[{"text": ..., "file_name": ..., "section": ...}]`（没有来源时只有 text）。
    交给 `source_list()` 去重后就得到前端要显示的来源 —— 前端只显示这里给出的东西、
    不自己推断，这样引用不可能凭空出现。

    唯一例外：单块（含来源标签）本身就超过 MAX_CONTEXT_LENGTH（配置互相矛盾，例如把
    MAX_CONTEXT_LENGTH 调得比一个 chunk 还小）。此时只能截断这一块 —— 总比返回空好 ——
    并记一条 WARNING。

    `decompose` 是**可选**的拆问函数（`src.decompose.decompose_question`）：给了它、且它把
    问句拆成 ≥2 条，就走「每问分路检索」（2026-11-08 实测同预算下覆盖 45.9% → 81.1%）。
    不给（默认）时行为与改造前完全一致。`history` 是**不含当前提问**的历史快照，
    只用于两件事：拆问时补全省略式追问、以及拆出来的每条子问过一遍 `resolve_query`。
    """
    hits = _retrieve_hits(question, collection, full_text, use_vector, top_k, files, decompose,
                          history=history)
    if not hits:
        return "", []

    used, total = [], 0                             # used: [(块文本, meta, 带标签的块)]
    for text, meta in hits:                         # 相关度序：预算先给更相关的块
        block = _format_source_block(text, meta)
        addition = len(block) + (2 if used else 0)  # 块之间是 "\n\n"
        if used and total + addition > MAX_CONTEXT_LENGTH:
            break
        used.append((text, meta, block))
        total += addition

    if total > MAX_CONTEXT_LENGTH:                  # 只有「第一块自己就超预算」会走到这
        logger.warning("单块长度（含来源标签）已超过 MAX_CONTEXT_LENGTH=%d，只能截断第一块；"
                       "请调大 MAX_CONTEXT_LENGTH 或调小 CHUNK_SIZE", MAX_CONTEXT_LENGTH)
        text, meta, block = used[0]
        return block[:MAX_CONTEXT_LENGTH] + "\n...(截断)", [_used_entry(text, meta)]

    if len(used) < len(hits):
        logger.info("context 按块边界装配：保留 %d/%d 块（%d 字符，预算 %d）",
                    len(used), len(hits), total, MAX_CONTEXT_LENGTH)

    ordered = _order_pairs_by_position([(text, meta) for text, meta, _ in used], full_text)
    context = "\n\n".join(_format_source_block(text, meta) for text, meta in ordered)
    return context, [_used_entry(text, meta) for text, meta in ordered]


def _used_entry(text, meta):
    """一条「用块」记录：块文本 + 可核验的来源；没有来源就只留 text。"""
    entry = {"text": text}
    name = (meta or {}).get("file_name")
    if name:
        entry["file_name"] = name
    section = (meta or {}).get("section")
    if section:
        entry["section"] = section
    return entry


def source_list(used):
    """把用块整理成给前端显示的来源列表：按（文件 + 页/工作表）去重，丢掉没有来源的。

    没有来源（P2 之前建的老索引）时**什么都不显示** —— 既不编造，也不显示「来源未标注」
    这种没有信息量的东西。顺序按首次出现，也就是原文顺序。
    """
    out, seen = [], set()
    for item in used or []:
        if not isinstance(item, dict):
            continue                      # 兼容老调用方传进来的纯文本列表
        name = item.get("file_name")
        if not name:
            continue
        section = item.get("section") or ""
        key = (name, section)
        if key in seen:
            continue
        seen.add(key)
        entry = {"file_name": name}
        if section:
            entry["section"] = section
        out.append(entry)
    return out



def _normalise_head(text):
    """去掉开头可能出现的空白与 markdown 强调符号，好让标记匹配稳定。

    模型很爱把标记写成 `**【无答案】**`（放在加粗里）。这里剥掉开头的空白与 `*`/`#`，
    匹配完再把标记本身删掉，剩下的 `**` 交给 markdown 渲染器去处理即可。
    """
    stripped = text.lstrip()
    while stripped[:1] in ("*", "#"):
        stripped = stripped[1:].lstrip()
    return stripped


def refusal_state(pending):
    """看流式输出的开头，判断「是不是拒答」。

    返回 True（拒答）/ False（正常回答）/ None（**还看不出来**，要再攒几个 token）。

    为什么需要三态：标记可能被拆在两个 token 里（`【无答` + `案】`），一拿到第一个 token 就
    下结论会把拒答误判成正常回答。攒到第一个不属于标记的字符才算「不是拒答」。
    """
    head = _normalise_head(pending or "")
    if not head:
        return None
    if head.startswith(REFUSAL_MARKER):
        return True
    if REFUSAL_MARKER.startswith(head):
        return None                     # 还是标记的前缀，继续等
    return False


def strip_refusal_marker(pending):
    """把开头的拒答标记（连同可能的 markdown 强调符）从文本里去掉。"""
    head = _normalise_head(pending or "")
    if not head.startswith(REFUSAL_MARKER):
        return pending
    rest = head[len(REFUSAL_MARKER):].lstrip("*").lstrip()
    return rest


class LLMStreamError(RuntimeError):
    """流式调用大模型失败。

    刻意不让 stream_answer 把错误文案当成回答 yield 出去：那样 server 会把它当
    正常回答写进 messages，下一轮又被当作上下文喂回模型。呈现方式交给 server 决定。
    """


def stream_answer(question, context, history):
    # client 为 None 表示没配 API_KEY（见 config.build_client）。此时不该把
    # AttributeError 当成「调用大模型失败」，而要直接告诉用户怎么配。
    if client is None:
        raise LLMStreamError("未配置 API_KEY：请在项目根目录的 .env 中填入 API_KEY 后重启服务")
    system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(context=context)
    history_messages = [msg for msg in history[-5:] if msg["role"] in ["user", "assistant"]]
    llm_messages = [
        {"role": "system", "content": system_prompt},
        *history_messages,
        {"role": "user", "content": question}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            temperature=0.2,
            stream=True
        )
        for chunk in response:
            # 有些 OpenAI 兼容实现会发 usage-only chunk（choices 为空），必须判空
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                yield content
    except Exception as e:
        logger.error("LLM 流式调用失败: %s", e)
        raise LLMStreamError(f"调用大模型失败：{e}") from e
