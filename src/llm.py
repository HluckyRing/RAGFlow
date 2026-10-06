from src.config import logger, client, MODEL_NAME, MAX_CONTEXT_LENGTH
from src.prompts import QA_SYSTEM_PROMPT_TEMPLATE
from src.retrieval import hyde_retrieve_hits
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


def retrieve_and_build_context(question, full_text, collection, use_vector, top_k=None, files=None):
    """检索，并按「块边界」装配参考背景，每块带上来源。

    两件事叠在一起：
    1. 按块边界装配（P1）：先按相关度决定要哪些块（预算优先给更相关的），再把选中的块
       按原文顺序拼起来；不再 `context[:MAX]` 从中间硬切，宁可比预算短也不给半截块。
    2. 带上来源（P2）：每块前面加 `【来源：文件名 · 第 N 页】`，引用才可核验。

    返回 (context, used_chunks)。used_chunks 是 context 里真正用到的块文本、与 context 同序。

    唯一例外：单块（含来源标签）本身就超过 MAX_CONTEXT_LENGTH（配置互相矛盾，例如把
    MAX_CONTEXT_LENGTH 调得比一个 chunk 还小）。此时只能截断这一块 —— 总比返回空好 ——
    并记一条 WARNING。
    """
    hits = hyde_retrieve_hits(question, collection, full_text, use_vector,
                              top_k=top_k, files=files)
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
        text, _meta, block = used[0]
        return block[:MAX_CONTEXT_LENGTH] + "\n...(截断)", [text]

    if len(used) < len(hits):
        logger.info("context 按块边界装配：保留 %d/%d 块（%d 字符，预算 %d）",
                    len(used), len(hits), total, MAX_CONTEXT_LENGTH)

    ordered = _order_pairs_by_position([(text, meta) for text, meta, _ in used], full_text)
    context = "\n\n".join(_format_source_block(text, meta) for text, meta in ordered)
    return context, [text for text, _ in ordered]



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
