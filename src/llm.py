from src.config import logger, client, MODEL_NAME, MAX_CONTEXT_LENGTH
from src.prompts import QA_SYSTEM_PROMPT_TEMPLATE
from src.retrieval import hyde_retrieve
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


def _order_by_position(chunks, full_text):
    """把选中的块按它们在原文里的先后排序。

    检索返回的是相关度序。跨相邻两块的答案如果按相关度序摆进 prompt，读起来是断的；
    按原文序更接近人阅读的顺序。

    位置取自 full_text.find(chunk)。索引是历史遗留（例如切片尺寸改过但还没重建）
    时可能找不到某个块，这种块排到最后并保持原有相对顺序 —— 只影响顺序，不丢内容。
    """
    if len(chunks) < 2 or not full_text:
        return list(chunks)
    positioned = []
    for i, chunk in enumerate(chunks):
        pos = full_text.find(chunk)
        positioned.append((pos, i, chunk))
    # 找不到位置的（pos < 0）排最后，并按原有顺序保持稳定
    positioned.sort(key=lambda item: (item[0] < 0, item[0] if item[0] >= 0 else item[1]))
    return [chunk for _, _, chunk in positioned]


def retrieve_and_build_context(question, full_text, collection, use_vector, top_k=None):
    """检索，并按「块边界」装配参考背景。

    旧实现是 context[:MAX_CONTEXT_LENGTH] 硬切，会把某个 chunk 拦腰截断：答案正好
    落在被切掉的那半截时，模型看到的是一句断话。现在两步走：先按相关度决定要哪些块
    （预算优先给更相关的），再把选中的块按原文顺序拼起来 —— 宁可比预算短，也不给半截块。

    返回 (context, used_chunks)。used_chunks 就是 context 里真正用到的那些块、且与
    context 同序；旧实现返回 chunks[:3]，那是一段没有任何调用方使用的残留代码。

    唯一例外：单块本身就超过 MAX_CONTEXT_LENGTH（说明配置互相矛盾，例如把
    MAX_CONTEXT_LENGTH 调得比一个 chunk 还小）。此时只能截断这一块 —— 总比返回空好 ——
    并记一条 WARNING。
    """
    chunks = hyde_retrieve(question, collection, full_text, use_vector, top_k=top_k)
    if not chunks:
        return "", []

    used, total = [], 0
    for chunk in chunks:                              # 相关度序：预算先给更相关的块
        addition = len(chunk) + (2 if used else 0)    # 块之间是 "\n\n"
        if used and total + addition > MAX_CONTEXT_LENGTH:
            break
        used.append(chunk)
        total += addition

    if total > MAX_CONTEXT_LENGTH:                    # 只有「第一块自己就超预算」会走到这
        logger.warning("单块长度已超过 MAX_CONTEXT_LENGTH=%d，只能截断第一块；"
                       "请调大 MAX_CONTEXT_LENGTH 或调小 CHUNK_SIZE", MAX_CONTEXT_LENGTH)
        return used[0][:MAX_CONTEXT_LENGTH] + "\n...(截断)", used

    if len(used) < len(chunks):
        logger.info("context 按块边界装配：保留 %d/%d 块（%d 字符，预算 %d）",
                    len(used), len(chunks), total, MAX_CONTEXT_LENGTH)

    ordered = _order_by_position(used, full_text)
    return "\n\n".join(ordered), ordered



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
