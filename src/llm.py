import re
from src.config import logger, client, MODEL_NAME, MAX_CONTEXT_LENGTH
from src.prompts import QA_SYSTEM_PROMPT_TEMPLATE
from src.retrieval import hyde_retrieve

STOP_ENTITIES = {
    "什么", "如何", "为什么", "怎样", "哪个", "哪些", "一个", "这个",
    "那个", "这些", "那些", "它们", "他们", "她们", "自己", "大家", "咱们"
}


def extract_entities(text):
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    return [w for w in words if w not in STOP_ENTITIES]


def resolve_query(question, history):
    if len(question) < 15 and any(w in question for w in ["它", "这", "其", "那", "她", "他"]):
        user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
        if len(user_msgs) >= 2:
            prev_q = user_msgs[-2]
            entities = extract_entities(prev_q)
            if entities:
                return " ".join(entities) + " " + question
    return question


def retrieve_and_build_context(question, full_text, collection, use_vector, top_k=None):
    chunks = hyde_retrieve(question, collection, full_text, use_vector, top_k=top_k)
    if not chunks:
        return "", []
    context = "\n\n".join(chunks)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n...(截断)"
    return context, chunks[:3]


def get_answer(question, full_text, collection, use_vector, history):
    search_query = resolve_query(question, history)
    context, source_chunks = retrieve_and_build_context(search_query, full_text, collection, use_vector)
    if not context:
        return "（未找到任何相关内容）", []

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
            temperature=0.2
        )
        answer = response.choices[0].message.content
    except Exception as e:
        logger.error("LLM 调用失败: %s", e)
        answer = f"调用大模型失败：{str(e)}"

    return answer, source_chunks


def stream_answer(question, context, history):
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
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("LLM 流式调用失败: %s", e)
        yield f"\n\n调用大模型失败：{str(e)}"
