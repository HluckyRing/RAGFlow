import re
from src.config import logger, client, MODEL_NAME, MAX_CONTEXT_LENGTH
from src.prompts import QA_SYSTEM_PROMPT_TEMPLATE
from src.retrieval import hyde_retrieve

STOP_ENTITIES = {
    "什么", "如何", "为什么", "怎样", "哪个", "哪些", "一个", "这个",
    "那个", "这些", "那些", "它们", "他们", "她们", "自己", "大家", "咱们"
}


def _extract_entities(text):
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    return [w for w in words if w not in STOP_ENTITIES]


def get_answer(question, full_text, collection, use_vector, history):
    if len(question) < 15 and any(w in question for w in ["它", "这", "其", "那", "她", "他"]):
        user_msgs = [msg["content"] for msg in history if msg["role"] == "user"]
        if len(user_msgs) >= 2:
            prev_q = user_msgs[-2]
            entities = _extract_entities(prev_q)
            if entities:
                search_query = " ".join(entities) + " " + question
            else:
                search_query = question
        else:
            search_query = question
    else:
        search_query = question

    chunks = hyde_retrieve(search_query, collection, full_text, use_vector)
    if not chunks:
        return "（未找到任何相关内容）", []

    context = "\n\n".join(chunks)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "\n...(截断)"

    source_chunks = chunks[:3]

    system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(context=context)

    history_messages = []
    for msg in history[-5:]:
        if msg["role"] in ["user", "assistant"]:
            history_messages.append(msg)

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
