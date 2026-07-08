import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def load_local_knowledge():
    try:
        with open("test.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "本地知识库为空，请检查 test.txt 文件。"

def simple_rag_qa(user_question):
    context = load_local_knowledge()
    if len(context) > 500:
        context = context[:500]
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role":"system","content":"根据【参考背景】回答问题，背景没有就说不知道。"},
            {"role": "user", "content": f"【参考背景】：{context}\n\n用户问题：{user_question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("📖 本地知识库机器人已启动！")
    question = input("请输入你的问题：")
    answer = simple_rag_qa(question)
    print(f"🤖 回答：{answer}")