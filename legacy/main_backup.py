# -*- coding: utf-8 -*-
import os
import ssl
import re
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# 解决 SSL 问题
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 加载环境变量
load_dotenv()
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


# ------------------- 1. 读取 PDF -------------------
def load_pdf_text():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None, "错误：没有找到 data 文件夹"

    files = os.listdir(data_dir)
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None, "错误：data 文件夹里没有 PDF 文件"

    try:
        reader = PdfReader(os.path.join(data_dir, pdf_files[0]))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return None, "错误：PDF 内容为空或扫描版（无法提取文字）"

        return text, None
    except Exception as e:
        return None, f"错误：读取 PDF 失败 - {e}"


# ------------------- 2. 文本切分（纯 Python，不依赖任何库） -------------------
def split_text(text, chunk_size=500, overlap=50):
    """将长文本切分成小块，每块约 chunk_size 个字符"""
    # 按标点符号切分句子
    sentences = re.split(r'(?<=[。！？；\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sent in sentences:
        # 如果当前块加上新句子不超过 chunk_size，就加上
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            # 否则保存当前块，开始新块（带 overlap）
            if current_chunk:
                chunks.append(current_chunk)
            # overlap：保留前一个块的最后 overlap 个字符
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + sent
            else:
                current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ------------------- 3. 检索：关键词匹配（找出最相关的 2 块） -------------------
def retrieve_relevant_chunks(question, chunks):
    """用关键词匹配找出最相关的 2 块"""
    # 提取问题中的关键词（去掉停用词）
    stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到",
                 "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "它", "他", "她", "们",
                 "与", "或", "等", "但", "而", "因", "为", "对", "从", "把", "被", "让", "给", "跟", "比", "更", "最",
                 "太", "非常", "十分", "特别", "相当", "比较", "挺", "蛮", "可", "以", "能", "够", "得", "地", "也"}

    # 提取中文词（2 个字以上）
    question_words = re.findall(r'[\u4e00-\u9fa5]{2,}', question)
    # 也提取英文单词
    question_words += re.findall(r'[a-zA-Z]{2,}', question.lower())
    # 过滤停用词
    question_words = [w for w in question_words if w not in stopwords]

    if not question_words:
        # 如果没有关键词，返回前 2 块
        return chunks[:2]

    # 计算每块得分
    scored = []
    for chunk in chunks:
        score = 0
        chunk_lower = chunk.lower()
        for word in question_words:
            score += chunk_lower.count(word.lower())
        if score > 0:
            scored.append((score, chunk))

    # 按得分从高到低排序，取前 2 块
    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in scored[:2]]

    if not top_chunks:
        # 如果没有任何匹配，返回前 2 块
        return chunks[:2]

    return top_chunks


# ------------------- 4. 问答函数 -------------------
def answer_question(question, full_text):
    # 切分文本
    chunks = split_text(full_text)
    print(f"📄 文本已切分成 {len(chunks)} 块")

    # 检索相关块
    relevant = retrieve_relevant_chunks(question, chunks)
    context = "\n---\n".join(relevant)

    # 调用 DeepSeek
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "deepseek-v4-flash"),
        messages=[
            {"role": "system",
             "content": "请严格根据【参考背景】回答问题。如果背景中没有相关信息，请直接说'背景中没有提到相关内容'，不要编造。"},
            {"role": "user", "content": f"【参考背景】：\n{context}\n\n用户问题：{question}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# ------------------- 5. 主程序 -------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 本地 PDF 知识库问答系统（关键词检索版）")
    print("=" * 50)

    # 加载 PDF
    print("\n📂 正在读取 PDF...")
    full_text, error = load_pdf_text()

    if error:
        print(f"❌ {error}")
        input("\n按回车键退出...")
        exit()

    print(f"✅ 成功读取 PDF，共 {len(full_text)} 个字符")

    # 预切分文本
    chunks = split_text(full_text)
    print(f"📄 文本已切分成 {len(chunks)} 块")

    print("\n" + "=" * 50)
    print("🎉 知识库加载完成！输入问题开始问答")
    print("💡 提示：输入 'exit' 或 'quit' 退出程序")
    print("=" * 50)

    while True:
        try:
            question = input("\n💬 请输入你的问题：").strip()

            if question.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break

            if not question:
                print("⚠️ 问题不能为空，请重新输入")
                continue

            print("🤔 正在检索并生成回答...")
            answer = answer_question(question, full_text)
            print(f"\n🤖 回答：\n{answer}")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出错了：{e}")
            import traceback

            traceback.print_exc()

    input("\n按回车键退出...")