# -*- coding: utf-8 -*-
"""中文文本的轻量处理：清掉脏字符、剥离疑问词、抽取检索/指代用的实义词。

不引入分词依赖 —— 中文按 2/3-gram 切分。检索（retrieval）和指代消解（llm）
都需要同一套「哪些词才算有意义」的判断，集中在这里，避免两处逻辑再次各自跑偏。
"""
import re

from src.config import logger

# 孤立代理字符（U+D800–U+DFFF）。字体映射损坏的 PDF 经 pypdf 解析后会产出它们：
# - 进 embedding：sentence-transformers 的 tokenizer 抛
#   TypeError: TextEncodeInput must be Union[...]
# - 进 state：json.dump(ensure_ascii=False) 抛 UnicodeEncodeError（surrogates not allowed）
# 已实测：lxml 会直接拒绝代理字符（XMLSyntaxError），所以 docx/xlsx/pptx 带不进这种脏东西，
# txt/csv/md 的解码回退链也产生不了；实际来源就是 pypdf。
SURROGATE_RE = re.compile("[\ud800-\udfff]")

# 剥离用的疑问/提问短语，按长度从长到短依次替换
QUESTION_PHRASES = (
    "是什么意思", "是什么", "是多少", "有哪些", "有什么", "怎么样",
    "什么是", "什么叫", "请问", "如何", "怎么", "怎样", "为什么", "为何",
    "哪个", "哪些", "哪种", "简述", "简要", "介绍", "说明", "解释", "列举",
    "一下", "多少", "几个", "吗", "呢",
)

# 参与 n-gram 过滤的停用词。
# 只收多字词：2/3-gram 不可能等于单个汉字，所以旧代码里那一大堆单字停用词
# 对 n-gram 打分毫无作用（这也正是原实现停用词表形同虚设的原因之一）。
CONTENT_STOPWORDS = {
    # 疑问词（多数已在 strip_question_words 阶段被剥掉，这里再兜一层）
    "什么", "为什么", "为何", "怎么", "怎样", "如何", "哪个", "哪些", "哪种",
    "是否", "多少", "几个", "怎么样",
    # 指示代词与人称代词
    "这个", "那个", "这些", "那些", "它", "它们", "他", "他们", "她", "她们",
    "自己", "大家", "咱们", "我们", "你们", "它的", "他的", "她的", "他们的",
    "上述", "下列", "下面", "前面", "以下",
    # 虚词与连接词
    "一个", "一下", "可以", "以及", "或者", "但是", "因为", "所以", "如果",
    "那么", "就是", "不是", "没有", "关于", "对于", "根据", "通过", "进行",
    # 提问语境词
    "相关", "内容", "问题", "意思", "含义", "介绍", "说明", "解释", "简述",
    "列举", "请问", "举例",
}


def strip_surrogates(text):
    """剥掉孤立代理字符，正常文本原样返回。

    「文本进系统」的第一道清洗：pypdf 遇到字体映射损坏的 PDF 会产出代理字符，
    它既能弄炸 embedding，也能弄炸 state 的 JSON 落盘（见上方注释）。
    按仓库规矩不静默丢字符 —— 剥掉了几个会记一条 WARNING。
    """
    if not text or SURROGATE_RE.search(text) is None:
        return text
    cleaned, count = SURROGATE_RE.subn("", text)
    logger.warning("文本含 %d 个孤立代理字符（常见于字体映射损坏的 PDF），已剥离", count)
    return cleaned


def strip_question_words(text):
    """把「什么是/如何/请问…」这类提问短语替换成空格，留下实义片段。"""
    out = text
    for phrase in sorted(QUESTION_PHRASES, key=len, reverse=True):
        out = out.replace(phrase, " ")
    return out


def content_phrases(text):
    """去掉疑问词后剩下的实义片段（中文连续串 / 英文单词），用于指代消解。"""
    stripped = strip_question_words(text)
    phrases = [p for p in re.findall(r'[\u4e00-\u9fa5]{2,}', stripped)
               if p not in CONTENT_STOPWORDS]
    phrases += [w for w in re.findall(r'[A-Za-z][A-Za-z0-9]{1,}', stripped)
                if w.lower() not in CONTENT_STOPWORDS]
    return phrases


def extract_terms(text, ngram=(2, 3)):
    """抽出用于关键词打分的词：中文 2/3-gram + 英文单词。

    用 n-gram 而不是整串匹配，是为了让「净资产收益率」这种提问能命中只写了
    「净资产」或「收益率」的段落 —— 原实现把整句话当一个词，几乎永远命中不了。
    """
    terms = set()
    stripped = strip_question_words(text)
    for run in re.findall(r'[\u4e00-\u9fa5]+', stripped):
        if run in CONTENT_STOPWORDS:
            continue
        for n in ngram:
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                if gram not in CONTENT_STOPWORDS:
                    terms.add(gram)
    for word in re.findall(r'[A-Za-z][A-Za-z0-9]{1,}', stripped):
        low = word.lower()
        if low not in CONTENT_STOPWORDS:
            terms.add(low)
    return terms
