# -*- coding: utf-8 -*-
"""跨文件取样：让「枚举 / 汇总型」提问不要再静默漏掉整份文件。

问题（2026-11-12 第十二轮实测）
------------------------------
真实用户问「列出所有的实验」「该知识库的主要内容是什么」时，检索按**与问句的相关度**排序、
再按 `MAX_DISTANCE` 砍，于是**相关度稍低的那份文件整份消失**：13 份文件的语料里，装配后的
context 只覆盖 12 份，用户和模型都不知道少了一份。这类提问要的不是「最相关的 k 块」，
而是**跨全部文件的代表性取样**。

三层职责，都在这个模块里，都是纯函数（不碰 Chroma、不花 API）：
1. `looks_enumerative()` —— 识别「枚举 / 汇总」措辞。**这是一张措辞表，永远是脆的**，
   所以它只决定「要不要多花一次补位查询」，绝不决定「答什么、算不算对」；
2. `sample_per_file()` —— 把已有命中按文件轮流取样：先保证每份文件各 1 块，再轮到第 2 块；
3. `missing_files()` / `coverage_report()` —— 数清楚还差哪几份（补位与告警都靠它）。

为什么「按文件轮流」而不是「再按相似度排一遍」：枚举型提问的每个文件都同等重要，
用相似度排序必然把「问句字面更像」的那几份排在前面，把边缘但必要的那几份挤掉。
"""
from collections import OrderedDict
import re

# 「枚举 / 汇总」措辞。宁可漏判（少补一次位，行为退回今天）也不要误判（多花一次查询）——
# 但它本身不改变答案正确性，所以误判的代价只是一次检索。
ENUMERATION_MARKERS = (
    "列出", "列举", "所有", "全部", "各", "分别", "几个", "哪些", "多少个",
    "总结", "汇总", "整理", "归纳", "概述", "主要内容", "讲了什么",
)


def looks_enumerative(question, markers=ENUMERATION_MARKERS):
    """这条提问是不是「枚举 / 汇总型」。只用于「要不要做跨文件取样」。"""
    text = str(question or "")
    return any(marker in text for marker in markers)


def file_names_of(hits):
    """命中块涉及的文件名（去重、保持首次出现顺序）。没有来源的块不计。"""
    names = OrderedDict()
    for _text, meta in hits or []:
        name = (meta or {}).get("file_name")
        if name:
            names.setdefault(name, None)
    return list(names)


def missing_files(hits, file_names):
    """`file_names` 里哪些文件在 `hits` 里一块都没有。"""
    seen = set(file_names_of(hits))
    return [name for name in file_names or [] if name and name not in seen]


def sample_per_file(hits, budget):
    """按文件轮流取样：每份文件先各取 1 块，再轮流取第 2 块……直到 `budget` 块。

    - 顺序以**文件首次出现的先后**为准（也就是相似度最高的那份先取第 1 块），
      文件之间是公平的，但文件内部仍按相似度；
    - 没有来源（`meta` 为 None）的块归到 `None` 这一组，按原顺序排在最后参与轮转
      —— 老索引没有 metadatas 时不能让这些块整批消失。
    """
    buckets = OrderedDict()
    for text, meta in hits or []:
        name = (meta or {}).get("file_name")
        buckets.setdefault(name, []).append((text, meta))
    if not buckets:
        return []

    picked = []
    round_index = 0
    while len(picked) < max(0, budget or 0):
        added = False
        for bucket in buckets.values():
            if round_index < len(bucket):
                picked.append(bucket[round_index])
                added = True
                if len(picked) >= budget:
                    break
        if not added:
            break
        round_index += 1
    return picked


def coverage_report(hits, file_names):
    """覆盖盘点：`{"covered", "total", "missing", "blocks"}`。"""
    names = file_names_of(hits)
    total = [name for name in (file_names or []) if name]
    return {
        "covered": len([name for name in total if name in set(names)]),
        "total": len(total),
        "missing": missing_files(hits, total),
        "blocks": len(hits or []),
    }


# ── 导航页 / 行政页（2026-11-12 第十四轮实测出来的真病根）──
#
# 枚举 / 汇总型提问检索到的头部块里，绝大多数是**结构页而不是内容页**：
#   · 「第N章知识导图 第1章 | 操作系统引论 第2章 | 进程的描述与控制 …」——每章都有一页，
#     跟任何「讲的是什么 / 主要内容」的提问都极像，于是把正文页全挤掉；
#   · 「内容导航： 1.1 … 1.2 …」——目录页；
#   · 「第一次作业 1 简答题 2 3 4 … 标黄色为本次作业」——作业编号页，一个字的信息都没有。
# 模型拿到这些只能复述「材料只给了章节名」，于是「覆盖变全」并不等于「答得更全」。
#
# 这里刻意**只降权不删除**：小语料里那页可能就是唯一的答案（例如「有哪些题型」的题型清单
# 就长在提纲页上），删干净会让原本能答的变成答不出。顺序永远是「内容页在前、导航页在后」。
NAVIGATION_MARKERS = (
    "知识导图", "内容导航", "第一次作业", "第二次作业", "第三次作业", "第四次作业",
    "标黄色为本次作业", "学习进步", "学而时习之", "课程思政小讲堂",
)
# 「第1章 | 第2章 | …」这种把章节名排成一张表的页面：同一块里出现 ≥5 个「第N章」几乎只有导图页
CHAPTER_LIST_MIN = 5
_CHAPTER_RE = re.compile(r"第\s*[0-9一二三四五六七八九十]{1,3}\s*章")


def is_navigational(text):
    """这一块是不是导航 / 行政页（目录、知识导图、作业编号页）。

    两条判据：命中 `NAVIGATION_MARKERS`，或同一块里把章节名排成表（≥5 个「第N章」）。
    只用于**排序**，不用于删除 —— 见本节开头那段的说明。
    """
    body = str(text or "")
    if not body.strip():
        return False
    if any(marker in body for marker in NAVIGATION_MARKERS):
        return True
    return len(_CHAPTER_RE.findall(body)) >= CHAPTER_LIST_MIN


def prefer_content(hits):
    """稳定重排：内容页在前、导航页在后（各自保持原有相对顺序）。

    为什么是「稳定分区」而不是打分排序：块之间原本的名次是检索器给的，唯一该动的信息是
    「这块是不是结构页」。多打一层分就会引入新的口径，评测与线上都要跟着重校。
    """
    content, navigation = [], []
    for item in hits or []:
        (navigation if is_navigational(item[0]) else content).append(item)
    return content + navigation
