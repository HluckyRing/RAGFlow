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

# 「枚举 / 汇总」措辞在 `src/gates/`（默认 zh，`GATE_PACK` 选包）：宁可漏判（少补一次位，
# 行为退回今天）也不要误判（多花一次查询）—— 但它本身不改变答案正确性。
import src.config as _config
from src.gates import load_pack as _load_pack

_gates = _load_pack(_config.GATE_PACK, logger=_config.logger)
ENUMERATION_MARKERS = _gates.enumeration_markers


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
NAVIGATION_MARKERS = _gates.navigation_markers
# 「第1章 | 第2章 | …」这种把章节名排成一张表的页面：同一块里出现 ≥5 个「第N章」几乎只有导图页
CHAPTER_LIST_MIN = 5
_CHAPTER_RE = re.compile(r"第\s*[0-9一二三四五六七八九十]{1,3}\s*章")


def is_navigational(text, structural=None):
    """这一块是不是导航 / 行政页（目录、知识导图、作业编号页）。

    三条判据，按可靠性排：
    1. **跨文件近重复**（`structural` 参数，见 `structural_texts()`）—— 语言无关，
       这是主力判据；
    2. 命中 `NAVIGATION_MARKERS` 措辞表 —— 换语言就失效，只当兜底；
    3. 同一块里把章节名排成表（≥5 个「第N章」）—— 只对中文教材式 PPT 成立。

    只用于**排序**，不用于删除 —— 见本节开头那段的说明。
    """
    body = str(text or "")
    if not body.strip():
        return False
    if structural and _text_key(body) in structural:
        return True
    if any(marker in body for marker in NAVIGATION_MARKERS):
        return True
    return len(_CHAPTER_RE.findall(body)) >= CHAPTER_LIST_MIN


def prefer_content(hits):
    """稳定重排：内容页在前、导航页在后（各自保持原有相对顺序）。

    为什么是「稳定分区」而不是打分排序：块之间原本的名次是检索器给的，唯一该动的信息是
    「这块是不是结构页」。多打一层分就会引入新的口径，评测与线上都要跟着重校。
    """
    items = list(hits or [])
    structural = structural_texts(items)      # 先算「跨文件近重复」这一条语言无关的判据
    content, navigation = [], []
    for item in items:
        (navigation if is_navigational(item[0], structural) else content).append(item)
    return content + navigation


# ── 同一份文件内部跨节取样（2026-11-12 第十六轮）──
#
# 真实需求来自真实日志：「帮我整理第一章的内容」。这句话要的不是「跨文件的代表性」，
# 而是**把那一章（同一份文件里的 64 页）铺开讲**。文件级取样对它完全无效 —— 实测两臂都拒答，
# 因为第 1 章的正文块在检索结果里只剩几个，其余名额被别的文件占满。
#
# 两个关键设计：
# 1. **先定位是哪一份文件**：只认问句里的章号（「第一章」「第2章」），再拿章号去文件名里找。
#    不做「问问句像不像某个文件名」那种模糊匹配 —— 匹配错了会答成另一章，比不取样更糟。
# 2. **按节均匀铺开**，不是「取最相关的 k 块」：整理 / 摘要要覆盖整章，按相关度取必然抓头几条。
#    又因为装配阶段还有 `MAX_CONTEXT_LENGTH` 的字数上限（约 12 块），取样块数按**装配预算**
#    定而不是按检索预算定 —— 取 39 块再被截成 14 块，等于只覆盖了前四分之一章。

_CHINESE_NUMBERS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
                    "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14,
                    "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20}
_CHAPTER_MENTION_RE = re.compile(r"第\s*([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*章")
_SECTION_PAGE_RE = re.compile(r"第\s*([0-9]{1,4})\s*页")


def chapter_numbers(question):
    """问句里提到的章号（「第一章」→ 1、「第2章」→ 2）。没提到就是空集合。"""
    numbers = set()
    for raw in _CHAPTER_MENTION_RE.findall(str(question or "")):
        if raw.isdigit():
            numbers.add(int(raw))
        else:
            value = _CHINESE_NUMBERS.get(raw)
            if value:
                numbers.add(value)
    return numbers


def pick_target_file(question, file_names):
    """问句指向哪一份文件：用**章号**去文件名里找。

    只认章号这一条硬信号，返回唯一匹配；找不到或有歧义都返回 `None`
    —— 定位错了会答成另一章，比不取样更糟。
    """
    numbers = chapter_numbers(question)
    if not numbers:
        return None
    matched = []
    for name in file_names or []:
        for number in numbers:
            if any(f"第{number}{sep}章" in name for sep in ("", " ")) or \
                    any(f"第{sep}{number}章" in name for sep in (" ",)):
                matched.append(name)
                break
    return matched[0] if len(matched) == 1 else None


def section_order(meta):
    """排序键：能解析出「第 N 页」就按页号，否则退回 chunk_index（保证稳定）。"""
    meta = meta or {}
    match = _SECTION_PAGE_RE.search(str(meta.get("section") or ""))
    if match:
        return (0, int(match.group(1)), int(meta.get("chunk_index") or 0))
    return (1, int(meta.get("chunk_index") or 0), 0)


def sample_evenly(hits, blocks):
    """在**同一份文件内部**按节均匀取样：`blocks` 块铺满整份文件的节区间。

    为什么均匀而不是按相似度：整理 / 摘要类需求要覆盖整章；按相关度取必然抓头几条、
    尾部整段丢掉。返回顺序按节号升序（装配进 context 也是这个顺序）。
    """
    ordered = sorted(hits or [], key=lambda item: section_order(item[1]))
    if not blocks or blocks <= 0 or len(ordered) <= blocks:
        return ordered
    step = len(ordered) / float(blocks)
    picked, seen = [], set()
    for index in range(blocks):
        item = ordered[int(index * step)]
        if id(item) in seen:
            continue
        seen.add(id(item))
        picked.append(item)
    return picked


# 每块进 context 时还要带一行「【来源：文件名 · 第 N 页】」，估算字数时得把它算进去。
SOURCE_LABEL_OVERHEAD = 40


def blocks_that_fit(hits, max_blocks, max_chars=8000):
    """按**字符预算**反推能取几块：短页多取、长块少取。

    为什么不能写死块数：同一套代码在 PPT（每页 ~140 字）与 md（每块 480 字）上，
    「12 块」分别是 1.7k 与 5.8k 字符 —— 写死要么浪费预算、要么被装配截断后只覆盖前半段。
    """
    items = list(hits or [])
    if not items:
        return 0
    average = sum(len(text or "") for text, _meta in items) / float(len(items))
    per_block = average + SOURCE_LABEL_OVERHEAD
    if per_block <= 0:
        return max(1, int(max_blocks or 1))
    fits = int(max_chars // per_block)
    return max(1, min(int(max_blocks or 1), fits))


# ── 语言无关的「结构页」检测（2026-11-12 第十七轮）──
#
# 措辞表换语言就失效，但**结构页有多硬的语言无关特征**：同一段结构说明（目录、知识导图、
# 章节清单）在不同文件里几乎逐字重复。中文语料里「第N章知识导图」在 12 份 PPT 里各有一页，
# 英文语料里同样会是「Table of contents / Chapter 1 …」各有一页。
#
# 做法：字符 4-gram 的 Jaccard 相似度 ≥ 0.6 视为「近重复」，且这份近重复要**跨 ≥3 份文件**
# 才算结构页 —— 两个人引用同一句话是内容，每份文件都抄一遍同一张目录才是结构。
# 只在**候选块之间**比较（≤ 40 块，O(n²) 也在毫秒级），不需要额外检索。
_STRUCTURAL_SHINGLE = 4
_STRUCTURAL_JACCARD = 0.6
_STRUCTURAL_MIN_FILES = 3


def _shingles(text):
    body = re.sub(r"\s+", "", str(text or ""))
    if len(body) < _STRUCTURAL_SHINGLE:
        return {body} if body else set()
    return {body[i:i + _STRUCTURAL_SHINGLE]
            for i in range(len(body) - _STRUCTURAL_SHINGLE + 1)}


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _text_key(text):
    return re.sub(r"\s+", "", str(text or ""))


def structural_texts(hits, threshold=_STRUCTURAL_JACCARD, min_files=_STRUCTURAL_MIN_FILES):
    """挑出「跨 ≥`min_files` 份文件近重复」的块文本 —— 语言无关的结构页判据。

    返回**归一化文本**的集合（`prefer_content` 直接拿它比对）。
    """
    items = list(hits or [])
    if len(items) < min_files:
        return set()
    shingle_cache = [_shingles(text) for text, _meta in items]
    flagged = set()
    for index, (text, meta) in enumerate(items):
        name = (meta or {}).get("file_name")
        neighbours = set()
        for other_index, (_other_text, other_meta) in enumerate(items):
            if other_index == index:
                continue
            other_name = (other_meta or {}).get("file_name")
            if not other_name or other_name == name:
                continue                    # 同一份文件内重复不算（同一章的页眉页脚会重复）
            if _jaccard(shingle_cache[index], shingle_cache[other_index]) >= threshold:
                neighbours.add(other_name)
        if len(neighbours) + 1 >= min_files:
            flagged.add(_text_key(text))
    return flagged
