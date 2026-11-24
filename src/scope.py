# -*- coding: utf-8 -*-
"""范围词 → 文件过滤器 + 检索探针改写（2026-11-19 第十九轮）。

问题（真实对话实测，逐块复现过）
--------------------------------
用户问「**结合第2章课件**，在同一训练集上不断提高多项式回归的阶数，通常可以观察到什么」，
系统只捞到 1 块（第1章 p46「回归问题的工作框架」），于是老实拒答「材料里没有第2章的内容」；
把前缀删掉的同一条问题却能捞到 3–4 块（第2章 p91 过拟合页 + 第1章 p72/p73）并答对。

根因不是阈值也不是语料，而是两件事叠在一起：

1. **「第2章课件」被当成检索词**。这类词属于**文档元信息层**，向量空间里离它最近的恰好是
   封面页、内容提要页、目录页 —— 它们把真正讲答案的那页从第 8 名挤到第 16 名；
2. 双路融合的 `intersect` 闸门要求**两路都召回**，于是交集塌成 1 块（HyDE 侧第 1 名都没用）。

人的做法是「先定位到第 2 章那份文件，再在里面找」—— 这个模块就是这一步：把范围词
从**检索词**变成**过滤器**，同时把范围词从探针文本里删掉。

与 `coverage.pick_target_file` 的关系
------------------------------------
定位逻辑**一份实现**：复用 `coverage.chapter_numbers`（「第一章」/「第2章」→ 1/2）与
`coverage.pick_target_file`（章号 → 文件名，唯一匹配才算）。那边给「整章枚举」用，
这边给「单问的事实提问」用 —— 判据必须同一份，否则同一个问句在两条路上定位到不同文件。

三条纪律
--------
- **认不准就不动手**：没有章号、或章号匹配到多份文件（「第1章和第2章有什么区别」）一律返回
  `None`，行为退回改造前 —— 过滤错了会把答案锁死在错的文件里，比不过滤更糟；
- **只删范围短语，不删信息需求**：删完必须还剩下一个像问题的句子（≥ `MIN_QUESTION_CHARS`），
  否则不动；
- **纯函数、零依赖**：不碰 Chroma、不花 API、不读环境变量（开关在 `config` 里，由调用方判断）。
"""
import re

from src.coverage import chapter_numbers, pick_target_file

# 范围短语的引导词：「结合第2章课件」「根据第2章的讲义」「参考第3章PPT」
_LEAD_WORDS = r"(?:结合|根据|参考|按照|依据|基于)"
_POLITENESS = r"(?:请|请你|麻烦你?|帮我)?"
_NUMBER = r"([0-9]{1,2}|[一二三四五六七八九十]{1,3})"
# 「…课件 / 讲义 / PPT / 幻灯片 / 材料 / 教材」这些词本身没有信息量，只是说明「去哪儿找」
_MATERIAL_WORDS = r"(?:课件|讲义|ppt|PPT|Ppt|幻灯片|材料|教材|教案)"
_TAIL = r"[，,、；;：:。\s]*"

# ① 完整形态：「结合第2章课件，」/「请根据第2章的讲义：」
_FULL_SCOPE_RE = re.compile(
    _POLITENESS + r"\s*" + _LEAD_WORDS + r"\s*第\s*" + _NUMBER + r"\s*章\s*(?:的)?\s*"
    + _MATERIAL_WORDS + r"?\s*" + _TAIL
)
# ② 简写形态：「第2章课件里…」/「根据第2章的课件」
_MATERIAL_SCOPE_RE = re.compile(
    _POLITENESS + r"\s*(?:" + _LEAD_WORDS + r"\s*)?第\s*" + _NUMBER
    + r"\s*章\s*(?:的)?\s*" + _MATERIAL_WORDS + r"\s*" + _TAIL
)

# 删完至少还要剩这么多字，否则算「整句都是范围词」，不删
MIN_QUESTION_CHARS = 4


def strip_scope(question):
    """删掉问句里的范围短语，返回**剩下的信息需求**；删不掉或删完太短时返回 `None`。

    只删**一处**（问句里出现两处范围词本身就是「认不准」，交给 `scope_target` 返回 `None`）。
    """
    text = str(question or "").strip()
    if not text:
        return None
    for pattern in (_FULL_SCOPE_RE, _MATERIAL_SCOPE_RE):
        match = pattern.search(text)
        if not match:
            continue
        rest = (text[:match.start()] + " " + text[match.end():]).strip(_TAIL + " ")
        rest = re.sub(r"\s{2,}", " ", rest).strip()
        if len(rest) < MIN_QUESTION_CHARS:
            return None
        return rest
    return None


def scope_target(question, file_names):
    """问句指向哪一份文件，以及**改写后的检索探针**。

    返回 `{"file_name": 文件名, "question": 改写后的问句}`；认不准时返回 `None`（不动手）。

    `question` 可能等于原问句（问句里有章号但没有「课件/讲义」这类范围短语，
    例如「第2章讲了什么」）—— 那时**仍然过滤**（人要的就是这一章），只是没有可删的词。

    两条定位路径（2026-11-23 加第二条）：

    1. **章号**（原有）：`chapter_numbers` + `pick_target_file`，唯一匹配才算；
    2. **标题词**（新）：问句里有「指代 + 材料」标记（「那份 PPT」「那个课件」），且问句里出现了某份
       文件标题的前缀（≥3 字，例如「k近邻那份材料」↔`k近邻算法`），且**只命中一份**文件。

    「这份材料」「上面那个实验」这类**不做**：前者要「上一轮 sources / 最近上传顺序」，后者指的是话题
    而不是文件 —— 没有依据就猜，会把答案锁死在错的文件里（见 `.agents/scope-deixis-spec.md`）。
    """
    names = [str(name) for name in (file_names or []) if name]
    if not names:
        return None
    if chapter_numbers(question):
        target = pick_target_file(question, names)
        if not target:
            return None
        stripped = strip_scope(question)
        probe = stripped or str(question or "").strip()
        if not probe:
            return None
        return {"file_name": target, "question": probe}

    target = name_target(question, names)
    if not target:
        return None
    probe = strip_deictic(question) or str(question or "").strip()
    if not probe:
        return None
    return {"file_name": target, "question": probe}


# ── 标题词定位（2026-11-23 第二十一轮，待办 ④）──
#
# 真实需求：「操作系统引论那个课件里，进程调度有哪些算法」—— 没有章号，但**点了文件标题**。
# 这类提问原来一律不定位（`scope_target` 只认章号），结果过滤这条能力对它们完全失效。
#
# 为什么要求「指代 + 材料」标记：光有标题词不够保守 —— 「决策树和随机森林有什么区别」
# 里的「决策树」可能是在问通用概念，锁到某一份文件会把跨文件的答案掐掉。
# 有「那份 PPT / 那个课件」这类标记，才说明用户是在**指一份材料**。
_DEICTIC = r"(?:这份|那份|这个|那个|该|本|刚才上传的?|上面|前面)"
_MATERIAL = r"(?:文件|材料|课件|讲义|幻灯片|文档|pdf|PDF|Ppt|ppt|PPT)"
_DEICTIC_MATERIAL_RE = re.compile(_DEICTIC + r"\s*" + _MATERIAL)
# 探针改写：删掉「那份 PPT 里的」「那个课件里」这类包装词（标题词保留 —— 它是信息）
_DEICTIC_WRAP_RE = re.compile(
    _DEICTIC + r"\s*" + _MATERIAL + r"\s*(?:里|中|内)?\s*(?:的)?\s*")
_EXT_RE = re.compile(r"\.(?:pdf|pptx?|docx?|md|markdown|xlsx?|txt|csv)$", re.I)
_CHAPTER_PREFIX_RE = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十]{1,3}\s*章\s*[-—_:：、]?\s*")
_NAME_SEP_RE = re.compile(r"[\s\-—–_·、,，.。:：;；()（）\[\]【】《》\"'“”‘’/\\|]+")
NAME_PREFIX_MIN = 3          # 标题前缀最少几个字符（「决策」「回归」这种 2 字太泛）


def _normalize(text):
    """比对用：去掉所有空白并转小写（「K 近邻」和「k近邻」要能对上）。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def title_core(file_name):
    """文件名 → 标题核心：去扩展名、去「第N章」前缀、去分隔符。"""
    name = _EXT_RE.sub("", str(file_name or "").strip())
    name = _CHAPTER_PREFIX_RE.sub("", name)
    return _normalize(_NAME_SEP_RE.sub("", name))


def title_match_length(question, file_name, min_prefix=NAME_PREFIX_MIN):
    """问句命中了这份文件标题的**多长**前缀（0 = 没命中；越长越可信）。"""
    core = title_core(file_name)
    if len(core) < min_prefix:
        return 0
    probe = _normalize(question)
    for length in range(len(core), min_prefix - 1, -1):
        if core[:length] in probe:
            return length
    return 0


def title_matches(question, file_name, min_prefix=NAME_PREFIX_MIN):
    """问句里有没有这份文件标题的前缀（≥`min_prefix` 个字）。"""
    return title_match_length(question, file_name, min_prefix) > 0


def name_target(question, file_names):
    """按标题词定位：问句里有「指代 + 材料」标记，且**证据串在语料里唯一**时返回它。

    两级判唯一（2026-11-23 实测出来的第二级）：
    1. **最长命中优先**：问「虚拟存储器那份材料」时，`存储器管理` 也能靠前缀「存储器」命中，
       但 `虚拟存储器` 命中得更长 —— 先取命中最长的那些；
    2. **证据串还要在语料里唯一**：问「存储器那份材料」时最长命中是「存储器」（`存储器管理` 的前缀），
       可这个串同时出现在 `虚拟存储器`、`磁盘存储器管理` 的标题里 —— 用户到底指哪份说不清，
       **认不准就不动手**（只靠第 1 级会把答案锁死在 `存储器管理` 上，这是实测抓到的误定位）。
    """
    text = str(question or "")
    if not _DEICTIC_MATERIAL_RE.search(text):
        return None
    names = [name for name in (file_names or []) if name]
    scored = [(title_match_length(text, name), name) for name in names]
    best = max((length for length, _name in scored), default=0)
    if best <= 0:
        return None
    champions = [name for length, name in scored if length == best]
    if len(champions) != 1:
        return None
    winner = champions[0]
    evidence = title_core(winner)[:best]
    sharing = [name for name in names if evidence and evidence in title_core(name)]
    return winner if len(sharing) == 1 else None


def strip_deictic(question):
    """把「那份 PPT 里的」这类包装词从探针里删掉；删不掉或删完太短就返回 `None`（不动）。"""
    text = str(question or "").strip()
    if not text:
        return None
    match = _DEICTIC_WRAP_RE.search(text)
    if not match:
        return None
    rest = (text[:match.start()] + " " + text[match.end():]).strip(" \t，,、；;：:。")
    rest = re.sub(r"\s{2,}", " ", rest).strip()
    if rest == text or len(rest) < MIN_QUESTION_CHARS:
        return None
    return rest
