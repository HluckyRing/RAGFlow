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
    """
    names = [str(name) for name in (file_names or []) if name]
    if not names:
        return None
    if not chapter_numbers(question):
        return None
    target = pick_target_file(question, names)
    if not target:
        return None
    stripped = strip_scope(question)
    probe = stripped or str(question or "").strip()
    if not probe:
        return None
    return {"file_name": target, "question": probe}
