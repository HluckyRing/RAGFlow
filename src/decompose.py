# -*- coding: utf-8 -*-
"""拆问（query 分解）：把一个复合问句拆成若干能各自检索的子问题。

为什么要有它（2026-11-08 实测，见 docs/retrieval-baseline-2026-10-06.md）
================================================================================
真实语料上的对照实验说明：`top_k` 不是杠杆（覆盖@5 = 覆盖@10），而**每问分路检索**
在同一份 context 预算下把覆盖从 45.9% 抬到 81.1%。但那个实验里的子问题是 **gold**
（样本本来就由子问题拼出来）—— 线上没有 gold，得先有一步「把复合问句拆开」。

三条纪律（与 `answerability.py` 同源）
--------------------------------------
- **关思考、小预算**：`deepseek-flash` 是推理模型，`max_tokens` 给小了会被 reasoning
  吃光、`content` 返回空串（HyDE 就是这样静默失效过一次）。拆问不需要推理链。
- **只拆出一条时用原问句**：模型给的「同义改写」只会让检索变差，且没有任何收益；
  单问（绝大多数提问）因此完全不受影响。
- **任何故障都不许堵死问答**：缺 API_KEY、调用失败、空内容、解析失败一律退成
  `[question]` + WARNING —— 退回的是「和以前一样」的行为，绝不退成拒答。

成本：一次额外的 LLM 调用（实测 8 条样例 0.4–1.4 s，中位约 0.75 s）。所以入口有一道
**廉价闸门** `looks_compound()`：看着就不是复合问句的直接跳过，一次 API 都不花。
"""
import json
import re

import src.config as config
from src.config import logger
from src.prompts import DECOMPOSE_SYSTEM_PROMPT

# 一次最多拆几问：再多就不是「一个问题」，而是把整段话贴进来了
MAX_PARTS = 6
# 太短的「子问题」多半是模型吐出的标点或序号残渣；太长的多半是它把整句又抄了一遍
MIN_PART_CHARS = 2
MAX_PART_CHARS = 120
MAX_TOKENS = 300

# 拆问用的模型客户端：单独取一次，测试可以替换成替身
client = config.client

# ── 廉价闸门 ──
# 这些连接词出现时，一句里几乎必然有两件事（「年假有几天还有迟到怎么算」）
COMPOUND_MARKERS = ("还有", "以及", "另外", "顺便", "同时", "并且", "再问", "顺便问")
# 中文里的「问」很多只能靠疑问词认出来（逗号连写时没有问号）。
# 「呢」是**省略式追问**的信号（「一线城市限额多少，境外城市呢？」）—— 2026-11-08 实测：
# 不收它就会漏判 2/18 条复合样本，而收它只多 0~1 条单问误判（误判只白花一次调用，
# 漏判是整套功能对那条提问静默失效，所以宁可多问）。
QUESTION_WORDS = ("多少", "几", "怎么", "什么", "多久", "多长", "多大", "哪些", "哪个",
                  "哪里", "何时", "是否", "能不能", "可不可以", "要不要", "如何",
                  "为什么", "谁", "吗", "呢")
_SEGMENT_SPLIT_RE = re.compile(r"[，,；;、]")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+\s*[.、)）]|\(\d+\))\s*")
_QUOTE_PAIRS = (("「", "」"), ("“", "”"), ('"', '"'), ("'", "'"), ("『", "』"))


def _segments(text):
    return [piece for piece in _SEGMENT_SPLIT_RE.split(text) if piece.strip()]


def looks_compound(question):
    """要不要花钱问模型。保守：宁可多问一次，也别把复合问句当成单问放过。

    三条线索任一成立就算「可能是复合问句」：≥2 个问号 / 出现「还有、以及」这类连接词 /
    逗号分隔出的片段里有 ≥2 段带疑问词。**最后一条是防止**「报销超过 1000 元，要附什么？」
    这种「一个条件 + 一个问」被误判成两问（它只有一段带疑问词）。
    """
    text = (question or "").strip()
    if not text:
        return False
    if text.count("？") + text.count("?") >= 2:
        return True
    if any(marker in text for marker in COMPOUND_MARKERS):
        return True
    segments = _segments(text)
    if len(segments) >= 2:
        with_question_word = sum(1 for segment in segments
                                 if any(word in segment for word in QUESTION_WORDS))
        if with_question_word >= 2:
            return True
    return False


def _strip_fence(text):
    """去掉 ```json / ``` 围栏（模型很爱加）。"""
    lines = [line for line in text.splitlines() if not _FENCE_RE.match(line)]
    return "\n".join(lines).strip()


def _clean_part(part):
    """洗干净一条子问题：去序号、去项目符号、去成对引号、压空白。"""
    text = str(part or "").strip()
    text = _BULLET_RE.sub("", text).strip()
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip()


def _json_blobs(text):
    """可能装着数组的几段文本：整段、第一个 `[`…最后一个 `]`、第一个 `{`…最后一个 `}`。"""
    blobs = [text]
    starts, ends = text.find("["), text.rfind("]")
    if 0 <= starts < ends:
        blobs.append(text[starts:ends + 1])
    starts, ends = text.find("{"), text.rfind("}")
    if 0 <= starts < ends:
        blobs.append(text[starts:ends + 1])
    return blobs


def _as_list(data):
    """从解析出来的 JSON 里取子问题列表：数组、或带 sub_questions/parts 的对象。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("sub_questions", "subquestions", "parts", "questions"):
            if isinstance(data.get(key), list):
                return data[key]
    return None


def parse_parts(raw, max_parts=None):
    """把模型输出解析成子问题列表；解析不出来返回 `[]`（由调用方退回原问句）。

    先按 JSON 解析（首选格式），不行再按行解析 —— 模型偶尔不理「只输出 JSON」，
    按行拿至少还能用，比整段丢弃强。清洗规则：去序号/项目符号/成对引号、去重、
    丢掉过短或过长的条目、按 `max_parts` 截断。
    """
    limit = MAX_PARTS if max_parts is None else max(1, int(max_parts))
    text = _strip_fence((raw or "").strip())
    if not text:
        return []
    items = None
    for blob in _json_blobs(text):
        try:
            parsed = json.loads(blob)
        except ValueError:
            continue
        items = _as_list(parsed)
        if items is not None:
            break
    if items is None:
        items = text.splitlines()

    parts = []
    for item in items:
        if isinstance(item, (dict, list)):
            continue
        part = _clean_part(item)
        if len(part) < MIN_PART_CHARS or len(part) > MAX_PART_CHARS:
            continue
        if part in parts:
            continue
        parts.append(part)
        if len(parts) >= limit:
            break
    return parts


def decompose_question(question, max_parts=None, use_gate=True):
    """把 `question` 拆成若干子问题；任何异常都退回 `[question]`。

    返回的列表**至少一条**（拆不出东西时就是原问句本身），调用方可以无条件地逐条检索。
    只拆出一条时返回的是**原问句**而不是模型改写过的版本 —— 见模块 docstring。
    """
    text = (question or "").strip()
    if not text:
        return []
    if use_gate and not looks_compound(text):
        return [text]
    if client is None:
        logger.warning("未配置 API_KEY，跳过拆问，按原问题检索")
        return [text]
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=MAX_TOKENS,
            # 同 HyDE / 可答性判断：推理模型不关思考会把小预算吃光、content 是空串
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("拆问返回空内容（finish_reason=%s），按原问题检索",
                           getattr(response.choices[0], "finish_reason", "?"))
            return [text]
        parts = parse_parts(content, max_parts=max_parts)
        if len(parts) <= 1:
            # 单问：用原问句。模型偶尔会把单问改写成同义句，那对检索只有坏处。
            if not parts:
                logger.warning("拆问输出无法解析（内容=%r），按原问题检索", content[:40])
            return [text]
        return parts
    except Exception as e:
        logger.warning("拆问失败，按原问题检索: %s", str(e)[:80])
        return [text]
