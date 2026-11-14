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
# 多轮时喂给模型的历史：只给最近两轮（user/assistant 各一条算一轮），并且每条截断。
# 目的只有一个 —— 让省略式追问（「那病假呢」「那这个规定还有效吗」）能被补成可检索的问句；
# 给得越多越容易把不相关的上下文拽进拆问，反而拆歪。
HISTORY_MESSAGES = 4
HISTORY_CHAR_LIMIT = 200

# 拆问用的模型客户端：单独取一次，测试可以替换成替身
client = config.client

# ── 廉价闸门 ──
# 措辞表在 `src/gates/`（默认 zh）：闸门靠措辞判断意图，换语言就失效，所以外置成可插拔的包，
# 并在调用点打触发率日志 —— 失效必须看得见。
import src.config as _config
from src.gates import load_pack as _load_pack

_gates = _load_pack(_config.GATE_PACK, logger=_config.logger)
COMPOUND_MARKERS = _gates.compound_markers
QUESTION_WORDS = _gates.question_words
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


def history_digest(history):
    """把最近几轮压成一段短文，喂给拆问模型（纯函数，便于测试）。

    只取**用户/助手**两类角色的最近 `HISTORY_MESSAGES` 条，每条截到 `HISTORY_CHAR_LIMIT` 字。
    为什么要给历史：多轮里的省略式追问（「那病假呢？」「那这个规定还有效吗？」）单看本句
    是检索不到东西的 —— 主语在**上一轮**里。拆问模型拿到历史就能把子问补成独立问句，
    这比事后按规则拼实体准得多（规则拼出来是「年假 那病假呢」这种半截话）。
    """
    lines = []
    for message in history or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
        if not content:
            continue
        lines.append(("用户：" if role == "user" else "助手：") + content[:HISTORY_CHAR_LIMIT])
    # 先筛角色再截最近几条：反过来的话，中间夹一条系统消息就会白占一个名额
    return "\n".join(lines[-HISTORY_MESSAGES:])


def _build_messages(question, history):
    """拆问的 messages：有历史就把历史放在问题前面（没有历史时与上一版逐字相同）。"""
    digest = history_digest(history)
    if not digest:
        return [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    system = (DECOMPOSE_SYSTEM_PROMPT
              + "\n6. 如果某个子问题是省略式追问（例如「那病假呢」「那这个规定还有效吗」），"
                "结合【上一轮对话】把它补成能单独检索的问句（补的是主语/话题，不要编造限制条件）。")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【上一轮对话】\n{digest}\n\n【本轮问题】\n{question}"},
    ]


def decompose_question(question, history=None, max_parts=None, use_gate=True):
    """把 `question` 拆成若干子问题；任何异常都退回 `[question]`。

    返回的列表**至少一条**（拆不出东西时就是原问句本身），调用方可以无条件地逐条检索。
    只拆出一条时返回的是**原问句**而不是模型改写过的版本 —— 见模块 docstring。

    `history` 是**不含当前提问**的历史快照（与 `server.chat` 一致）：给了它，省略式追问
    才能被补成独立子问。没有历史时提示词与上一版逐字相同（可复现）。
    """
    text = (question or "").strip()
    if not text:
        return []
    if use_gate:
        # 触发率要能量出来：闸门靠措辞表判断意图，换语言/换语料可能整片失效，
        # 而失效是静默的（功能悄悄不生效）。固定前缀方便日志里数 hit/miss。
        if not looks_compound(text):
            logger.info("闸门 compound=miss pack=%s", _gates.name)
            return [text]
        logger.info("闸门 compound=hit pack=%s", _gates.name)
    if client is None:
        logger.warning("未配置 API_KEY，跳过拆问，按原问题检索")
        return [text]
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=_build_messages(text, history),
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
