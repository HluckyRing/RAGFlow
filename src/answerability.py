# -*- coding: utf-8 -*-
"""可答性判断：材料里到底有没有能回答这个问题的信息。

为什么要有它（2026-11-05 实测，见 docs/retrieval-baseline-2026-10-06.md「重排能救到哪一步」）
================================================================================
无关提问被强行作答有两个漏点，都不是调参能解决的：

1. **向量侧**：域内「看着像但没答案」的提问，两条探针都贴近语料，重排分甚至能到 0.95
   （`n23`），阈值扫描 0.3 / 0.5 / 0.8 全是「既漏又误杀」；
2. **关键词兜底侧**：重排把域内样本挡下后，2-gram 词面命中又把它捞回来；要求命中词数 ≥5
   只能把域内从 8/8 降到 1/8，却误杀 16/36 正样本。

所以这里不再问「有多相关」，而是直接问模型「这段材料能不能回答这个问题」。

三条纪律
--------
- **关思考、小预算**：`deepseek-flash` 是推理模型，`max_tokens` 给小了会被 reasoning 吃光、
  `content` 返回空串（HyDE 就是这么静默失效的，见 `retrieval._generate_hyde`）；
- **只认「能」/「不能」**：解析认不出就返回 `None`（不知道），绝不猜成「不能」去拒答；
- **任何故障都不许堵死问答**：缺 API_KEY、调用失败、空内容一律 `None` + WARNING，由上层
  退回「按老行为作答」。降级必须出声，但不许变成拒答。
"""
import src.config as config
from src.config import logger
from src.prompts import ANSWERABILITY_SYSTEM_PROMPT

# 从 config 拷一份模块级引用：测试与评测脚本可以按开关覆盖它，不必碰全局 config
ANSWERABILITY_ENABLED = config.ANSWERABILITY_CHECK
MAX_TOKENS = config.ANSWERABILITY_MAX_TOKENS

ANSWER_YES = "能"
ANSWER_NO = "不能"
# 判定用的模型客户端：单独取一次，测试可以替换成替身（也方便日后换更便宜的模型）
client = config.client


def parse_verdict(raw):
    """把模型输出解析成 True / False / None（None = 不知道）。

    先看「不能」再看「能」：模型常见输出是「不能，因为只写了响应时效」，子串判断顺序反了
    会把所有否定判成肯定。
    """
    text = (raw or "").strip()
    if not text:
        return None
    head = text[:6]
    if ANSWER_NO in head or text.startswith(("否", "无", "没有")):
        return False
    if ANSWER_YES in head or text.startswith(("是", "可以", "有")):
        return True
    return None


def judge_answerable(question, context, max_tokens=None):
    """判断 `context` 能不能回答 `question`。

    返回 True（能）/ False（不能）/ None（判不了，调用方按老行为继续作答）。
    """
    if not (context or "").strip():
        return True                     # 没有材料就没什么可判的，检索层已经决定返回空
    if client is None:
        logger.warning("未配置 API_KEY，跳过可答性判断，按原行为作答")
        return None
    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": ANSWERABILITY_SYSTEM_PROMPT},
                {"role": "user", "content": f"【问题】{question}\n\n【材料】\n{context}"},
            ],
            temperature=0,
            max_tokens=MAX_TOKENS if max_tokens is None else max_tokens,
            # 同 HyDE：切到推理模型后不关思考，小预算会被 reasoning 吃光、content 是空串
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("可答性判断返回空内容（finish_reason=%s），按原行为作答",
                           getattr(response.choices[0], "finish_reason", "?"))
            return None
        verdict = parse_verdict(content)
        if verdict is None:
            logger.warning("可答性判断输出无法解析（内容=%r），按原行为作答", content[:40])
        return verdict
    except Exception as e:
        logger.warning("可答性判断失败，按原行为作答: %s", str(e)[:80])
        return None
