# -*- coding: utf-8 -*-
"""跨编码器精排（reranker）：把向量召回的候选按「query × 块」的匹配度重新打分。

为什么需要它
------------
双编码器（embedding）把 query 和块**分别**编码，只在高维空间里比距离。2026-10-06 实测：
无关问题只要碰巧离某块近就会被留下，而任何绝对距离阈值都分不开正负 —— 合成语料在 HyDE
口径下正样本 max 0.5316 > 负样本 min 0.5074，真实语料同样重叠；`n01`（量子纠缠）与 `n03`
（莫扎特奏鸣曲）两条无关问题就是这样被「强行作答」的（误召回 0/5 → 2/5）。跨编码器把 query
和块**拼在一起**过一遍模型，给出逐对的匹配分，才有机会把「整批都无关」这种情况挡住。

设计要点
--------
- **懒加载**：真要重排时才 import torch / sentence-transformers，评测与单测因此不必碰模型
  （仓库约定：测试套件不加载 sentence-transformers）。
- **可用性优先**：没装依赖 / 模型没下载 / 加载失败 / 打分抛错 —— 一律记 WARNING 并返回 None，
  调用方退回纯距离路径，绝不让整次问答失败；失败过就记住，不每次问答都重试一遍加载。
- **用原问题打分**：跨编码器的训练分布是 question → passage；HyDE 文本是「像文档的片段」，
  拿它当 query 属于分布外。顺带这一步也把「HyDE 编偏」的影响压回来。
"""
from src.config import RERANK_MODEL, logger

_scorer = None          # 进程内单例
_scorer_failed = False  # 加载失败过就不再反复重试


def _make_cross_encoder(model_name):
    """真正去构造 CrossEncoder。

    单独抽出来是为了让测试能注入一个「加载失败」的替身，不必真去下载 1.1GB 权重。
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _get_scorer(model_name=None):
    """懒加载并复用 CrossEncoder；不可用（含加载失败）返回 None。"""
    global _scorer, _scorer_failed
    if _scorer is not None or _scorer_failed:
        return _scorer
    name = model_name or RERANK_MODEL
    try:
        _scorer = _make_cross_encoder(name)
    except Exception as e:
        logger.warning("reranker 加载失败（%s），本次退回向量距离排序: %s", name, str(e)[:120])
        _scorer_failed = True
        return None
    return _scorer


def score_pairs(query, texts):
    """对 (query, 块) 逐对打分，返回 [float]；模型不可用或打分出错返回 None。"""
    scorer = _get_scorer()
    if scorer is None or not texts:
        return None
    try:
        scores = scorer.predict([(query, text) for text in texts])
    except Exception as e:
        logger.warning("reranker 打分失败，退回向量距离排序: %s", str(e)[:120])
        return None
    return [float(s) for s in scores]


def rerank(query, hits, top_n, min_score):
    """按跨编码器分数重排候选，返回 [(text, meta, score)]。

    按分数降序，只留 `score >= min_score` 的前 `top_n` 条。

    **两种「没有结果」必须分开**：模型不可用 → 返回 `None`（调用方退回距离阈值，
    免得把一堆远块原样塞给模型）；用了、但一条都不达标 → 返回 `[]`（说明这批候选确实都不
    相关，交给关键词兜底）。
    """
    if not hits:
        return []
    scores = score_pairs(query, [text for text, _ in hits])
    if scores is None:
        return None
    ranked = sorted(((text, meta, score) for (text, meta), score in zip(hits, scores)),
                    key=lambda item: item[2], reverse=True)
    return [item for item in ranked if item[2] >= min_score][:top_n]
