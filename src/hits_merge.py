# -*- coding: utf-8 -*-
"""多路检索结果的合并口径（纯函数，不碰 Chroma / 模型 / 配置）。

为什么单独一个模块：**这套口径有两处调用方** —— 线上的「每问分路检索」（`llm.py`）
和评测脚本（`scripts/eval_compound_retrieval.py`）。口径写两份必然漂，而漂了以后
评测数字就不再说明线上行为（本项目在「归一化」「命中判定」上都吃过这个亏）。

口径（2026-11-08 实测：同一份 5 块 context 下，整句 45.9% → 每问分路 81.1%）
--------------------------------------------------------------------------------
**按名次轮流分名额**：先给每一问它的第 1 名，再轮流给第 2 名……直到装满 budget 块。
这样一来「某一问的第 2 名」永远不会把「另一问的第 1 名」挤掉 —— 这正是整句检索失败的地方
（一句复合问句被压成一个向量，几件事互相拉扯）。同一片正文被两问都召回时只占一个名额。
"""
import re

# 块身份：优先 (文件名, 片号)；老索引没有 meta 时退回正文本身。
# 去重必须按**块身份**而不是正文 —— 两个不同文件里恰好有相同的正文，那是两块。
def hit_key(hit):
    text, meta = hit
    if isinstance(meta, dict):
        name, index = meta.get("file_name"), meta.get("chunk_index")
        if name is not None and index is not None:
            return (name, index)
    return text


def interleave_hits(part_hits, budget):
    """每问轮流取名额，直到装满 `budget` 块（去重、保序、绝不超预算）。

    某问已经没料时名额自动留给还在出货的那些问，不空转。
    """
    budget = max(0, int(budget or 0))
    ordered, seen = [], set()
    depth = max((len(hits or []) for hits in part_hits), default=0)
    for rank in range(depth):
        for hits in part_hits:
            if rank >= len(hits or []):
                continue
            key = hit_key(hits[rank])
            if key in seen:
                continue
            seen.add(key)
            ordered.append(hits[rank])
            if len(ordered) >= budget:
                return ordered
    return ordered


def hit_identity_text(hit):
    """`(文件名, 片号, 正文前 40 字)`：只给日志/评测明细用，**不含完整正文**。"""
    text, meta = hit
    name = (meta or {}).get("file_name") if isinstance(meta, dict) else None
    index = (meta or {}).get("chunk_index") if isinstance(meta, dict) else None
    return (name, index, re.sub(r"\s+", " ", str(text or ""))[:40])
