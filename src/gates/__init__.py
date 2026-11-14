# -*- coding: utf-8 -*-
"""闸门措辞包的选择入口（`GATE_PACK` 决定用哪一套）。

设计原则：
- **默认 `zh`**，与本项目今天的语料一致；
- 未知包名**不崩**，退回 `zh` 并打 WARNING —— 一个拼错的环境变量不该让服务起不来，
  但也绝不能安静地换掉判断口径；
- 包里缺字段同样报 WARNING 并补空元组：宁可闸门不触发（fail-safe），不许 `AttributeError`。

加一门新语言 = 加一个 `src/gates/xx.py`，里面至少有
`COMPOUND_MARKERS / QUESTION_WORDS / ENUMERATION_MARKERS / NAVIGATION_MARKERS` 四个元组。
"""
from types import SimpleNamespace

from src.gates import zh

# 字段名 → 缺省值。加包时照这个填。
REQUIRED = ("COMPOUND_MARKERS", "QUESTION_WORDS", "ENUMERATION_MARKERS", "NAVIGATION_MARKERS")

PACKS = {"zh": zh}


def pack_names():
    return tuple(sorted(PACKS))


def load_pack(name=None, logger=None):
    """按名字取一套闸门措辞；未知名字退回 `zh` 并告警。"""
    wanted = str(name or "zh").strip().lower() or "zh"
    module = PACKS.get(wanted)
    if module is None:
        if logger is not None:
            logger.warning("未知的 GATE_PACK=%r，退回 zh（已知：%s）", name, ", ".join(pack_names()))
        module = zh
        wanted = "zh"
    values = {}
    for field in REQUIRED:
        value = getattr(module, field, None)
        if value is None:
            if logger is not None:
                logger.warning("闸门包 %r 缺少 %s，按空表处理（闸门不会触发）", wanted, field)
            value = ()
        values[field.lower()] = tuple(value)
    return SimpleNamespace(name=wanted, **values)
