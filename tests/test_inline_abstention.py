# -*- coding: utf-8 -*-
"""「让 QA 自己拒答」（inline）这条路的判定口径回归。

拒答判定已经从「文案启发式」换成**结构化标记**（`prompts.REFUSAL_MARKER`）：
- 提示词要求模型答不出时以标记开头，服务端据此推 `refused` 事件；
- 测量脚本 `scripts/measure_inline_abstention.py` 用同一套契约判「这条算不算拒答」。

所以这里钉住：认标记、认 markdown 加粗、**不**把正文中间出现的标记当拒答、空回答算拒答。
"""
import json

import pytest

from scripts import measure_inline_abstention as mia
from src.prompts import REFUSAL_MARKER


@pytest.mark.parametrize("answer", [
    REFUSAL_MARKER,
    REFUSAL_MARKER + "参考背景只写了保留策略，没有说明异地机房步骤。",
    "**" + REFUSAL_MARKER + "**材料里没有相关内容。",
    "  " + REFUSAL_MARKER + "没有。",
    "",
    "   ",
])
def test_recognises_real_abstention(answer):
    assert mia.looks_like_abstention(answer) is True


@pytest.mark.parametrize("answer", [
    "根据《01-财务指标.md》，市盈率（PE）等于股价除以每股收益。",
    "净资产收益率是净利润与平均股东权益的百分比。该文件还提到杜邦分析把它拆成三部分。",
    # 正文中间提到标记的用法 —— 那是正常回答，不是拒答
    "根据《a.md》，" + REFUSAL_MARKER + "这种标记用于拒答。",
])
def test_does_not_flag_a_normal_answer(answer):
    assert mia.looks_like_abstention(answer) is False

# ── 拆问臂：两臂只差一个开关（2026-11-12 第十一轮）──
#
# 这条路的纪律是「不许在这层重新实现拆问」：`decomposed` 是线上 `llm._retrieve_hits`
# 的薄包装，「拆出 1 条退回整句」「分路全空退回整句」「每条子问过 resolve_query」都归
# 线上那一份实现。这里只钉住「开关有没有生效」和「关着时一次多余调用都没有」。

def test_pick_hits_uses_whole_retrieval_when_decompose_is_off():
    calls = []

    def whole(question):
        calls.append(("whole", question))
        return ["整句块"]

    def decomposed(question):
        calls.append(("decomposed", question))
        raise AssertionError("关着拆问时不该调用分路取块")

    assert mia.pick_hits("问一句", whole, decomposed=decomposed, decompose=None) == ["整句块"]
    assert calls == [("whole", "问一句")]


def test_pick_hits_uses_decomposed_retrieval_when_switched_on():
    calls = []

    def whole(question):
        calls.append(("whole", question))
        raise AssertionError("开着拆问时不该只走整句")

    def decomposed(question):
        calls.append(("decomposed", question))
        return ["子问块1", "子问块2"]

    hits = mia.pick_hits("复合问", whole, decomposed=decomposed, decompose=lambda q: ["a", "b"])
    assert hits == ["子问块1", "子问块2"]
    assert calls == [("decomposed", "复合问")]


def test_pick_hits_falls_back_to_whole_when_no_decomposed_backend():
    """防御性：给了 decompose 但没给分路取块函数时不许崩，退回整句。"""
    assert mia.pick_hits("问", lambda q: ["整句块"], decomposed=None,
                         decompose=lambda q: ["a"]) == ["整句块"]


def test_top_k_is_shared_by_both_arms():
    """两臂预算必须一样：拆问臂「每问各查 top_k 再分名额到 top_k」，整句臂一次 top_k。"""
    assert mia.TOP_K == 5, "改预算会同时改变两臂；改之前先想清楚这条实验还比什么"

# ── 整句臂必须走线上入口 + 预算可对齐（2026-11-12 第十三轮的质量门）──
#
# 两条都是「量错了会得出假结论」的地方：
# 1. 整句臂原来直接调 `hyde_retrieve_hits`，绕过了线上入口外面的跨文件取样那一层 ——
#    于是 `COVERAGE_SAMPLING` 这类线上开关根本量不到；
# 2. 预算写死 5 而线上是 `dynamic_top_k=max(10, 文件数×3)`：拿 5 去量 5 文件的语料，
#    会把「预算不够」误当成「取样的收益」（实测就是这样造出过 9/15 → 15/15 的假胜利）。

def test_whole_arm_goes_through_the_production_entry(tmp_path, monkeypatch):
    """整句臂必须调用 `llm._retrieve_hits`（线上入口），否则线上开关量不到。"""
    calls = []

    def fake_retrieve(question, collection, full_text, use_vector, top_k, files, decompose,
                      history=None):
        calls.append({"top_k": top_k, "decompose": decompose})
        return []

    import src.llm as llm_mod
    import src.retrieval as retrieval_mod

    monkeypatch.setattr(llm_mod, "_retrieve_hits", fake_retrieve)
    # `main` 会把 `retrieval._generate_hyde` 换成本地缓存查表——那是模块级全局，
    # 不还原就会污染后面的 HyDE 测试（实测挂了 4 条 test_retrieval）。
    monkeypatch.setattr(retrieval_mod, "_generate_hyde", retrieval_mod._generate_hyde)

    class _TinyEmbedding:
        """替身向量：这个用例只验「走没走线上入口、预算传没传下去」，
        不需要真的加载 embedding 模型（真加载会让整个套件从 21 s 变 71 s）。"""

        def __call__(self, input):                      # noqa: A002 - Chroma 的签名
            return [[0.0, 1.0] for _ in input]

        def name(self):
            return "tiny-test-embedding"

    monkeypatch.setattr(retrieval_mod, "_get_embedding_fn", lambda: _TinyEmbedding())

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("甲材料：实验一讲模拟退火。", encoding="utf-8")
    queries = tmp_path / "q.json"
    queries.write_text(json.dumps([{
        "id": "s1", "kind": "partial", "question": "有哪些内容",
        "expect_files": ["a.md"], "answer_terms": ["模拟退火", "实验一"], "absent_terms": [],
        "parts": [{"text": "提到模拟退火", "answer_terms": ["模拟退火"]},
                  {"text": "提到实验一", "answer_terms": ["实验一"]}],
    }], ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out.json"

    code = mia.main(["--corpus", str(corpus), "--queries", str(queries),
                     "--hyde-cache", str(tmp_path / "hyde.json"),
                     "--json", str(out), "--top-k", "15"])
    assert code == 0
    assert calls, "整句臂没有走线上入口"
    assert calls[0]["top_k"] == 15, "--top-k 必须传下去（不然比的是预算不是取样）"
    assert calls[0]["decompose"] is None
