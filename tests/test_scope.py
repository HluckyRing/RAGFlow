# -*- coding: utf-8 -*-
"""范围词 → 文件过滤器（`SCOPE_FILTER`）：把「第2章课件」从检索词变成过滤器。

背景（2026-11-19 真实对话逐块复现）：同一条问题
「结合第2章课件，在同一训练集上不断提高多项式回归的阶数，通常可以观察到什么」
只捞到 1 块（第1章 p46 的通用框架页）→ 老实拒答；去掉前缀却能捞到 3–4 块并答对。
原因有两层：元信息词（「第2章」「课件」）把封面页 / 内容提要页顶到前面；以及
`intersect` 闸门要求两路都召回，交集于是塌成 1 块。这个文件钉住四条边界：

1. 认不准就不动手（没章号 / 章号匹配多份文件 / 删完太短）；
2. 开关关着时**逐位不变**（`where` 不传、探针文本原样）；
3. 范围过滤把命中砍光 → 退回全库，绝不让一次过滤把问答答空；
4. 候选池放大（`FUSION_CANDIDATES`）只改「每路取多少候选」，最终块数仍截到 `top_k`。
"""
import src.retrieval as retrieval
from src import scope


# ── 纯函数：范围短语识别与改写 ──

def test_strip_scope_removes_the_instruction_prefix_only():
    """「结合第2章课件，」是**去哪儿找**，不是**要找什么** —— 只删它，不删信息需求。"""
    question = "结合第2章课件，在同一训练集上不断提高多项式回归的阶数，通常可以观察到什么"

    assert scope.strip_scope(question) == "在同一训练集上不断提高多项式回归的阶数，通常可以观察到什么"


def test_strip_scope_handles_politeness_and_spacing():
    assert scope.strip_scope("请根据第 3 章的讲义：k 值怎么选") == "k 值怎么选"
    assert scope.strip_scope("参考第2章PPT，决定系数怎么看") == "决定系数怎么看"


def test_strip_scope_returns_none_when_the_whole_question_is_scope():
    """「结合第2章课件」本身没有问题可言：删完什么都不剩，就不许动手。"""
    assert scope.strip_scope("结合第2章课件") is None


def test_strip_scope_returns_none_without_a_material_word():
    """只有章号没有「课件/讲义」这类词 —— 这句可能是「整理第一章」，不属于本模块的范围。"""
    assert scope.strip_scope("帮我整理第一章的内容") is None


# ── 定位：章号 → 唯一的一份文件 ──

FILES = ["第1章 - 绪论.pdf", "第2章 - 回归分析.pdf", "第3章 - k近邻算法.pdf"]


def test_scope_target_finds_the_named_chapter_and_rewrites_the_probe():
    target = scope.scope_target("结合第2章课件，过拟合和欠拟合怎么区分", FILES)

    assert target == {"file_name": "第2章 - 回归分析.pdf", "question": "过拟合和欠拟合怎么区分"}


def test_scope_target_keeps_the_question_when_there_is_only_a_chapter_number():
    """「第2章讲了什么」没有可删的范围短语，但仍然要**限定到这一份文件**（人要的就是这一章）。"""
    target = scope.scope_target("第2章讲了什么", FILES)

    assert target == {"file_name": "第2章 - 回归分析.pdf", "question": "第2章讲了什么"}


def test_scope_target_refuses_when_the_chapter_is_ambiguous():
    """章号匹配到两份文件（同名章节的两门课）→ 认不准就不动手，绝不猜。"""
    files = ["第2章 - 回归分析.pdf", "第2章 - 进程管理.pdf"]

    assert scope.scope_target("结合第2章课件，过拟合是什么", files) is None


def test_scope_target_ignores_questions_without_a_chapter_number():
    assert scope.scope_target("过拟合是什么", FILES) is None


def test_scope_target_ignores_empty_file_list():
    assert scope.scope_target("结合第2章课件，过拟合是什么", []) is None
    assert scope.scope_target("结合第2章课件，过拟合是什么", ["", None]) is None


# ── 接线：检索层真的换了 where、也真的换了探针 ──

class _ScopeCollection:
    """记录每次 `query` 的 (探针文本, n_results, where)，并按 where 决定返回什么。

    `where_results` 非空时表示「文件过滤后的结果」—— 给「过滤砍光后要退回全库」那条用例用。
    """

    def __init__(self, by_query, distances=None, where_results=None, total=10):
        self.by_query = by_query
        self.distances = distances or {}
        self.where_results = where_results
        self.total = total
        self.calls = []

    def count(self):
        return self.total

    def query(self, query_texts, n_results, where=None):
        self.calls.append((list(query_texts), n_results, where))
        table = self.where_results if (where is not None and self.where_results is not None) else self.by_query
        docs, dists, metas = [], [], []
        for text in query_texts:
            found = list(table.get(text, []))
            docs.append(found)
            dists.append(list(self.distances.get(text, [0.30] * len(found))))
            metas.append([{"file_name": "第2章 - 回归分析.pdf"} for _ in found])
        return {"documents": docs, "distances": dists, "metadatas": metas}


def _enable_scope(monkeypatch, enabled=True, hyde_text="假设答案", gate=None):
    monkeypatch.setattr(retrieval, "SCOPE_FILTER", enabled)
    monkeypatch.setattr(retrieval, "FUSION_ENABLED", True)
    monkeypatch.setattr(retrieval, "FUSION_GATE",
                        retrieval.GATE_INTERSECT if gate is None else gate)
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", False)
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: hyde_text)


def test_scope_filter_limits_the_query_and_rewrites_the_probe(monkeypatch):
    _enable_scope(monkeypatch)
    coll = _ScopeCollection({"假设答案": ["块甲"], "过拟合是什么": ["块甲"]})

    retrieval.hyde_retrieve_hits_with_path(
        "结合第2章课件，过拟合是什么",
        coll, "", use_vector=True, files=[(name, "正文") for name in FILES])

    texts, _n, where = coll.calls[0]
    assert where == {"file_name": "第2章 - 回归分析.pdf"}
    assert texts == ["假设答案", "过拟合是什么"], "探针里的范围短语必须已经被删掉"


def test_scope_filter_off_keeps_the_old_behaviour_bit_for_bit(monkeypatch):
    """开关关着：不传 where、探针文本一个字都不改 —— 这是回退保命的那条路。"""
    _enable_scope(monkeypatch, enabled=False)
    coll = _ScopeCollection({"假设答案": ["块甲"], "结合第2章课件，过拟合是什么": ["块甲"]})

    retrieval.hyde_retrieve_hits_with_path(
        "结合第2章课件，过拟合是什么",
        coll, "", use_vector=True, files=[(name, "正文") for name in FILES])

    texts, _n, where = coll.calls[0]
    assert where is None
    assert texts == ["假设答案", "结合第2章课件，过拟合是什么"]


def test_scope_filter_cannot_run_without_a_file_list(monkeypatch):
    """没有文件列表就没有「文件名 → 那一份」的映射：只能退回老行为，绝不瞎猜。"""
    _enable_scope(monkeypatch)
    coll = _ScopeCollection({"假设答案": ["块甲"], "结合第2章课件，过拟合是什么": ["块甲"]})

    retrieval.hyde_retrieve_hits_with_path("结合第2章课件，过拟合是什么", coll, "", use_vector=True)

    _texts, _n, where = coll.calls[0]
    assert where is None


def test_scope_filter_falls_back_to_the_whole_corpus_when_it_hits_nothing(monkeypatch):
    """范围过滤把命中砍光（例如章号认对了、文件里却没这页）→ 退回全库，绝不许答空。"""
    _enable_scope(monkeypatch)
    coll = _ScopeCollection(
        by_query={"假设答案": ["块甲"], "过拟合是什么": ["块甲"]},
        where_results={},                      # 过滤后一块都没有
    )

    _path, hits = retrieval.hyde_retrieve_hits_with_path(
        "结合第2章课件，过拟合是什么",
        coll, "", use_vector=True, files=[(name, "正文") for name in FILES])

    assert [text for text, _meta in hits] == ["块甲"]
    assert coll.calls[0][2] == {"file_name": "第2章 - 回归分析.pdf"}
    assert coll.calls[1][2] is None, "退回全库那一次不许再带 where"


# ── 候选池放大：只给交集留容错，不改最终块数 ──

def _long_lists(n=20):
    pool = [f"块{i}" for i in range(n)]
    return {name: list(pool) for name in ("假设答案", "问题")}


def test_fusion_candidates_widen_the_pool_but_keep_the_top_k_budget(monkeypatch):
    _enable_scope(monkeypatch, enabled=False)
    monkeypatch.setattr(retrieval, "FUSION_CANDIDATES", 20)
    monkeypatch.setattr(retrieval, "RRF_K", 60)
    coll = _ScopeCollection(_long_lists(), total=400)

    _path, hits = retrieval.hyde_retrieve_hits_with_path(
        "问题", coll, "", use_vector=True, top_k=10)

    _texts, n_results, _where = coll.calls[0]
    assert n_results == 20, "每路先取 20 个候选（闸门之后才截到 top_k）"
    assert len(hits) == 10, "块预算不变：最终仍只留 top_k 块"


def test_fusion_candidates_off_by_default(monkeypatch):
    """默认 0 = 每路只取 top_k，与改造前逐位一致。"""
    _enable_scope(monkeypatch, enabled=False)
    monkeypatch.setattr(retrieval, "FUSION_CANDIDATES", 0)
    coll = _ScopeCollection(_long_lists(), total=400)

    retrieval.hyde_retrieve_hits_with_path("问题", coll, "", use_vector=True, top_k=10)

    _texts, n_results, _where = coll.calls[0]
    assert n_results == 10
