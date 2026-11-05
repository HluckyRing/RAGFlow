# -*- coding: utf-8 -*-
"""关键词降级检索回归：必须按相关性排序，而不是永远返回文档前两块。"""
import logging
from types import SimpleNamespace

import pytest

import src.pdf_ingestion as pdf_ingestion
import src.retrieval as retrieval
from src.text_utils import extract_terms, strip_question_words

PE_CHUNK = "市盈率（PE）等于股价除以每股收益，是衡量股票估值高低的重要指标。"
ROE_CHUNK = "净资产收益率（ROE）是净利润与平均股东权益的百分比，反映股东权益的收益水平。"
CHUNKS = [PE_CHUNK, ROE_CHUNK]


# ── 抽词 ──

def test_strip_question_words():
    assert strip_question_words("什么是市盈率").strip() == "市盈率"
    assert strip_question_words("ROE 是什么").strip() == "ROE"


def test_extract_terms_uses_ngrams():
    terms = extract_terms("什么是净资产收益率")
    assert {"净资", "资产", "收益"} <= terms
    # 疑问词已被剥掉，不会留下「什么」「么是」这类垃圾词
    assert "什么" not in terms
    assert "么是" not in terms


# ── 打分排序 ──

def test_ranks_relevant_chunk_first():
    # 旧实现在这里两块都是 0 分，于是回退成「返回前 2 块」，PE 块反而排在前面
    top = retrieval.retrieve_keyword("什么是净资产收益率", CHUNKS)
    assert top, "应该至少命中一块"
    assert top[0] == ROE_CHUNK


def test_returns_empty_when_nothing_matches():
    assert retrieval.retrieve_keyword("量子纠缠的退相干时间", ["完全无关的一段文字内容"]) == []


def test_respects_top_n():
    assert len(retrieval.retrieve_keyword("净资产收益率", [ROE_CHUNK] * 5, top_n=2)) == 2


# ── HyDE 只在真正查向量时才调用 ──

class _FakeCollection:
    def __init__(self):
        self.queries = []

    def count(self):
        return 1

    def query(self, query_texts, n_results):
        self.queries.append(query_texts[0])
        return {"documents": [[ROE_CHUNK]]}


def test_vector_mode_uses_hyde(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设性答案文本")
    coll = _FakeCollection()
    out = retrieval.hyde_retrieve("什么是净资产收益率", coll, ROE_CHUNK, use_vector=True)
    assert coll.queries == ["假设性答案文本"]
    assert out == [ROE_CHUNK]


def test_keyword_mode_skips_hyde_and_uses_original_question(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: called.append(q) or "假设性答案文本")

    out = retrieval.hyde_retrieve("什么是净资产收益率", None, ROE_CHUNK, use_vector=False)

    assert called == [], "关键词模式下不该白花一次 API 调用"
    assert out and out[0] == ROE_CHUNK


# ── 相关性阈值（l2 距离）──

class _DistanceCollection:
    """按给定距离返回候选块，用来确定性地测阈值过滤。"""

    def __init__(self, docs, distances):
        self._docs = docs
        self._distances = distances

    def count(self):
        return len(self._docs)

    def query(self, query_texts, n_results):
        return {"documents": [self._docs], "distances": [self._distances]}


def test_filter_by_distance_keeps_close_drops_far():
    kept = retrieval._filter_by_distance(["近的块", "远的块"], [0.40, 0.90], max_distance=0.55)
    assert kept == ["近的块"]


def test_filter_by_distance_keeps_everything_when_no_distances():
    docs = ["a", "b"]
    assert retrieval._filter_by_distance(docs, [], max_distance=0.55) == docs


def test_vector_results_over_threshold_are_dropped(monkeypatch):
    """阈值内的块保留，超阈值的块不进 context。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["相关的块", "无关的块"], [0.40, 0.62])

    out = retrieval.hyde_retrieve("问题", coll, "全文", use_vector=True)

    assert out == ["相关的块"]


def test_all_above_threshold_falls_back_to_keyword(monkeypatch):
    """向量全部超阈值 → 走关键词兜底；关键词也不命中才最终返回空。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["块A", "块B"], [0.80, 0.90])

    out = retrieval.hyde_retrieve("量子纠缠退相干时间", coll, "完全无关的一段文字", use_vector=True)

    assert out == []


# ── 检索路径标签（评测靠它区分「向量召回」与「关键词兜底」）──

def test_hits_with_path_reports_the_vector_path(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.40])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_VECTOR
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_hits_with_path_reports_the_keyword_fallback(monkeypatch):
    """向量全超阈值、靠兜底捞回来：路径必须标成兜底，不能记成向量。

    两条路的分数含义不同（l2 距离 vs 2-gram 打分），混在一起会误判阈值的好坏。
    """
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.90])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert [text for text, _ in hits] == [ROE_CHUNK], "兜底按词面命中，这条应该被捞回来"


def test_hits_with_path_stays_keyword_when_the_fallback_finds_nothing(monkeypatch):
    """兜底也没命中（负样本的样子）：路径仍然是兜底 —— 那条路确实跑了。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.90])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "量子纠缠退相干时间", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert hits == []


def test_hits_with_path_in_pure_keyword_mode():
    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", None, ROE_CHUNK, use_vector=False)

    assert path == retrieval.PATH_KEYWORD
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_hits_without_path_keeps_the_old_signature(monkeypatch):
    """`hyde_retrieve_hits` / `hyde_retrieve` 的老签名不能变（调用方与老测试都在用）。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.40])

    assert retrieval.hyde_retrieve_hits(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True) == [(ROE_CHUNK, None)]
    assert retrieval.hyde_retrieve(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True) == [ROE_CHUNK]


# ── 缺 API_KEY 时的降级（P1-16）──

def test_hyde_skips_with_clear_hint_when_client_is_none(monkeypatch, caplog):
    """没配 API_KEY 时 client 是 None：HyDE 直接用原问题，并给出明确中文提示，
    而不是把它混在「HyDE 生成失败」的普通告警里。"""
    monkeypatch.setattr(retrieval, "client", None)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("什么是净资产收益率")

    assert out == "什么是净资产收益率"
    assert any("API_KEY" in r.getMessage() for r in caplog.records)


# ── HyDE 必须真的生成内容（2026-10-06：推理模型吃光预算导致静默退化）──

def _use_fake_client(monkeypatch, content=None, finish_reason="stop", boom=False):
    """装上假 client，返回记录调用参数的 completions 替身。"""
    completions = SimpleNamespace(calls=[], )

    def create(**kwargs):
        completions.calls.append(kwargs)
        if boom:
            raise RuntimeError("网关 500")
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content))])

    completions.create = create
    monkeypatch.setattr(retrieval, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=completions)))
    return completions


def test_hyde_disables_thinking_so_reasoning_does_not_eat_the_budget(monkeypatch):
    """deepseek-flash 是推理模型：不关思考的话，200 的 cap 会被 reasoning 吃光、
    content 返回空串，HyDE 就变成「原问题直查」而且一声不响（2026-10-06 实测：
    finish_reason=length、reasoning_tokens=200、content 长度 0）。"""
    completions = _use_fake_client(monkeypatch, content="假设性文档片段")

    out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "假设性文档片段"
    sent = completions.calls[0]
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["max_tokens"] == 200, "cap 留着当防跑飞闸，靠关思考而不是放宽它"


def test_hyde_warns_and_falls_back_when_content_is_empty(monkeypatch, caplog):
    """空内容必须出声：静默退回原问题会让所有「HyDE 口径」的结论变成假的。"""
    _use_fake_client(monkeypatch, content="", finish_reason="length")

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "漏损率怎么算"
    assert any("空内容" in r.getMessage() for r in caplog.records)


def test_hyde_falls_back_on_api_error(monkeypatch, caplog):
    _use_fake_client(monkeypatch, boom=True)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "漏损率怎么算"
    assert any("HyDE 生成失败" in r.getMessage() for r in caplog.records)


# ── reranker 接入（2026-10-06：无关问题靠距离分不开，改用 query×块 逐对打分）──

def _enable_rerank(monkeypatch, rerank_impl, top_n=1, min_score=0.0, candidates=10):
    """打开重排开关并注入替身 —— 测试绝不加载真的模型（见 conftest 的 autouse）。"""
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANK_CANDIDATES", candidates)
    monkeypatch.setattr(retrieval, "RERANK_TOP_N", top_n)
    monkeypatch.setattr(retrieval, "RERANK_MIN_SCORE", min_score)
    monkeypatch.setattr(retrieval.rerank_mod, "rerank", rerank_impl)


def test_rerank_candidates_skip_the_distance_filter_and_use_the_original_question(monkeypatch):
    """开了重排：候选**不再**先用距离砍（距离在这条路上不是最终裁判），
    而且打分用**原问题**而不是 HyDE 文本（跨编码器的训练分布是 question → passage）。"""
    seen = {}

    def fake_rerank(query, hits, top_n, min_score):
        seen["query"] = query
        seen["texts"] = [text for text, _ in hits]
        return [(hits[-1][0], hits[-1][1], 0.9)]

    _enable_rerank(monkeypatch, fake_rerank)
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["近块", "远块"], [0.40, 0.99])      # 远块远超阈值

    path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert [text for text, _ in hits] == ["远块"], "超阈值的候选也要交给重排"
    assert seen["query"] == "原问题"
    assert seen["texts"] == ["近块", "远块"]
    assert path == retrieval.PATH_VECTOR


def test_rerank_unavailable_falls_back_to_the_distance_filter(monkeypatch):
    """重排不可用（None）必须退回距离阈值 —— 否则会把一堆远块原样塞给模型。"""
    _enable_rerank(monkeypatch, lambda *a, **k: None)
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["近块", "远块"], [0.40, 0.99])

    _path, hits = retrieval.hyde_retrieve_hits_with_path("问题", coll, "全文", use_vector=True)

    assert [text for text, _ in hits] == ["近块"]


def test_rerank_dropping_everything_falls_back_to_keyword(monkeypatch):
    """重排把候选全判为不相关 → 命中为空 → 走关键词兜底，与「距离全丢」语义一致。"""
    _enable_rerank(monkeypatch, lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["块A"], [0.30])                    # 距离上其实很近

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "量子纠缠退相干时间", coll, "完全无关的一段文字", use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert hits == []


def test_rerank_off_keeps_the_old_distance_behaviour(monkeypatch):
    """开关关掉时行为必须与从前逐位一致（这是回退保命的那条路）。"""
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", False)
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["近块", "远块"], [0.40, 0.99])

    _path, hits = retrieval.hyde_retrieve_hits_with_path("问题", coll, "全文", use_vector=True)

    assert [text for text, _ in hits] == ["近块"]


def test_retrieval_exposes_all_rerank_knobs():
    """评测脚本要靠 `retrieval.RERANK_*` 打印与覆盖这些开关。

    少暴露一个就是运行时 AttributeError（2026-10-06 实测：`RERANK_MODEL` 漏在导入列表外，
    `--rerank` 一跑就崩），所以这里把名字钉住。
    """
    for name in ("RERANK_ENABLED", "RERANK_MODEL", "RERANK_CANDIDATES",
                 "RERANK_TOP_N", "RERANK_MIN_SCORE"):
        assert hasattr(retrieval, name), f"retrieval 缺 {name}"


# ── 来源元数据（P2）──

class _MetaCollection:
    """返回带 metadatas 的候选块，用来验证来源能一路带到上层。"""

    def __init__(self, docs, distances, metas):
        self._docs, self._distances, self._metas = docs, distances, metas

    def count(self):
        return len(self._docs)

    def query(self, query_texts, n_results):
        return {"documents": [self._docs], "distances": [self._distances],
                "metadatas": [self._metas]}


def test_hits_carry_source_metadata(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    meta = {"file_name": "a.md", "chunk_index": 0, "section": "第 1 页"}
    coll = _MetaCollection(["相关的块"], [0.30], [meta])

    hits = retrieval.hyde_retrieve_hits("问题", coll, "全文", use_vector=True)

    assert hits == [("相关的块", meta)]


def test_hits_without_metadata_give_none_not_a_fake_source(monkeypatch):
    """老索引没有 metadatas：来源必须是 None，由上层标「未标注」，不能编。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _MetaCollection(["相关的块"], [0.30], [None])

    hits = retrieval.hyde_retrieve_hits("问题", coll, "全文", use_vector=True)

    assert hits == [("相关的块", None)]


def test_chroma_meta_drops_none_fields():
    """Chroma 的 metadata 不收 None，所以建索引前要把值为 None 的字段删掉。"""
    assert retrieval.chroma_meta({"file_name": "a.md", "section": None}) == {"file_name": "a.md"}
    assert retrieval.chroma_meta(None) == {"file_name": "未知文件"}


def test_hyde_retrieve_still_returns_plain_texts(monkeypatch):
    """旧签名保持兼容：只给文本，老调用方不用改。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _MetaCollection(["块A"], [0.30], [{"file_name": "a.md"}])

    assert retrieval.hyde_retrieve("问题", coll, "全文", use_vector=True) == ["块A"], "应提示缺少 API_KEY"


# ── 双路融合：原问题 + HyDE 各当一次探针，按 RRF 融合名次 ──

class _TwoProbeCollection:
    """按 query 文本返回不同候选，用来确定性地测「两路召回 + 融合」。

    真实 Chroma 一次 `query(query_texts=[a, b])` 返回两条候选列表；这里照抄那个形状。
    """

    def __init__(self, by_query, distances=None):
        self.by_query = by_query
        self.distances = distances or {}
        self.calls = []

    def count(self):
        return 10

    def query(self, query_texts, n_results):
        self.calls.append(list(query_texts))
        docs, dists, metas = [], [], []
        for text in query_texts:
            found = list(self.by_query.get(text, []))
            docs.append(found)
            dists.append(list(self.distances.get(text, [0.30] * len(found))))
            metas.append([{"file_name": "a.md"} for _ in found])
        return {"documents": docs, "distances": dists, "metadatas": metas}


def _enable_fusion(monkeypatch, enabled=True, rrf_k=60, hyde_text="假设答案", gate=None):
    monkeypatch.setattr(retrieval, "FUSION_ENABLED", enabled)
    monkeypatch.setattr(retrieval, "RRF_K", rrf_k)
    # 默认用 union 写用例：多数用例要验证的是「两路都参与」，闸门口径另有专门用例
    monkeypatch.setattr(retrieval, "FUSION_GATE",
                        retrieval.GATE_UNION if gate is None else gate)
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", False)
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: hyde_text)


def test_rrf_fuse_puts_the_document_both_probes_agree_on_first():
    """RRF 的基本性质：两路都排第 1 的文档，必须排在「只有一路排第 1」的前面。

    这就是融合的全部价值 —— 单探针的「第 1 名」可能只是那一侧的口味。
    """
    fused = retrieval._rrf_fuse([["共同块", "只有HyDE"], ["共同块", "只有原问题"]], k=60)

    assert [text for text, _score in fused][:1] == ["共同块"]
    assert set(text for text, _ in fused) == {"共同块", "只有HyDE", "只有原问题"}


def test_fusion_off_queries_once_with_the_hyde_text(monkeypatch):
    """开关关掉时行为与改造前逐位一致（这是回退保命的那条路）。"""
    _enable_fusion(monkeypatch, enabled=False)
    coll = _TwoProbeCollection({"假设答案": [ROE_CHUNK]})

    _path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert coll.calls == [["假设答案"]]
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_fusion_on_queries_both_probes_in_one_call(monkeypatch):
    """两路探针合并成一次 query 调用（Chroma 会一次性算两条 embedding），不是两次往返。"""
    _enable_fusion(monkeypatch)
    coll = _TwoProbeCollection({"假设答案": [ROE_CHUNK], "净资产收益率是什么": [PE_CHUNK]})

    retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert coll.calls == [["假设答案", "净资产收益率是什么"]]


def test_fusion_recalls_a_chunk_only_the_original_question_probe_sees(monkeypatch):
    """HyDE 改写会把短问句里的关键词揉进「像文档的片段」，那一侧可能整段都超阈值；
    原问题那一侧却贴着答案。单探针在那种情况下只能退到关键词兜底，融合能直接救回来。"""
    _enable_fusion(monkeypatch)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["无关块"], "原问题里的问题": [ROE_CHUNK]},
        distances={"假设答案": [0.90], "原问题里的问题": [0.35]},
    )

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "原问题里的问题", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_VECTOR, "这仍然是向量召回，不是关键词兜底"
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_fusion_drops_chunks_far_from_both_probes(monkeypatch):
    """距离闸门仍在：两路都判「远」的块不许进 context（否则误召回会反弹）。"""
    _enable_fusion(monkeypatch)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["远块"], "量子纠缠退相干时间": ["远块"]},
        distances={"假设答案": [0.90], "量子纠缠退相干时间": [0.95]},
    )

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "量子纠缠退相干时间", coll, "完全无关的一段文字", use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert hits == []


def test_fusion_skips_the_second_probe_when_hyde_degrades_to_the_question(monkeypatch):
    """HyDE 退化（空内容/失败）时假设文本就是原问题：再查一遍是白花一次 embedding。"""
    _enable_fusion(monkeypatch, hyde_text="净资产收益率是什么")
    coll = _TwoProbeCollection({"净资产收益率是什么": [ROE_CHUNK]})

    retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert coll.calls == [["净资产收益率是什么"]]


def test_fusion_tolerates_a_collection_that_returns_one_list(monkeypatch):
    """老替身/老实现只回一条列表时不能炸：缺的那一路直接当作没召回。"""
    class _OneListCollection(_TwoProbeCollection):
        def query(self, query_texts, n_results):
            self.calls.append(list(query_texts))
            return {"documents": [[ROE_CHUNK]], "distances": [[0.30]]}

    _enable_fusion(monkeypatch)
    coll = _OneListCollection({})

    _path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_fusion_keeps_the_rerank_contract(monkeypatch):
    """融合 + 重排同时开：候选仍然不做距离预砍，打分仍然用**原问题**。"""
    seen = {}

    def fake_rerank(query, hits, top_n, min_score):
        seen["query"] = query
        seen["texts"] = [text for text, _ in hits]
        return [(hits[0][0], hits[0][1], 0.9)]

    _enable_fusion(monkeypatch)
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANK_CANDIDATES", 10)
    monkeypatch.setattr(retrieval, "RERANK_TOP_N", 1)
    monkeypatch.setattr(retrieval, "RERANK_MIN_SCORE", 0.0)
    monkeypatch.setattr(retrieval.rerank_mod, "rerank", fake_rerank)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["近块"], "原问题": ["远块"]},
        distances={"假设答案": [0.30], "原问题": [0.95]},
    )

    path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert seen["query"] == "原问题"
    assert set(seen["texts"]) == {"远块", "近块"}, "超阈值的候选也要交给重排"
    assert path == retrieval.PATH_VECTOR
    assert [text for text, _ in hits] == ["近块"]


def test_retrieval_exposes_fusion_knobs():
    """评测脚本要靠 `retrieval.FUSION_ENABLED` / `RRF_K` / `FUSION_GATE` 覆盖开关
    （漏暴露就是 AttributeError）。"""
    for name in ("FUSION_ENABLED", "RRF_K", "FUSION_GATE"):
        assert hasattr(retrieval, name), f"retrieval 缺 {name}"


def test_union_gate_keeps_a_chunk_only_one_probe_finds_close(monkeypatch):
    """union：一块只要贴着**任一**探针就算命中（这是召回侧的假设）。"""
    _enable_fusion(monkeypatch)
    monkeypatch.setattr(retrieval, "FUSION_GATE", retrieval.GATE_UNION)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["只有HyDE近"], "原问题": ["只有原问题近"]},
        distances={"假设答案": [0.30], "原问题": [0.30]},
    )

    _path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert set(text for text, _ in hits) == {"只有HyDE近", "只有原问题近"}


def test_intersect_gate_drops_chunks_that_only_one_probe_finds_close(monkeypatch):
    """intersect：两路都近才进 context —— 这是给「无关提问被强行作答」准备的精度闸门。

    无关提问往往只是**某一侧**看起来像（HyDE 生成的片段会漂向语料主题），
    要求两路同时贴近，才能把它挡在门外。
    """
    _enable_fusion(monkeypatch)
    monkeypatch.setattr(retrieval, "FUSION_GATE", retrieval.GATE_INTERSECT)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["只有HyDE近", "两路都近"], "原问题": ["两路都近", "只有原问题近"]},
        distances={"假设答案": [0.30, 0.32], "原问题": [0.33, 0.30]},
    )

    _path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert [text for text, _ in hits] == ["两路都近"]


def test_intersect_gate_drops_a_shared_chunk_that_is_far_from_one_probe(monkeypatch):
    """两路都召回了、但其中一路判它远 —— 也不算命中（阈值仍然是每路的闸门）。"""
    _enable_fusion(monkeypatch)
    monkeypatch.setattr(retrieval, "FUSION_GATE", retrieval.GATE_INTERSECT)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["共享块"], "原问题": ["共享块"]},
        distances={"假设答案": [0.30], "原问题": [0.90]},
    )

    path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert path == retrieval.PATH_KEYWORD, "全被挡下 → 与「距离全丢」同义，走关键词兜底"
    assert hits == []


def test_intersect_gate_with_one_empty_probe_does_not_wipe_the_other(monkeypatch):
    """一路什么都没召回时，intersect 不能把另一路的命中全判死 —— 那种情况等于退化单探针。"""
    _enable_fusion(monkeypatch)
    monkeypatch.setattr(retrieval, "FUSION_GATE", retrieval.GATE_INTERSECT)
    coll = _TwoProbeCollection(by_query={"原问题": [ROE_CHUNK]}, distances={"原问题": [0.30]})

    _path, hits = retrieval.hyde_retrieve_hits_with_path("原问题", coll, ROE_CHUNK, use_vector=True)

    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_intersect_gate_does_not_narrow_the_rerank_candidate_pool(monkeypatch):
    """开了重排就由跨编码器当最终裁判：闸门口径**不**先砍候选（那是没重排时的分工）。"""
    seen = {}

    def fake_rerank(query, hits, top_n, min_score):
        seen["texts"] = [text for text, _ in hits]
        return [(hits[0][0], hits[0][1], 0.9)]

    _enable_fusion(monkeypatch, gate=retrieval.GATE_INTERSECT)
    monkeypatch.setattr(retrieval, "RERANK_ENABLED", True)
    monkeypatch.setattr(retrieval, "RERANK_CANDIDATES", 10)
    monkeypatch.setattr(retrieval, "RERANK_TOP_N", 1)
    monkeypatch.setattr(retrieval, "RERANK_MIN_SCORE", 0.0)
    monkeypatch.setattr(retrieval.rerank_mod, "rerank", fake_rerank)
    coll = _TwoProbeCollection(
        by_query={"假设答案": ["近块"], "原问题": ["远块"]},
        distances={"假设答案": [0.30], "原问题": [0.95]},
    )

    retrieval.hyde_retrieve_hits_with_path("原问题", coll, "全文", use_vector=True)

    assert set(seen["texts"]) == {"近块", "远块"}, "intersect 也不许在重排之前把远块砍掉"


def test_unknown_gate_name_falls_back_to_union():
    """拼错的 FUSION_GATE 不能把检索带崩：config 里已经归一到 union，这里钉住常量。"""
    assert retrieval.GATE_UNION == "union"
    assert retrieval.GATE_INTERSECT == "intersect"


# ── 关键词兜底的切片缓存 ──

def _counting_split(monkeypatch):
    calls = []
    real = pdf_ingestion.split_files

    def counting(files, chunk_size=None, overlap=None):
        calls.append(len(files))
        return real(files, chunk_size=chunk_size, overlap=overlap)

    monkeypatch.setattr(retrieval, "split_files", counting)
    return calls


def test_pairs_cache_splits_once_for_the_same_content(monkeypatch):
    """同一条对话连问两次，兜底不该把 297KB 正文重新切一遍（真实语料实测 8.3ms/次）。"""
    calls = _counting_split(monkeypatch)
    files = [("a.md", "第一段。\n\n第二段。")]

    first = retrieval._pairs_with_meta(files, "")
    second = retrieval._pairs_with_meta([("a.md", "第一段。\n\n第二段。")], "")

    assert calls == [1], "内容没变就该命中缓存"
    assert first == second


def test_pairs_cache_invalidates_when_content_changes(monkeypatch):
    """换了正文（上传/删除文件）必须重新切：缓存键是内容哈希，不是文件数。"""
    calls = _counting_split(monkeypatch)

    retrieval._pairs_with_meta([("a.md", "旧正文。")], "")
    retrieval._pairs_with_meta([("a.md", "新正文。")], "")

    assert calls == [1, 1]


def test_pairs_cache_tracks_the_chunk_size(monkeypatch):
    """评测脚本会临时改 pdf_ingestion.CHUNK_SIZE 做对照：键里必须带上它，否则换了切片长度还吃旧缓存。"""
    calls = _counting_split(monkeypatch)
    files = [("a.md", "一句。" * 200)]
    monkeypatch.setattr(pdf_ingestion, "CHUNK_SIZE", 480)

    wide = retrieval._pairs_with_meta(files, "")
    monkeypatch.setattr(pdf_ingestion, "CHUNK_SIZE", 120)
    narrow = retrieval._pairs_with_meta(files, "")

    assert calls == [1, 1]
    assert len(narrow) > len(wide)


def test_pairs_cache_is_bounded(monkeypatch):
    """缓存必须有上限：不然多轮对话/多份上传会把正文全部留在内存里。"""
    _counting_split(monkeypatch)

    for i in range(retrieval.PAIRS_CACHE_MAX + 5):
        retrieval._pairs_with_meta([("a.md", f"第 {i} 份正文。")], "")

    assert len(retrieval._split_cache) <= retrieval.PAIRS_CACHE_MAX


def test_keyword_fallback_reuses_the_cached_lowercase_texts(monkeypatch):
    """兜底打分复用缓存的 lowercase，而不是每次把 1199 个块重新 lower 一遍（实测 1.0ms）。"""
    seen = {}
    monkeypatch.setattr(retrieval, "split_files", lambda files, **k: [(c, None) for c in CHUNKS])
    monkeypatch.setattr(retrieval, "FUSION_ENABLED", False)
    real_scores = retrieval._keyword_scores

    def spy(question, chunks, lowered=None):
        seen["lowered"] = lowered
        return real_scores(question, chunks, lowered=lowered)

    monkeypatch.setattr(retrieval, "_keyword_scores", spy)

    retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", None, "全文", use_vector=False, files=[("a.md", "x")])

    assert seen["lowered"] == [c.lower() for c in CHUNKS]


def test_keyword_scores_accepts_precomputed_lowercase():
    """预先算好的 lowercase 必须算出与现算逐位一致的分数（否则缓存会悄悄改排序）。"""
    question = "什么是净资产收益率"

    assert (retrieval._keyword_scores(question, CHUNKS)
            == retrieval._keyword_scores(question, CHUNKS, lowered=[c.lower() for c in CHUNKS]))
