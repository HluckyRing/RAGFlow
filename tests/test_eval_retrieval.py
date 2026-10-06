# -*- coding: utf-8 -*-
"""检索评测脚本的纯函数回归（P0）。

刻意不 import torch、不加载模型：仓库约定测试套件不碰 sentence-transformers，
所以这里只评测「评测逻辑本身」——归一化、命中判定、指标汇总、阈值建议、切窗统计，
以及**评测 fixture 自身的完整性**；模型相关部分只能在真跑
`python scripts/eval_retrieval.py` 时验证。

最后一组用例是这套 fixture 的保险丝：标注写错一个字，recall 就会变成假 0，
而看起来像是"检索变差了"。所以「每条 gold_phrase 必须真的能在语料里找到」
必须是一条测试，而不是靠人眼。
"""
import json
import sys
from pathlib import Path

import pytest

from scripts import eval_retrieval as ev
from src.text_utils import extract_terms

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "tests" / "fixtures" / "eval_corpus"
QUERIES_PATH = ROOT / "tests" / "fixtures" / "eval_queries.json"


# ── 归一化 ──

def test_normalize_ignores_width_case_and_whitespace():
    assert ev.normalize("ＡＢ　ｃ 1") == "abc1"
    assert ev.normalize("净资产 收益率") == "净资产收益率"


def test_normalize_handles_empty_input():
    assert ev.normalize("") == ""
    assert ev.normalize(None) == ""


def test_normalize_unifies_dash_variants():
    assert ev.normalize("营业收入 − 成本") == ev.normalize("营业收入 - 成本")
    assert ev.normalize("营业收入 — 成本") == ev.normalize("营业收入 - 成本")


# ── 语料与标注加载 ──

def test_load_corpus_reads_text_files_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("第二个", encoding="utf-8")
    (tmp_path / "a.md").write_text("第一个", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")

    got = ev.load_corpus(tmp_path)

    assert list(got) == ["a.md", "b.txt"], "只读 .md/.txt，且按文件名排序"
    assert got["a.md"] == "第一个"


def test_load_corpus_rejects_missing_or_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.load_corpus(tmp_path / "nope")
    # 目录存在但没有任何 .md/.txt 也算不可用，不能静默跑出 0 个 chunk
    with pytest.raises(FileNotFoundError):
        ev.load_corpus(tmp_path)


def _write_queries(tmp_path, data):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_queries_accepts_minimal_samples(tmp_path):
    path = _write_queries(tmp_path, [
        {"question": "净资产收益率是什么", "kind": "positive",
         "expect_files": ["01-财务指标.md"], "gold_phrases": ["净资产收益率"]},
        {"question": "量子纠缠", "kind": "negative"},
    ])

    got = ev.load_queries(path)

    assert [q["kind"] for q in got] == ["positive", "negative"]
    assert [q["id"] for q in got] == ["q01", "q02"], "没写 id 时按序号补"
    assert got[1]["gold_phrases"] == [] and got[1]["expect_files"] == []


@pytest.mark.parametrize("data, needle", [
    ({"question": "x", "kind": "negative"}, "JSON 数组"),
    ([{"kind": "positive", "expect_files": ["a.md"], "gold_phrases": ["g"]}], "question"),
    ([{"question": " ", "kind": "negative"}], "question"),
    ([{"question": "x", "kind": "positive", "gold_phrases": ["g"]}], "expect_files"),
    ([{"question": "x", "kind": "positive", "expect_files": ["a.md"]}], "gold_phrases"),
    ([{"question": "x", "kind": "maybe"}], "kind"),
])
def test_load_queries_rejects_bad_input(tmp_path, data, needle):
    path = _write_queries(tmp_path, data)

    with pytest.raises(ValueError) as excinfo:
        ev.load_queries(path)

    assert needle in str(excinfo.value)


def test_load_queries_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        ev.load_queries(_write_queries(tmp_path, []))
    assert "空" in str(excinfo.value)


# ── 命中判定 ──

def test_rank_of_first_hit_finds_position():
    chunks = ["无关内容", "这里提到净资产收益率", "另说一句"]
    assert ev.rank_of_first_hit(chunks, ["净资产收益率"]) == 2


def test_rank_of_first_hit_returns_earliest_rank():
    chunks = ["提到毛利率", "也提到净资产收益率"]
    assert ev.rank_of_first_hit(chunks, ["净资产收益率", "毛利率"]) == 1


def test_rank_of_first_hit_none_when_no_match_or_no_gold():
    assert ev.rank_of_first_hit(["无关"], ["净资产收益率"]) is None
    assert ev.rank_of_first_hit(["无关"], []) is None
    assert ev.rank_of_first_hit([], ["净资产收益率"]) is None


def test_rank_of_first_hit_ignores_width_and_whitespace():
    assert ev.rank_of_first_hit(["净资产收益率 是 15 %"], ["净资产收益率是15%"]) == 1


# ── 指标汇总 ──

def test_summarize_computes_recall_and_mrr():
    results = [
        {"kind": "positive", "rank": 1, "returned": 1},
        {"kind": "positive", "rank": 3, "returned": 3},
        {"kind": "positive", "rank": None, "returned": 2},
    ]

    got = ev.summarize(results, 3)

    assert got["recall_at_k"] == pytest.approx(2 / 3)
    assert got["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_summarize_ignores_hits_beyond_k():
    results = [{"kind": "positive", "rank": 4, "returned": 4}]

    got = ev.summarize(results, 3)

    assert got["recall_at_k"] == 0.0
    assert got["mrr"] == 0.0


def test_summarize_negative_false_positive_rate():
    results = [
        {"kind": "positive", "rank": 1, "returned": 1},
        {"kind": "negative", "rank": None, "returned": 0},
        {"kind": "negative", "rank": None, "returned": 2},
    ]

    got = ev.summarize(results, 3)

    assert got["negative_false_positives"] == 1
    assert got["negative_false_positive_rate"] == pytest.approx(0.5)


def test_summarize_handles_empty_groups():
    got = ev.summarize([], 5)
    assert got["recall_at_k"] == 0.0 and got["mrr"] == 0.0
    assert got["negative_false_positive_rate"] == 0.0


# ── 阈值建议 ──

def test_suggest_threshold_takes_midpoint_of_separation_band():
    assert ev.suggest_threshold([0.36, 0.47], [0.64, 0.70]) == pytest.approx(0.555, abs=1e-3)


def test_suggest_threshold_none_when_bands_overlap_or_missing():
    assert ev.suggest_threshold([0.60], [0.50]) is None, "分不开就不能编一个阈值出来"
    assert ev.suggest_threshold([], [0.5]) is None
    assert ev.suggest_threshold([0.4], []) is None


# ── 切窗统计 ──

def test_exceeds_model_window_reports_only_over_limit():
    long_chunk = "很长的文本" * 100
    chunks = ["短", "中等长度文本", long_chunk]

    over = ev.exceeds_model_window(chunks, 10, len)

    assert [chunk for _, chunk in over] == [long_chunk]


def test_exceeds_model_window_empty_when_all_fit():
    assert ev.exceeds_model_window(["abc", "中文"], 100, len) == []


# ── fixture 自身的完整性（标错了会让指标变成假的）──

@pytest.fixture(scope="module")
def corpus():
    return ev.load_corpus(CORPUS_DIR)


@pytest.fixture(scope="module")
def queries():
    return ev.load_queries(QUERIES_PATH)


def test_fixture_has_enough_samples(queries):
    positives = [q for q in queries if q["kind"] == "positive"]
    negatives = [q for q in queries if q["kind"] == "negative"]
    assert len(positives) >= 15, "正样本太少，指标没有统计意义"
    assert len(negatives) >= 3, "至少要有几条负样本才能看误召回"


def test_fixture_ids_are_unique(queries):
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_every_gold_phrase_really_appears_in_corpus(corpus, queries):
    """标注写错一个字 -> recall 假性变 0 -> 看起来像检索退化。这条守住 fixture 本身。"""
    haystack = ev.normalize("\n\n".join(corpus.values()))

    missing = [(q["id"], gold) for q in queries for gold in q["gold_phrases"]
               if ev.normalize(gold) not in haystack]

    assert missing == [], f"这些标注片段在语料里找不到：{missing}"


def test_expect_files_exist(corpus, queries):
    unknown = [(q["id"], name) for q in queries for name in q["expect_files"]
               if name not in corpus]
    assert unknown == [], f"标注引用了不存在的语料文件：{unknown}"


def test_negative_questions_do_not_overlap_corpus_lexically(corpus, queries):
    """负样本必须「真的不在语料里」。

    只要有一个词撞上语料，关键词兜底就会召回它，误召回率就不再是「向量阈值」的
    信号，而变成 2-gram 松散的噪声——那样这条指标就没法用来判断阈值好坏了。
    """
    haystack = ev.normalize("\n\n".join(corpus.values()))

    overlaps = []
    for q in queries:
        if q["kind"] != "negative":
            continue
        hit = [term for term in extract_terms(q["question"]) if term in haystack]
        if hit:
            overlaps.append((q["id"], hit))

    assert overlaps == [], f"负样本与语料有词面重叠：{overlaps}"


# ── 无模型环境下 main 必须安全（输入校验先于加载模型）──

def test_main_returns_2_on_bad_input_without_loading_torch(tmp_path, capsys):
    rc = ev.main([
        "--corpus", str(tmp_path / "no-such-corpus"),
        "--queries", str(QUERIES_PATH),
    ])

    assert rc == 2
    assert "输入不合法" in capsys.readouterr().out
    assert "torch" not in sys.modules, "输入校验必须发生在加载模型之前"
