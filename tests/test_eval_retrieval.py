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


def test_load_corpus_builds_the_structural_pptx_from_its_spec(tmp_path):
    """带页标记的语料不往仓库塞二进制：按 spec 现场生成 pptx，再走真实 loader 抽文本。

    这样语料内容可读可 diff，同时保证「页码」真的由 loader 输出的页标记驱动，
    而不是评测自己编一个 section 出来。
    """
    (tmp_path / "a.md").write_text("纯文本语料", encoding="utf-8")
    (tmp_path / ev.PPTX_SPEC_NAME).write_text(json.dumps({
        "file_name": "07-x.pptx",
        "slides": [
            {"title": "第一页", "body": ["第一页正文"], "notes": "备注文字"},
            {"title": "第二页", "body": ["第二页正文"]},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    got = ev.load_corpus(tmp_path)

    assert list(got) == ["07-x.pptx", "a.md"], "结构语料与文本语料一起按文件名排序"
    text = got["07-x.pptx"]
    assert "=== 第 1 页 ===" in text and "=== 第 2 页 ===" in text
    assert "第一页正文" in text and "第二页正文" in text
    assert "[备注] 备注文字" in text, "备注必须走真实 loader 的抽取路径"


def test_load_corpus_rejects_a_malformed_pptx_spec(tmp_path):
    (tmp_path / "a.md").write_text("纯文本语料", encoding="utf-8")
    (tmp_path / ev.PPTX_SPEC_NAME).write_text('{"slides": []}', encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        ev.load_corpus(tmp_path)

    assert ev.PPTX_SPEC_NAME in str(excinfo.value)


def test_load_corpus_needs_python_pptx_for_the_spec(tmp_path, monkeypatch):
    """缺 python-pptx 要吵出来 —— 静默少一个语料文件会让 section 指标变成假的「无数据」。"""
    (tmp_path / "a.md").write_text("纯文本语料", encoding="utf-8")
    (tmp_path / ev.PPTX_SPEC_NAME).write_text(json.dumps({
        "file_name": "07-x.pptx", "slides": [{"body": ["正文"]}]},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ev, "HAS_PPTX", False)

    with pytest.raises(ValueError) as excinfo:
        ev.load_corpus(tmp_path)

    assert "python-pptx" in str(excinfo.value)


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


def test_load_queries_reads_expect_sections(tmp_path):
    """页码标注是可选的：没写 = 这个样本不测页码（不是「第 0 页」）。"""
    path = _write_queries(tmp_path, [
        {"question": "窑温", "kind": "positive", "expect_files": ["a.pptx"],
         "expect_sections": ["第 3 页"], "gold_phrases": ["窑温"]},
        {"question": "净资产收益率", "kind": "positive", "expect_files": ["b.md"],
         "gold_phrases": ["净资产收益率"]},
    ])

    got = ev.load_queries(path)

    assert got[0]["expect_sections"] == ["第 3 页"]
    assert got[1]["expect_sections"] == [], "没写页码标注时补空列表"


def test_load_queries_reads_the_negative_class(tmp_path):
    """负样本分两类：`far`（远域、零词面重叠）与 `adjacent`（域内但无答案、**有**重叠）。

    两条路的失败模式不一样：远域考「会不会被阈值外的噪声拉进来」，域内考「看起来像
    有没有就作答」。混成一锅统计，会把后者的问题藏在前者里。
    """
    path = _write_queries(tmp_path, [
        {"question": "量子纠缠", "kind": "negative", "negative_class": "far"},
        {"question": "年会的奖品预算走谁的审批", "kind": "negative",
         "negative_class": "adjacent"},
        {"question": "火星探测器", "kind": "negative"},
    ])

    got = ev.load_queries(path)

    assert [q["negative_class"] for q in got] == ["far", "adjacent", "far"], \
        "没写 negative_class 的老标注按 far 处理"


def test_load_queries_rejects_an_unknown_negative_class(tmp_path):
    path = _write_queries(tmp_path, [
        {"question": "x", "kind": "negative", "negative_class": "maybe"}])

    with pytest.raises(ValueError) as excinfo:
        ev.load_queries(path)

    assert "negative_class" in str(excinfo.value)


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


def test_summarize_splits_false_positives_by_negative_class():
    """远域与域内两条路的失败模式不同，必须分开报：混在一起会把「域内乱答」藏起来。"""
    results = [
        {"kind": "negative", "rank": None, "returned": 0, "negative_class": "far"},
        {"kind": "negative", "rank": None, "returned": 0, "negative_class": "far"},
        {"kind": "negative", "rank": None, "returned": 3, "negative_class": "adjacent"},
        {"kind": "negative", "rank": None, "returned": 2, "negative_class": "adjacent"},
        {"kind": "negative", "rank": None, "returned": 0, "negative_class": "adjacent"},
    ]

    got = ev.summarize(results, 3)

    assert got["negatives"] == 5 and got["negative_false_positives"] == 2, "总数不变（向后兼容）"
    assert (got["negative_far_total"], got["negative_false_positives_far"]) == (2, 0)
    assert (got["negative_adjacent_total"], got["negative_false_positives_adjacent"]) == (3, 2)
    assert got["negative_false_positive_rate_far"] == pytest.approx(0.0)
    assert got["negative_false_positive_rate_adjacent"] == pytest.approx(2 / 3)


def test_summarize_treats_negatives_without_a_class_as_far():
    """老结果没有 negative_class 字段时按 far 归类 —— 否则老脚本的汇总会静默少一半样本。"""
    got = ev.summarize([{"kind": "negative", "rank": None, "returned": 1}], 3)

    assert got["negative_far_total"] == 1
    assert got["negative_false_positives_far"] == 1
    assert got["negative_adjacent_total"] == 0
    assert got["negative_false_positive_rate_adjacent"] is None


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
    partials = [q for q in queries if q["kind"] == "partial"]
    assert len(positives) >= 15, "正样本太少，指标没有统计意义"
    # 2026-11-05：负样本从 5 条加到 23 条 —— 5 条撑不起「误召回 0/5」这种结论
    assert len(negatives) >= 15, "负样本太少，误召回率没有统计意义"
    adjacent = [q for q in negatives if q.get("negative_class") == "adjacent"]
    assert len(adjacent) >= 5, "缺「域内但无答案」的难负样本，误召回会被系统性低估"
    far = [q for q in negatives if q.get("negative_class") == "far"]
    assert len(far) >= 5, "缺远域负样本，就分不清「模型没乱答」还是「问得太远」"
    # 2026-11-08：部分可答（一个问句里一半有答案、一半没有）是产品里最常见的形状
    assert len(partials) >= 6, "缺部分可答样本，就看不出「整段拒答」这种过度拒答"


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


def test_expect_sections_really_exist_in_their_files(corpus, queries):
    """页码标错 -> section 指标变成假 0 -> 看起来像引用坏了。和 gold_phrase 一样要上保险丝。"""
    from src.pdf_ingestion import split_files

    available = {}
    for _chunk, meta in split_files(list(corpus.items())):
        if meta.get("section"):
            available.setdefault(meta["file_name"], set()).add(meta["section"])

    missing = []
    for q in queries:
        for section in q["expect_sections"]:
            sections = set().union(*(available.get(name, set()) for name in q["expect_files"]))
            if section not in sections:
                missing.append((q["id"], section))

    assert missing == [], f"这些页码标注在对应文件的切片里找不到：{missing}"


def test_fixture_keeps_a_structural_file_so_the_section_metric_stays_live(corpus, queries):
    """没有带页标记的语料，section 指标会静默退化成「无数据」—— 这条钉住它一直有。"""
    annotated = [q for q in queries if q["expect_sections"]]
    assert annotated, "至少要有一条 expect_sections 样本，否则这个指标是空的"

    names = {name for q in annotated for name in q["expect_files"]}
    assert names, "带页码标注的样本必须同时给出 expect_files"
    for name in names:
        assert "=== 第 " in corpus[name], f"{name} 里没有页标记，页码这条路测不到"


def test_negative_questions_do_not_overlap_corpus_lexically(corpus, queries):
    """**远域**负样本必须「真的不在语料里」。

    只要有一个词撞上语料，关键词兜底就会召回它，误召回率就不再是「向量阈值」的
    信号，而变成 2-gram 松散的噪声——那样这条指标就没法用来判断阈值好坏了。
    """
    haystack = ev.normalize("\n\n".join(corpus.values()))

    overlaps = []
    for q in queries:
        if q["kind"] != "negative" or q.get("negative_class") != "far":
            continue
        hit = [term for term in extract_terms(q["question"]) if term in haystack]
        if hit:
            overlaps.append((q["id"], hit))

    assert overlaps == [], f"远域负样本与语料有词面重叠：{overlaps}"


def test_adjacent_negatives_really_are_adjacent(corpus, queries):
    """**域内但无答案**的负样本反过来必须与语料有词面重叠。

    这类样本是产品里真正的「答非所问」（用户问的话很像库里的内容，但库里没有答案）。
    如果它也零重叠，那它和远域样本没区别 —— 统计上等于高估了鲁棒性。
    """
    haystack = ev.normalize("\n\n".join(corpus.values()))

    not_adjacent = []
    for q in queries:
        if q["kind"] != "negative" or q.get("negative_class") != "adjacent":
            continue
        hit = [term for term in extract_terms(q["question"]) if term in haystack]
        if not hit:
            not_adjacent.append(q["id"])

    assert not_adjacent == [], f"这些「域内」负样本其实与语料毫无词面交集：{not_adjacent}"


def test_every_negative_says_why_it_has_no_answer(queries):
    """「域内但无答案」是个判断题，必须写清楚没有答案的理由，否则后人无法复核。"""
    missing = [q["id"] for q in queries if q["kind"] == "negative" and not q["note"].strip()]

    assert missing == [], f"这些负样本没写 note：{missing}"


# ── 部分可答样本（2026-11-08）：一个问句里一半有答案、一半没有 ──

def test_load_queries_reads_partial_samples(tmp_path):
    """partial：`answer_terms` 是可答部分的关键事实，`absent_terms` 是语料里没有的那半个问题。"""
    path = _write_queries(tmp_path, [
        {"question": "年假有几天，病假工资怎么算", "kind": "partial",
         "expect_files": ["02-考勤与假期制度.md"],
         "answer_terms": ["每年 5 天"], "absent_terms": ["病假工资"]},
    ])

    got = ev.load_queries(path)

    assert got[0]["kind"] == "partial"
    assert got[0]["answer_terms"] == ["每年 5 天"]
    assert got[0]["absent_terms"] == ["病假工资"]
    assert got[0]["negative_class"] is None


@pytest.mark.parametrize("item, needle", [
    ({"question": "x", "kind": "partial", "expect_files": ["a.md"],
      "absent_terms": ["y"]}, "answer_terms"),
    ({"question": "x", "kind": "partial", "expect_files": ["a.md"],
      "answer_terms": ["z"]}, "absent_terms"),
    ({"question": "x", "kind": "partial", "answer_terms": ["z"],
      "absent_terms": ["y"]}, "expect_files"),
])
def test_load_queries_rejects_incomplete_partial_samples(tmp_path, item, needle):
    with pytest.raises(ValueError) as excinfo:
        ev.load_queries(_write_queries(tmp_path, [item]))
    assert needle in str(excinfo.value)


def test_partial_answer_terms_really_appear_in_corpus(corpus, queries):
    """可答部分的关键事实必须真的在语料里，否则这条样本永远「答不到」，指标会变成假 0。"""
    haystack = ev.normalize("\n\n".join(corpus.values()))
    partials = [q for q in queries if q["kind"] == "partial"]

    missing = [(q["id"], term) for q in partials for term in q["answer_terms"]
               if ev.normalize(term) not in haystack]
    assert missing == [], f"这些「可答部分」的事实语料里没有：{missing}"


def test_partial_absent_terms_really_are_absent(corpus, queries):
    """不可答部分的主题词必须真的不在语料里 —— 否则就成了「其实答得上」，分类全废。"""
    haystack = ev.normalize("\n\n".join(corpus.values()))
    partials = [q for q in queries if q["kind"] == "partial"]

    present = [(q["id"], term) for q in partials for term in q["absent_terms"]
               if ev.normalize(term) in haystack]
    assert present == [], f"这些「不可答部分」的主题词其实在语料里：{present}"


# ── 无模型环境下 main 必须安全（输入校验先于加载模型）──

def test_main_returns_2_on_bad_input_without_loading_torch(tmp_path, capsys):
    rc = ev.main([
        "--corpus", str(tmp_path / "no-such-corpus"),
        "--queries", str(QUERIES_PATH),
    ])

    assert rc == 2
    assert "输入不合法" in capsys.readouterr().out
    assert "torch" not in sys.modules, "输入校验必须发生在加载模型之前"


# ── 文件归属（P2）──

def test_rank_and_file_reports_matching_file():
    hits = [("无关内容", {"file_name": "b.md"}), ("净资产收益率", {"file_name": "a.md"})]

    assert ev.rank_and_file(hits, ["净资产收益率"], ["a.md"]) == (2, True)


def test_rank_and_file_reports_mismatching_file():
    hits = [("净资产收益率", {"file_name": "b.md"})]

    assert ev.rank_and_file(hits, ["净资产收益率"], ["a.md"]) == (1, False)


def test_rank_and_file_returns_none_without_metadata():
    """老索引没有元数据：归属是「不知道」，不能算成「归属错误」。"""
    hits = [("净资产收益率", None)]

    assert ev.rank_and_file(hits, ["净资产收益率"], ["a.md"]) == (1, None)


def test_rank_and_file_returns_none_when_nothing_hit():
    hits = [("无关内容", {"file_name": "a.md"})]

    assert ev.rank_and_file(hits, ["净资产收益率"], ["a.md"]) == (None, None)


def test_summarize_reports_file_attribution_rate():
    results = [
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": True},
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": False},
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": None},   # 没元数据，跳过
    ]

    got = ev.summarize(results, 5)

    assert got["file_attribution_total"] == 2, "分母只算带了元数据的样本"
    assert got["file_attribution_hits"] == 1
    assert got["file_attribution_rate"] == pytest.approx(0.5)


def test_summarize_file_attribution_is_none_without_metadata():
    """拿不到来源就别给一个会骗人的数字。"""
    results = [{"kind": "positive", "rank": 1, "returned": 1, "file_ok": None}]

    got = ev.summarize(results, 5)

    assert got["file_attribution_rate"] is None
    assert got["file_attribution_total"] == 0


# ── section 归属（P2 延伸：页码也可核验）──

def test_rank_and_source_reports_matching_section():
    hits = [("无关内容", {"file_name": "a.pptx", "section": "第 1 页"}),
            ("窑温通常分三段缓慢抬升", {"file_name": "a.pptx", "section": "第 3 页"})]

    assert ev.rank_and_source(
        hits, ["窑温通常分三段缓慢抬升"], ["a.pptx"], ["第 3 页"]) == (2, True, True)


def test_rank_and_source_reports_wrong_section():
    hits = [("窑温通常分三段缓慢抬升", {"file_name": "a.pptx", "section": "第 1 页"})]

    assert ev.rank_and_source(
        hits, ["窑温通常分三段缓慢抬升"], ["a.pptx"], ["第 3 页"]) == (1, True, False)


def test_rank_and_source_without_section_but_annotation_expected_is_false():
    """本该校验页码、块上却没有 section（切片不再按节，或 loader 不再输出页头）必须算错。

    这里不能返回 None：判成「不知道」会让它从分母里消失，这条指标就跟着静默失效了。
    """
    hits = [("窑温通常分三段缓慢抬升", {"file_name": "a.pptx"})]

    assert ev.rank_and_source(
        hits, ["窑温通常分三段缓慢抬升"], ["a.pptx"], ["第 3 页"]) == (1, True, False)


def test_rank_and_source_is_unknown_when_sample_has_no_section_annotation():
    hits = [("净资产收益率", {"file_name": "a.md"})]

    assert ev.rank_and_source(hits, ["净资产收益率"], ["a.md"]) == (1, True, None)


def test_rank_and_source_is_unknown_without_metadata_at_all():
    """老索引没有 metadatas：文件和页码都不知道，两个都跳过统计。"""
    hits = [("净资产收益率", None)]

    assert ev.rank_and_source(
        hits, ["净资产收益率"], ["a.md"], ["第 3 页"]) == (1, None, None)


def test_rank_and_source_returns_nothing_when_nothing_hit():
    hits = [("无关内容", {"file_name": "a.md"})]

    assert ev.rank_and_source(
        hits, ["净资产收益率"], ["a.md"], ["第 3 页"]) == (None, None, None)


def test_summarize_reports_section_attribution_rate():
    results = [
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": True, "section_ok": True},
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": True, "section_ok": False},
        {"kind": "positive", "rank": 1, "returned": 1, "file_ok": True, "section_ok": None},
    ]

    got = ev.summarize(results, 5)

    assert got["section_attribution_total"] == 2, "分母只算带页码标注的样本"
    assert got["section_attribution_hits"] == 1
    assert got["section_attribution_rate"] == pytest.approx(0.5)


def test_summarize_section_attribution_is_none_without_annotation():
    results = [{"kind": "positive", "rank": 1, "returned": 1,
                "file_ok": True, "section_ok": None}]

    got = ev.summarize(results, 5)

    assert got["section_attribution_rate"] is None
    assert got["section_attribution_total"] == 0


# ── 检索路径（向量 or 关键词兜底）──

def test_summarize_counts_keyword_fallback_samples():
    """走兜底的样本数要能直接读出来：这些样本的分数是 2-gram 打分，不是距离。"""
    results = [
        {"kind": "positive", "rank": 1, "returned": 1, "path": "vector"},
        {"kind": "positive", "rank": 1, "returned": 1, "path": "keyword"},
        {"kind": "negative", "rank": None, "returned": 0, "path": "keyword"},
    ]

    got = ev.summarize(results, 5)

    assert got["keyword_fallback_samples"] == 2


def test_summarize_keyword_count_survives_missing_path():
    """老结果没有 path 字段时不能炸（summarize 是纯函数，别处也可能复用）。"""
    got = ev.summarize([{"kind": "positive", "rank": 1, "returned": 1}], 5)

    assert got["keyword_fallback_samples"] == 0


# ── HyDE：调用缓存、自检、calibrate 口径（2026-10-06 实测过静默退化）──

def test_cached_generate_calls_once_per_question():
    """同一句话只生成一次假设文本。

    两个理由：同一问题要在多个语料上各跑一遍；评测里 calibrate 与检索也要用同一份文本 ——
    不缓存既白花调用，又让同一个问题拿到不同假设文本（大模型有随机性），距离带就不可比。
    """
    calls = []

    def generate(question):
        calls.append(question)
        return f"假设：{question}"

    cached = ev.cached_generate(generate)

    assert cached("问题A") == "假设：问题A"
    assert cached("问题A") == "假设：问题A"
    assert cached("问题B") == "假设：问题B"
    assert calls == ["问题A", "问题B"]


def test_cached_generate_can_share_an_external_cache():
    """外部传入 cache 时能读到里面存了几个问题（脚本要靠它打印真实调用次数）。"""
    cache = {}
    cached = ev.cached_generate(lambda q: "假设", cache)

    cached("A")

    assert list(cache) == ["A"]


def test_hyde_self_check_detects_silent_fallback():
    """`_generate_hyde` 失败/空内容时会**静默**退回原问题。

    --hyde 必须先自检：不自检的话，跑出来的「线上口径」其实就是 no-hyde
    （2026-10-06 实测踩过一次，两种口径的距离逐位相同）。
    """
    assert ev.hyde_self_check("漏损率怎么算", lambda q: "假设性文档片段") == (
        True, "假设性文档片段")
    assert ev.hyde_self_check("漏损率怎么算", lambda q: q)[0] is False
    assert ev.hyde_self_check("漏损率怎么算", lambda q: f"  {q}  ")[0] is False, "只有空白差异也算没生效"
    assert ev.hyde_self_check("漏损率怎么算", lambda q: "")[0] is False
    assert ev.hyde_self_check("漏损率怎么算", lambda q: None)[0] is False


def test_calibrate_query_text_follows_the_retrieval_path():
    """--calibrate 的原始查询必须与检索路径用同一个 query 文本。

    `--hyde` 下若仍拿原问题直查，打出来的就是 no-hyde 的距离带，据此给的
    「建议 MAX_DISTANCE」在线上口径下不可用 —— 2026-10-06 实测两轮数字逐位相同才发现的。
    """
    assert ev.calibrate_query_text("原问题", use_hyde=False, generate=lambda q: "假设") == "原问题"
    assert ev.calibrate_query_text("原问题", use_hyde=True, generate=lambda q: "假设") == "假设"


# ── 可答性判断的评测汇总（2026-11-05：域内负样本 8/8 全被作答，靠它验收）──

def test_summarize_answerability_counts_both_sides():
    """只统计**真的判过**的样本：检索本来就返回空的负样本不该进分母，判定不可用的也不算。"""
    results = [
        {"kind": "positive", "answerable": True},
        {"kind": "positive", "answerable": False},
        {"kind": "negative", "negative_class": "far", "returned": 0},        # 没判
        {"kind": "negative", "negative_class": "far", "judged": True,
         "answerable": False},
        {"kind": "negative", "negative_class": "adjacent", "judged": True,
         "answerable": False},
        {"kind": "negative", "negative_class": "adjacent", "judged": True,
         "answerable": True},
        {"kind": "negative", "negative_class": "adjacent", "judged": True,
         "answerable": None},                                               # 判不了
    ]

    got = ev.summarize_answerability(results)

    assert (got["positive_judged"], got["positive_rejected"]) == (2, 1)
    assert (got["far_judged"], got["far_rejected"]) == (1, 1)
    assert (got["adjacent_judged"], got["adjacent_rejected"]) == (2, 1), \
        "answerable=None 的样本两边都不算"
    assert got["unknown_verdicts"] == 1
    assert got["positive_rejected_rate"] == pytest.approx(0.5)


def test_summarize_answerability_is_empty_when_nothing_was_judged():
    got = ev.summarize_answerability([{"kind": "positive", "answerable": None}])

    assert got["positive_judged"] == 0
    assert got["positive_rejected_rate"] is None


def test_build_context_uses_the_same_source_labels_as_the_live_path():
    """可答性判断必须看线上真正喂给模型的形状（带【来源：…】），否则量的不是同一件事。"""
    context = ev.build_context([("ROE 是净利润/股东权益", {"file_name": "a.md"}),
                                ("第二块", None)])

    assert context.startswith("【来源：a.md】")
    assert "【来源：未标注】" in context
    assert "ROE 是净利润/股东权益" in context


# ── 精排分数标定 ──

def test_score_band_summarizes_and_handles_empty():
    assert ev.score_band([]) == {"n": 0, "min": None, "median": None, "max": None}

    got = ev.score_band([0.2, 0.9, 0.5])

    assert got["n"] == 3
    assert got["min"] == pytest.approx(0.2)
    assert got["median"] == pytest.approx(0.5)
    assert got["max"] == pytest.approx(0.9)


def test_suggest_score_threshold_takes_midpoint_when_bands_separate():
    """与距离相反：分数越大越相关，所以是 max(负) < min(正) 才算干净。"""
    assert ev.suggest_score_threshold([0.8, 0.9], [0.2, 0.4]) == pytest.approx(0.6)


def test_suggest_score_threshold_none_when_overlapping_or_missing():
    assert ev.suggest_score_threshold([0.5, 0.9], [0.4, 0.6]) is None, "重叠就没有可用的阈值"
    assert ev.suggest_score_threshold([], [0.4]) is None
    assert ev.suggest_score_threshold([0.9], []) is None


# ── HyDE 文本落盘缓存：让「开/关某个开关」的两轮跑在**同一批**假设文本上 ──

def test_save_and_load_hyde_cache_roundtrip(tmp_path):
    path = tmp_path / "hyde.json"

    ev.save_hyde_cache(str(path), {"漏损率怎么算": "假设片段"})

    assert ev.load_hyde_cache(str(path)) == {"漏损率怎么算": "假设片段"}


def test_load_hyde_cache_tolerates_missing_or_broken_file(tmp_path):
    """缓存文件不存在、不是 JSON、不是对象 —— 一律当成空缓存，不能把评测带崩。"""
    missing = tmp_path / "nope.json"
    assert ev.load_hyde_cache(str(missing)) == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{不是 json", encoding="utf-8")
    assert ev.load_hyde_cache(str(broken)) == {}

    not_object = tmp_path / "list.json"
    not_object.write_text("[1, 2, 3]", encoding="utf-8")
    assert ev.load_hyde_cache(str(not_object)) == {}


def test_saved_hyde_cache_makes_the_second_run_free_and_identical(tmp_path):
    """这是对比实验的地基：第二轮直接复用第一轮的假设文本，不再调大模型。

    没有它就只能拿两批「大模型随机生成的不同文本」比配置，MRR 的差异分不清是
    开关的作用还是 HyDE 的随机性（2026-10-06 实测同一份配置连跑两次 MRR 0.972 / 0.958）。
    """
    path = tmp_path / "hyde.json"
    first = ev.cached_generate(lambda q: f"假设：{q}")
    first("问题A")

    ev.save_hyde_cache(str(path), {"问题A": first("问题A")})

    def boom(question):                       # 第二轮绝不该再调大模型
        raise AssertionError("缓存命中时不应调用大模型")

    second = ev.cached_generate(boom, ev.load_hyde_cache(str(path)))

    assert second("问题A") == "假设：问题A"
