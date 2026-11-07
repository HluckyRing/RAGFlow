#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索质量离线评测（P0）：把「检索到底变好了没有」变成可重复的数字。

为什么需要它
------------
`MAX_DISTANCE` 是实测标定出来的（相关查询 l2 距离 0.3625~0.4722、无关查询
0.6402~0.6981，取分离带中点 0.55），而「CHUNK_SIZE=600 字是否超过了 embedding
的 512 token 窗口」此前只是推算。没有可重复的评测集，任何检索改动都只能靠感觉，
也说不清是变好了还是变差了。

设计要点
--------
- **默认 `--no-hyde`**：把 `retrieval._generate_hyde` 换成恒等函数，query 就是原问题。
  这样评测可重复、不花 API 额度，也把变量收敛到「切片 / 阈值 / 排序」本身；
  要测真实 HyDE 路径请显式加 `--hyde`。
- **默认用内存向量库**（`chromadb.EphemeralClient`）：绝不碰真实的 `chroma_db/`。
- 语料按文件名排序拼接成一个 `full_text`、建一个 collection，模拟「一个对话里多个文件」
  的真实形态（真实项目也是所有文件进同一个 collection）。
- 标注按**命中片段文本**而不是 chunk id：换切片后 chunk id 必然全变，
  按文本标注才能让改动前后可比。
- `--calibrate` 会额外对每条样本做一次原始向量查询（不经阈值过滤）来收集距离分布，
  据此给出 `MAX_DISTANCE` 的建议值。
- **带页标记的语料不存二进制**：语料目录里放一份文本 spec `eval_pptx.json`，评测现场用
  python-pptx 生成 pptx 再交给真实 `loaders.load_pptx` 抽文本 —— 页码因此由 loader 真正
  输出的 `=== 第 N 页 ===` 驱动，而不是评测自己编的。标注里的可选字段 `expect_sections`
  就是 section 级指标的输入。

用法
----
    python scripts/eval_retrieval.py                        # 默认 no-hyde，打印指标
    python scripts/eval_retrieval.py --calibrate --verbose   # 附带距离分布与逐条明细
    python scripts/eval_retrieval.py --hyde                 # 走真实 HyDE（花 API 额度）
    python scripts/eval_retrieval.py --json out.json        # 结果另存 JSON

退出码：0 正常；2 输入（语料 / 标注）不合法。
"""
import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS = ROOT / "tests" / "fixtures" / "eval_corpus"
DEFAULT_QUERIES = ROOT / "tests" / "fixtures" / "eval_queries.json"

TEXT_SUFFIXES = (".md", ".txt")
VALID_KINDS = ("positive", "negative")
# 负样本的两种口径（见 docs/retrieval-baseline-2026-10-06.md「负样本分两类」）：
# - far：与语料零词面重叠的远域提问，考「会不会被阈值外的噪声拉进来」；
# - adjacent：域内但语料里没有答案，考「看起来像就作答」——产品里真正的答非所问。
VALID_NEGATIVE_CLASSES = ("far", "adjacent")
# 带页标记的语料不放二进制：目录里放这份 spec，评测现场生成 pptx 再走真实 loader 抽文本。
PPTX_SPEC_NAME = "eval_pptx.json"
# 软依赖探测（与 loaders 的 HAS_PPTX 一个套路）：缺了要吵，不能静默少一个语料文件。
HAS_PPTX = importlib.util.find_spec("pptx") is not None

# NFKC 不管的破折号 / 连字符与特殊空格，统一成 ASCII：
# 避免「看起来一样但其实不是同一个字符」导致标注假性未命中。
_PUNCT_TABLE = str.maketrans({
    "\u2212": "-",   # − MINUS SIGN
    "\u2013": "-",   # – EN DASH
    "\u2014": "-",   # — EM DASH
    "\u00a0": " ",   # 不换行空格
    "\u3000": " ",   # 全角空格
})


def normalize(text):
    """归一化到可比对的形式：NFKC → 统一破折号/空格 → 去掉所有空白 → 小写。

    去掉空白是刻意的：语料里写「100 元」、标注里写「100元」应该算命中。
    """
    if not text:
        return ""
    unified = unicodedata.normalize("NFKC", text.translate(_PUNCT_TABLE))
    return re.sub(r"\s+", "", unified).lower()


def _build_pptx_bytes(spec):
    """按 spec 现场生成一个 pptx，返回可读的 BytesIO。

    为什么不把 .pptx 直接提交进仓库：二进制在 git 里不可读、也不可 diff，改一句话就要
    重新生成一个 blob。存成 JSON 既能 review，又能让评测真的走一遍 `loaders.load_pptx`
    （生成的字节 → 真实抽取），页码才不是评测自己编出来的。
    """
    if not HAS_PPTX:
        raise ValueError(
            f"{PPTX_SPEC_NAME} 需要 python-pptx 才能生成带页标记的语料：pip install python-pptx")
    import io

    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    blank = prs.slide_layouts[6]        # 空白版式：文字都用文本框显式摆放，不依赖模板占位符序号
    for slide_spec in spec["slides"]:
        slide = prs.slides.add_slide(blank)
        title = str(slide_spec.get("title") or "").strip()
        if title:
            box = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.0), Inches(0.9))
            box.text_frame.text = title
        # 一行正文 = 一个段落：loader 按段落逐条收集，gold_phrase 必须落在同一段里
        body = [str(line).strip() for line in (slide_spec.get("body") or []) if str(line).strip()]
        if body:
            box = slide.shapes.add_textbox(Inches(0.6), Inches(1.5), Inches(8.0), Inches(3.0))
            box.text_frame.text = body[0]
            for line in body[1:]:
                box.text_frame.add_paragraph().text = line
        notes = str(slide_spec.get("notes") or "").strip()
        if notes:
            slide.notes_slide.notes_text_frame.text = notes

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf


def _add_structural_corpus(corpus_dir, files):
    """把目录里的结构语料 spec 也变成一份语料（本体是抽取后的文本）。"""
    spec_path = Path(corpus_dir) / PPTX_SPEC_NAME
    if not spec_path.is_file():
        return
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"{PPTX_SPEC_NAME} 读不出来：{e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"{PPTX_SPEC_NAME} 必须是 JSON 对象")
    name = str(spec.get("file_name") or "").strip()
    if not name:
        raise ValueError(f"{PPTX_SPEC_NAME} 缺少 file_name")
    slides = spec.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError(f"{PPTX_SPEC_NAME} 的 slides 必须是非空数组")

    from src.loaders import load_pptx          # 延迟导入：只有真有结构语料时才需要
    files[name] = load_pptx(_build_pptx_bytes(spec))


def load_corpus(corpus_dir):
    """读目录下的语料，返回 {文件名: 抽取文本}，按文件名排序。

    - `.md/.txt` 直接读原文；
    - `eval_pptx.json`（`PPTX_SPEC_NAME`）按 spec 现场生成 pptx，再交给真实
      `loaders.load_pptx` 抽文本 —— 语料里因此带上了 `=== 第 N 页 ===` 页标记，
      section 级指标才有东西可量。
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"评测语料目录不存在：{corpus_dir}")
    files = {}
    for path in sorted(corpus_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files[path.name] = path.read_text(encoding="utf-8")
    if not files:
        raise FileNotFoundError(
            f"评测语料目录里没有任何 {'/'.join(TEXT_SUFFIXES)} 文件：{corpus_dir}")
    _add_structural_corpus(corpus_dir, files)
    return {name: files[name] for name in sorted(files)}


def load_queries(path):
    """读评测标注并校验。

    正样本必须给出 `gold_phrases`（命中片段）与 `expect_files`（预期来自哪个语料文件）；
    负样本表示「语料里根本没有答案」，预期检索返回空。
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("评测标注必须是 JSON 数组")

    queries = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 条标注不是 JSON 对象")
        kind = item.get("kind")
        if kind not in VALID_KINDS:
            raise ValueError(f"第 {i} 条标注 kind 非法：{kind!r}（只能是 {VALID_KINDS}）")
        question = str(item.get("question") or "").strip()
        if not question:
            raise ValueError(f"第 {i} 条标注缺少 question")
        gold_phrases = [str(g) for g in (item.get("gold_phrases") or [])]
        expect_files = [str(f) for f in (item.get("expect_files") or [])]
        # 页码/工作表标注是可选的：没写 = 这个样本不测 section（不进 section 指标的分母）
        expect_sections = [str(s) for s in (item.get("expect_sections") or [])]
        negative_class = str(item.get("negative_class") or "far").strip().lower()
        if kind == "positive":
            if not gold_phrases:
                raise ValueError(f"第 {i} 条正样本缺少 gold_phrases")
            if not expect_files:
                raise ValueError(f"第 {i} 条正样本缺少 expect_files")
            # 正样本没有「负样本类别」：留空而不是沿用 far，免得下游按类分组时把它也算进去
            negative_class = None
        elif negative_class not in VALID_NEGATIVE_CLASSES:
            raise ValueError(
                f"第 {i} 条负样本 negative_class 非法：{negative_class!r}"
                f"（只能是 {VALID_NEGATIVE_CLASSES}）")
        queries.append({
            "id": str(item.get("id") or f"q{i:02d}"),
            "question": question,
            "kind": kind,
            "negative_class": negative_class,
            "expect_files": expect_files,
            "expect_sections": expect_sections,
            "gold_phrases": gold_phrases,
            "note": str(item.get("note") or ""),
        })

    if not queries:
        raise ValueError("评测标注是空的")
    return queries


def rank_and_source(hits, gold_phrases, expect_files, expect_sections=None):
    """返回 (首个命中位次, 文件归属, 节归属)。

    归属只在命中块真的带了来源元数据时才有意义，所以「不知道」一律返回 None，
    由上层跳过统计，而不是把「不知道」算成「归属错误」：

    - 整块没有 metadatas（P2 之前建的老索引）→ 文件和节都是 None；
    - 这个样本没标 `expect_sections`（大多数 .md/.docx 语料本来就没有页码）
      → section 是 None。

    唯一例外：**标了页码、块上却没有 section**，那是「该有的页码丢了」（切片不再按节，
    或 loader 不再输出 `=== 第 N 页 ===`），必须算 False —— 判成 None 会让这条指标
    跟着一起静默失效，那正是要防的事。
    """
    needles = [n for n in (normalize(g) for g in gold_phrases) if n]
    if not needles:
        return None, None, None
    for rank, (text, meta) in enumerate(hits, 1):
        haystack = normalize(text)
        if not any(needle in haystack for needle in needles):
            continue
        if not meta:
            return rank, None, None
        name = meta.get("file_name")
        file_ok = None if (not name or not expect_files) else (name in expect_files)
        if not expect_sections:
            return rank, file_ok, None
        return rank, file_ok, meta.get("section") in expect_sections
    return None, None, None


def rank_and_file(hits, gold_phrases, expect_files):
    """兼容既有调用：只要 (位次, 文件归属)。"""
    rank, file_ok, _section_ok = rank_and_source(hits, gold_phrases, expect_files)
    return rank, file_ok


def rank_of_first_hit(chunks, gold_phrases):
    """第一个命中任一 gold_phrase 的 chunk 位次（1 起）；都没命中返回 None。"""
    needles = [n for n in (normalize(g) for g in gold_phrases) if n]
    if not needles:
        return None
    for rank, chunk in enumerate(chunks, 1):
        haystack = normalize(chunk)
        for needle in needles:
            if needle in haystack:
                return rank
    return None


def build_context(hits):
    """把命中装配成与线上同款的「【来源：…】+ 正文」背景。

    可答性判断必须看**线上真正喂给模型的形状**（带来源标签），否则量出来的不是同一件事。
    """
    from src.llm import _format_source_block

    return "\n\n".join(_format_source_block(text, meta) for text, meta in hits)


def summarize_answerability(results):
    """汇总可答性判断的效果：正样本误杀多少、两类负样本各拦下多少。

    只统计**真的判过**的样本（`answerable` 不为 None）：检索本来就返回空的负样本不需要判，
    判定不可用（None）的也不能算进任何一边。
    """
    def _group(kind, cls=None):
        rows = [r for r in results
                if r["kind"] == kind and r.get("answerable") is not None
                and (cls is None or (r.get("negative_class") or "far") == cls)]
        rejected = [r for r in rows if r["answerable"] is False]
        return len(rows), len(rejected)

    pos_total, pos_rejected = _group("positive")
    far_total, far_rejected = _group("negative", "far")
    adj_total, adj_rejected = _group("negative", "adjacent")
    unknown = sum(1 for r in results if r.get("answerable") is None and r.get("judged"))
    return {
        "positive_judged": pos_total,
        "positive_rejected": pos_rejected,
        "positive_rejected_rate": (pos_rejected / pos_total) if pos_total else None,
        "far_judged": far_total,
        "far_rejected": far_rejected,
        "adjacent_judged": adj_total,
        "adjacent_rejected": adj_rejected,
        "unknown_verdicts": unknown,
    }


def score_band(scores):
    """精排分数的分数带（与距离带对称，只是**越大越相关**）。"""
    values = [float(s) for s in scores]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": round(min(values), 4),
            "median": round(statistics.median(values), 4), "max": round(max(values), 4)}


def suggest_score_threshold(positive_scores, negative_scores):
    """从精排分数里取分离带中点当 `RERANK_MIN_SCORE` 建议；分不开时返回 None。

    与距离相反：分数越大越相关，所以只有 `max(负) < min(正)` 才是干净的分隔带。
    负样本用它的**最高分**（最容易被误召回的那条），正样本用命中块自己的分数。
    """
    pos = [float(s) for s in positive_scores]
    neg = [float(s) for s in negative_scores]
    if not pos or not neg:
        return None
    lower, upper = max(neg), min(pos)
    if lower >= upper:
        return None
    return round((lower + upper) / 2.0, 4)


def summarize(results, k):
    """汇总指标。results 每条形如 {'kind', 'rank', 'returned', 'file_ok', 'section_ok'}。

    - recall@k：正样本里命中片段出现在前 k 个 chunk 的比例
    - MRR：首个命中位次倒数的均值（超出 k 的算 0）
    - 文件 / 节归属正确率：命中块的文件、页码是否与标注一致
    - 负样本误召回率：负样本里「检索返回了非空结果」的比例
    """
    positives = [r for r in results if r["kind"] == "positive"]
    negatives = [r for r in results if r["kind"] == "negative"]

    hit_flags = [r["rank"] is not None and r["rank"] <= k for r in positives]
    reciprocal = [(1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= k) else 0.0
                  for r in positives]
    false_positives = [r for r in negatives if r.get("returned")]

    def _negative_split(cls):
        group = [r for r in negatives if (r.get("negative_class") or "far") == cls]
        bad = [r for r in group if r.get("returned")]
        return len(group), len(bad)
    # 归属只在命中块真的带了元数据时才统计：老索引没有 metadatas 时 file_ok 是 None，
    # 不能把「不知道」算成「错了」，否则这个指标会骗人。section 同理（还得有页码标注）。
    attributed = [r for r in positives if r.get("file_ok") is not None]
    attributed_ok = [r for r in attributed if r["file_ok"]]
    sectioned = [r for r in positives if r.get("section_ok") is not None]
    sectioned_ok = [r for r in sectioned if r["section_ok"]]
    # 路径标签延迟导入：别让评测模块在导入阶段就把 chromadb 拉进来（输入校验要能先跑）。
    # 用同一个常量而不是抄字面量，免得两边哪天跑偏。
    from src.retrieval import PATH_KEYWORD

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "hits_at_k": sum(hit_flags),
        "recall_at_k": (sum(hit_flags) / len(positives)) if positives else 0.0,
        "mrr": (sum(reciprocal) / len(reciprocal)) if reciprocal else 0.0,
        "file_attribution_hits": len(attributed_ok),
        "file_attribution_total": len(attributed),
        "file_attribution_rate": (len(attributed_ok) / len(attributed)) if attributed else None,
        "section_attribution_hits": len(sectioned_ok),
        "section_attribution_total": len(sectioned),
        "section_attribution_rate": (len(sectioned_ok) / len(sectioned)) if sectioned else None,
        # 向量全丢、最后靠关键词兜底捞回来的样本数（含负样本）：这些样本的分数是
        # 2-gram 词面打分而不是向量距离，混在一起看会误判阈值的好坏。
        "keyword_fallback_samples":
            sum(1 for r in results if r.get("path") == PATH_KEYWORD),
        "negative_false_positives": len(false_positives),
        "negative_false_positive_rate":
            (len(false_positives) / len(negatives)) if negatives else 0.0,
        # 远域 / 域内分开报：前者考阈值外的噪声，后者考「看起来像就作答」
        "negative_far_total": _negative_split("far")[0],
        "negative_false_positives_far": _negative_split("far")[1],
        "negative_false_positive_rate_far": (
            _negative_split("far")[1] / _negative_split("far")[0]
            if _negative_split("far")[0] else None),
        "negative_adjacent_total": _negative_split("adjacent")[0],
        "negative_false_positives_adjacent": _negative_split("adjacent")[1],
        "negative_false_positive_rate_adjacent": (
            _negative_split("adjacent")[1] / _negative_split("adjacent")[0]
            if _negative_split("adjacent")[0] else None),
    }


def suggest_threshold(positive_distances, negative_distances):
    """从两组距离里取分离带中点当阈值建议；分不开时返回 None。

    参数是「每条样本的最佳距离」：正样本用命中块的距离（越小越相关），
    负样本用最小距离（越小越容易被误召回）。只有 max(正) < min(负)
    才存在干净的分隔带，此时中点是可用阈值。
    """
    pos = [float(d) for d in positive_distances if d is not None]
    neg = [float(d) for d in negative_distances if d is not None]
    if not pos or not neg:
        return None
    lower, upper = max(pos), min(neg)
    if lower >= upper:
        return None
    return round((lower + upper) / 2.0, 4)


def exceeds_model_window(chunks, max_seq_length, encode_len):
    """返回 [(token 长度, chunk)]，只含超过 max_seq_length 的块。

    `encode_len` 注入进来是为了能在不加载模型的情况下单测这段统计逻辑。
    """
    over = []
    for chunk in chunks:
        length = encode_len(chunk)
        if length > max_seq_length:
            over.append((length, chunk))
    return over


def cached_generate(generate, cache=None):
    """把假设文本生成函数包一层「按问题缓存」，返回可直接替换 `retrieval._generate_hyde` 的函数。

    两个作用：
    - 同一问题只调一次大模型（评测里一个问题会在检索路径与 `--calibrate` 原始查询里各用一次，
      跨语料还会重复用）；
    - 保证检索路径与 `--calibrate` 量的是**同一个 query 文本**，距离带才可比。
    """
    cache = {} if cache is None else cache

    def cached(question):
        if question not in cache:
            cache[question] = generate(question)
        return cache[question]

    return cached


def load_hyde_cache(path):
    """读回上一轮落盘的假设文本。

    文件不存在、不是 JSON、不是对象时一律当成空缓存 —— 缓存坏掉不该让整轮评测失败。
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_hyde_cache(path, cache):
    """把假设文本落盘，让「换个开关再跑一轮」用**同一批**文本。

    否则两轮的差异里混着大模型的随机性（2026-10-06 实测同一份配置连跑两次
    MRR 分别是 0.972 与 0.958），根本分不清是开关的作用还是 HyDE 抽风。
    """
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def hyde_self_check(question, generate):
    """`--hyde` 自检：返回 `(是否真的生成了不一样的文本, 生成的文本)`。

    `_generate_hyde` 在失败或空内容时会**静默**退回原问题（2026-10-06 实测：`deepseek-flash`
    是推理模型，写死的 `max_tokens` 被 reasoning 吃光、`content` 是空串）。不自检的话，跑出来的
    「HyDE 口径」其实就是 no-hyde，而且两种口径的数字会逐位相同。
    """
    text = str(generate(question) or "").strip()
    # 空文本也算没生效：`_generate_hyde` 正是用它表示「降级了」
    return bool(text) and text != str(question).strip(), text


def calibrate_query_text(question, use_hyde, generate):
    """`--calibrate` 的原始查询要用与检索路径**同一个** query 文本。

    `--hyde` 下必须是那句假设文本：拿原问题直查得到的是 no-hyde 的距离带，据此给出的
    「建议 MAX_DISTANCE」在线上口径下不可用（2026-10-06 实测两轮数字逐位相同才发现）。
    """
    return generate(question) if use_hyde else question


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="检索质量离线评测（默认不花 API 额度、不碰真实向量库）")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                        help=f"评测语料目录，默认 {DEFAULT_CORPUS}")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES),
                        help=f"评测标注 JSON，默认 {DEFAULT_QUERIES}")
    parser.add_argument("--top-k", type=int, default=5,
                        help="每个问题召回的块数上限，默认 5")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="切片长度，默认用 src.config.CHUNK_SIZE")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="切片重叠，默认用 src.config.CHUNK_OVERLAP")
    parser.add_argument("--hyde", action="store_true",
                        help="走真实 HyDE（会调用大模型、花 API 额度）")
    parser.add_argument("--calibrate", action="store_true",
                        help="额外做一次原始向量查询，报告距离分布与建议阈值")
    parser.add_argument("--verbose", action="store_true", help="打印逐条明细")
    parser.add_argument("--json", default=None, help="把结果另存到该 JSON 文件")
    parser.add_argument("--rerank", action="store_true",
                        help="开启跨编码器精排（首次会下载约 1.1GB 权重）")
    parser.add_argument("--rerank-min-score", type=float, default=None,
                        help="覆盖 RERANK_MIN_SCORE（按 --calibrate 的实测分数带标定）")
    parser.add_argument("--fusion", action="store_true",
                        help="开启双路融合（原问题 + HyDE 各当一次向量探针，按 RRF 合并名次）")
    parser.add_argument("--fusion-gate", default=None, choices=["union", "intersect"],
                        help="融合闸门：union=任一路近就算近；intersect=两路都得近")
    parser.add_argument("--hyde-cache", default=None,
                        help="把生成的假设文本落盘到该 JSON；已存在就直接复用"
                             "（对比不同开关时保证两轮用的是同一批文本）")
    parser.add_argument("--answerability", action="store_true",
                        help="对「检索返回了内容」的样本再跑一次可答性判断（会调大模型）")
    parser.add_argument("--answerability-chunks", type=int, default=0,
                        help="判定只看前 N 个命中块（0 = 全部，按线上口径）")
    args = parser.parse_args(argv)

    # 输入校验放在最前面：语料 / 标注不合法时直接返回 2，不必先把模型加载起来。
    # （测试也靠这一条：无模型环境下 main 必须能安全返回 2。）
    try:
        corpus = load_corpus(args.corpus)
        queries = load_queries(args.queries)
    except (OSError, ValueError) as e:
        print(f"输入不合法：{e}")
        return 2

    import chromadb

    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, MAX_DISTANCE
    import src.pdf_ingestion as pdf_ingestion
    from src.pdf_ingestion import split_files, split_text
    import src.retrieval as retrieval

    chunk_size = CHUNK_SIZE if args.chunk_size is None else args.chunk_size
    chunk_overlap = CHUNK_OVERLAP if args.chunk_overlap is None else args.chunk_overlap
    if chunk_size <= 0:
        print(f"输入不合法：--chunk-size 必须是正整数，实际 {chunk_size}")
        return 2
    if args.top_k <= 0:
        print(f"输入不合法：--top-k 必须是正整数，实际 {args.top_k}")
        return 2

    # 「关键词兜底」里的 split_text 读的是 pdf_ingestion 的模块级常量，这里一并覆盖，
    # 保证向量路径与兜底路径用的是同一套切片 —— 否则 --chunk-size 600 的对比是失真的。
    if args.chunk_size is not None:
        pdf_ingestion.CHUNK_SIZE = args.chunk_size
    if args.chunk_overlap is not None:
        pdf_ingestion.CHUNK_OVERLAP = args.chunk_overlap

    if not args.hyde:
        # 评测模式：HyDE 恒等。可重复、不花额度，也不让大模型的随机性盖过要测的变量。
        retrieval._generate_hyde = lambda question: question  # noqa: SLF001
        print("检索模式：no-hyde（query = 原问题，不调大模型）")
    else:
        # 每个问题只生成一次假设文本：检索路径与 --calibrate 共用同一份，既不多花一倍调用，
        # 也让两条路量的是同一个 query。给了 --hyde-cache 就跨轮复用（对比实验的地基）。
        hyde_cache = load_hyde_cache(args.hyde_cache) if args.hyde_cache else {}
        retrieval._generate_hyde = cached_generate(retrieval._generate_hyde, hyde_cache)  # noqa: SLF001
        # 先自检再跑：HyDE 失效时会静默退回原问题，那样整轮「HyDE 口径」其实是 no-hyde。
        ok, hyde_text = hyde_self_check(queries[0]["question"], retrieval._generate_hyde)
        if not ok:
            print("错误：HyDE 没有真正生效 —— 生成的文本与原问题相同，这一轮不是线上口径，已中止。")
            print("   排查：上面有没有「HyDE 返回空内容 / 生成失败」告警？"
                  "推理模型的 max_tokens 被 reasoning 吃光就会这样。")
            return 2
        cached_before = len(hyde_cache)
        print(f"检索模式：HyDE（真实调用大模型，会花 API 额度）；"
              f"自检通过（首句生成 {len(hyde_text)} 字）；"
              f"假设文本缓存已有 {cached_before} 条")

    if args.fusion:
        # 评测要能明确对比「开/关融合」，所以这里直接按开关覆盖模块级常量
        retrieval.FUSION_ENABLED = True
        if args.fusion_gate is not None:
            retrieval.FUSION_GATE = args.fusion_gate
        print(f"双路融合：开启（原问题 + HyDE 各一路探针，RRF k={retrieval.RRF_K}，"
              f"闸门={retrieval.FUSION_GATE}）")
    else:
        retrieval.FUSION_ENABLED = False

    if args.rerank:
        # 评测要能明确对比「开/关精排」，所以这里直接按开关覆盖模块级常量
        retrieval.RERANK_ENABLED = True
        if args.rerank_min_score is not None:
            retrieval.RERANK_MIN_SCORE = args.rerank_min_score
        print(f"精排：开启（{retrieval.RERANK_MODEL}；候选 {retrieval.RERANK_CANDIDATES} "
              f"→ 保留 {retrieval.RERANK_TOP_N}；分数下限 {retrieval.RERANK_MIN_SCORE}）")
    else:
        retrieval.RERANK_ENABLED = False

    judge = None
    if args.answerability:
        from src import answerability as answerability_mod

        if answerability_mod.client is None:
            print("错误：--answerability 需要能调用大模型（缺 API_KEY），已中止。")
            return 2
        judge = answerability_mod.judge_answerable
        print(f"可答性判断：开启（max_tokens={answerability_mod.MAX_TOKENS}，关思考）")

    files = [(name, corpus[name]) for name in sorted(corpus)]
    full_text = "\n\n".join(text for _, text in files)
    # P2：按文件、按节切片（与生产一致），并把来源写进 metadatas —— 评测要能量到
    # 「命中的块是不是来自预期文件」，否则引用准不准只能靠肉眼读回答。
    pairs = split_files(files, chunk_size=chunk_size, overlap=chunk_overlap)
    chunks = [chunk for chunk, _ in pairs]
    if not chunks:
        print("输入不合法：语料切片后没有任何 chunk（文件内容为空？）")
        return 2

    embedding_fn = retrieval._get_embedding_fn()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="eval_kb", embedding_function=embedding_fn)
    collection.add(documents=chunks, ids=[f"eval_c{i}" for i in range(len(chunks))],
                   metadatas=[retrieval.chroma_meta(meta) for _, meta in pairs])

    positive_count = sum(1 for q in queries if q["kind"] == "positive")
    negative_count = len(queries) - positive_count
    print(f"语料：{len(corpus)} 个文件 / {len(chunks)} 个 chunk"
          f"（chunk_size={chunk_size}, overlap={chunk_overlap}）")
    print(f"样本：{positive_count} 正 + {negative_count} 负，top_k={args.top_k}")

    results = []
    positive_distances, negative_distances = [], []
    positive_scores, negative_scores = [], []
    judge_seconds = []
    started = time.perf_counter()
    for query in queries:
        question = query["question"]
        path, hits = retrieval.hyde_retrieve_hits_with_path(
            question, collection, full_text, use_vector=True, top_k=args.top_k, files=files)

        if query["kind"] == "positive":
            rank, file_ok, section_ok = rank_and_source(
                hits, query["gold_phrases"], query["expect_files"], query["expect_sections"])
            results.append({"id": query["id"], "kind": "positive", "rank": rank,
                            "returned": len(hits), "file_ok": file_ok,
                            "section_ok": section_ok, "path": path})
        else:
            results.append({"id": query["id"], "kind": "negative",
                            "rank": None, "returned": len(hits), "path": path,
                            "negative_class": query["negative_class"]})

        if judge is not None and hits:
            # 只对「检索返回了内容」的样本判：检索本来就空的不需要判（线上也走不到这一步）
            seen_hits = hits if args.answerability_chunks <= 0 else hits[:args.answerability_chunks]
            t0 = time.perf_counter()
            verdict = judge(question, build_context(seen_hits))
            judge_seconds.append(time.perf_counter() - t0)
            results[-1]["judged"] = True
            results[-1]["answerable"] = verdict

        if args.calibrate:
            # 原始查询：不经阈值过滤，用来看真实距离分布。
            # 必须与检索路径用同一个 query 文本（--hyde 下就是那句假设文本，且已在缓存里，
            # 不会多花一次调用）—— 否则打出来的是 no-hyde 的距离带。
            raw = collection.query(
                query_texts=[calibrate_query_text(question, args.hyde, retrieval._generate_hyde)],
                n_results=min(args.top_k, len(chunks)))
            docs = (raw.get("documents") or [[]])[0]
            dists = (raw.get("distances") or [[]])[0]
            if query["kind"] == "positive":
                gold_hits = [dist for doc, dist in zip(docs, dists)
                             if rank_of_first_hit([doc], query["gold_phrases"]) == 1]
                if gold_hits:
                    positive_distances.append(min(gold_hits))
            elif dists:
                negative_distances.append(min(dists))

        if args.calibrate and retrieval.RERANK_ENABLED:
            # 精排分数的分离带：距离分不开时靠它决定 RERANK_MIN_SCORE。
            # 另取一次原始候选（不经 min_score 过滤）、并用**原问题**打分。
            raw = collection.query(
                query_texts=[calibrate_query_text(question, args.hyde, retrieval._generate_hyde)],
                n_results=min(retrieval.RERANK_CANDIDATES, len(chunks)))
            cand = (raw.get("documents") or [[]])[0]
            cand_scores = retrieval.rerank_mod.score_pairs(question, cand) if cand else None
            if cand_scores:
                if query["kind"] == "positive":
                    gold_scores = [s for doc, s in zip(cand, cand_scores)
                                   if rank_of_first_hit([doc], query["gold_phrases"]) == 1]
                    if gold_scores:
                        positive_scores.append(max(gold_scores))
                else:
                    negative_scores.append(max(cand_scores))
    elapsed = time.perf_counter() - started

    summary = summarize(results, args.top_k)

    # 切窗诊断：chunk 是否真的超出了 embedding 的 token 窗口
    token_over, max_seq_length = [], None
    try:
        model = embedding_fn._model  # SentenceTransformer（chromadb 的 EF 把它挂在 _model）
        tokenizer = model.tokenizer
        max_seq_length = int(getattr(model, "max_seq_length", 512))
        token_over = exceeds_model_window(
            chunks, max_seq_length,
            lambda text: len(tokenizer.encode(text, add_special_tokens=True)))
    except Exception as e:  # 拿不到 tokenizer 不该让整个评测失败
        print(f"（切窗诊断不可用：{e}）")

    print()
    print("=== 检索评测结果 ===")
    print(f"embedding 模型：{EMBEDDING_MODEL}")
    print(f"recall@{args.top_k} = {summary['recall_at_k']:.3f}"
          f"（{summary['hits_at_k']}/{summary['positives']}）")
    print(f"MRR       = {summary['mrr']:.3f}")
    attr = summary["file_attribution_rate"]
    attr_text = ("无数据（命中的块没带来源元数据）" if attr is None else
                 f"{attr:.3f}（{summary['file_attribution_hits']}/"
                 f"{summary['file_attribution_total']}）")
    print(f"文件归属正确率 = {attr_text}")
    sec = summary["section_attribution_rate"]
    sec_text = ("无数据（没有带页码标注的样本，或命中块没带来源元数据）" if sec is None else
                f"{sec:.3f}（{summary['section_attribution_hits']}/"
                f"{summary['section_attribution_total']}）")
    print(f"section 归属正确率 = {sec_text}")
    print(f"关键词兜底样本 = {summary['keyword_fallback_samples']}/{len(results)}"
          f"（这些样本的分数是 2-gram 词面打分，不是向量距离）")
    print(f"负样本误召回率 = {summary['negative_false_positive_rate']:.3f}"
          f"（{summary['negative_false_positives']}/{summary['negatives']}）")
    far_rate = summary["negative_false_positive_rate_far"]
    adj_rate = summary["negative_false_positive_rate_adjacent"]
    print(f"  ├ 远域（零词面重叠）误召回 = "
          f"{'无数据' if far_rate is None else f'{far_rate:.3f}'}"
          f"（{summary['negative_false_positives_far']}/{summary['negative_far_total']}）")
    print(f"  └ 域内（无答案但有重叠）误召回 = "
          f"{'无数据' if adj_rate is None else f'{adj_rate:.3f}'}"
          f"（{summary['negative_false_positives_adjacent']}/"
          f"{summary['negative_adjacent_total']}）")
    print(f"耗时 = {elapsed:.1f}s（{len(queries)} 条样本）")

    if max_seq_length is not None:
        print()
        print("--- 切窗诊断 ---")
        print(f"模型窗口 {max_seq_length} token；超过窗口的 chunk："
              f"{len(token_over)}/{len(chunks)}")
        if token_over:
            lengths = sorted(length for length, _ in token_over)
            print(f"  超出长度：min={lengths[0]} max={lengths[-1]} "
                  f"中位={statistics.median(lengths):.0f}")
            print("  超窗口的块，尾部会被 tokenizer 静默截断，那部分内容不参与向量。")

    if args.calibrate:
        print()
        print("--- 距离标定（原问题直查，未经阈值过滤）---")
        if positive_distances:
            print(f"正样本命中块距离：min={min(positive_distances):.4f} "
                  f"max={max(positive_distances):.4f} n={len(positive_distances)}")
        else:
            print("正样本命中块距离：无数据（可能这轮全部未命中）")
        if negative_distances:
            print(f"负样本最小距离：  min={min(negative_distances):.4f} "
                  f"max={max(negative_distances):.4f} n={len(negative_distances)}")
        else:
            print("负样本最小距离：无数据")
        suggested = suggest_threshold(positive_distances, negative_distances)
        if suggested is None:
            print(f"建议 MAX_DISTANCE：分不开（正负距离重叠），当前默认 {MAX_DISTANCE}")
        else:
            print(f"建议 MAX_DISTANCE：{suggested}（当前默认 {MAX_DISTANCE}）")

        if retrieval.RERANK_ENABLED:
            print()
            print("--- 精排分数标定（原问题 × 候选，未经 min_score 过滤）---")
            pos_band, neg_band = score_band(positive_scores), score_band(negative_scores)
            if pos_band["n"]:
                print(f"正样本命中块分数：min={pos_band['min']} 中位={pos_band['median']} "
                      f"max={pos_band['max']} n={pos_band['n']}")
            else:
                print("正样本命中块分数：无数据")
            if neg_band["n"]:
                print(f"负样本最高分：    min={neg_band['min']} 中位={neg_band['median']} "
                      f"max={neg_band['max']} n={neg_band['n']}")
            else:
                print("负样本最高分：无数据")
            suggested_score = suggest_score_threshold(positive_scores, negative_scores)
            if suggested_score is None:
                print(f"建议 RERANK_MIN_SCORE：分不开（正负分数重叠），"
                      f"当前 {retrieval.RERANK_MIN_SCORE}")
            else:
                print(f"建议 RERANK_MIN_SCORE：{suggested_score}"
                      f"（当前 {retrieval.RERANK_MIN_SCORE}）")

    if judge is not None:
        verdicts = summarize_answerability(results)
        print()
        print("--- 可答性判断（只判「检索返回了内容」的样本）---")
        print(f"正样本被误杀 = {verdicts['positive_rejected']}/{verdicts['positive_judged']}"
              f"（判「不能」；越少越好，理想是 0）")
        print(f"远域负样本被拦下 = {verdicts['far_rejected']}/{verdicts['far_judged']}")
        print(f"域内负样本被拦下 = {verdicts['adjacent_rejected']}/{verdicts['adjacent_judged']}"
              f"（这是这次改造的主要目标）")
        print(f"判定不可用（None，按原行为作答）= {verdicts['unknown_verdicts']}")
        if judge_seconds:
            ordered = sorted(judge_seconds)
            print(f"判定耗时 = 平均 {sum(ordered) / len(ordered):.2f}s / 中位 "
                  f"{ordered[len(ordered) // 2]:.2f}s（{len(ordered)} 次调用）")

    misses = [r for r in results if r["kind"] == "positive" and r["rank"] is None]
    if judge is not None:
        rejected_pos = [r["id"] for r in results
                        if r["kind"] == "positive" and r.get("answerable") is False]
        print(f"被判「不能」的正样本：{', '.join(rejected_pos) or '无'}")
    false_positives = [r for r in results if r["kind"] == "negative" and r["returned"]]
    print()
    print(f"未命中正样本：{', '.join(r['id'] for r in misses) or '无'}")
    print(f"误召回负样本：{', '.join(r['id'] for r in false_positives) or '无'}")

    if args.verbose:
        print()
        print("--- 明细 ---")
        by_id = {q["id"]: q for q in queries}
        path_label = {retrieval.PATH_VECTOR: "向量", retrieval.PATH_KEYWORD: "关键词兜底"}
        for r in results:
            query = by_id[r["id"]]
            if r["kind"] == "positive":
                status = f"rank={r['rank']}" if r["rank"] else "未命中"
            else:
                status = "误召回" if r["returned"] else "正确返回空"
            print(f"  [{r['id']}] {status} [{path_label.get(r.get('path'), '未知')}]"
                  f" | {query['question']}")

    if args.json:
        payload = {
            "corpus_files": sorted(corpus),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": args.top_k,
            "hyde": bool(args.hyde),
            "model": EMBEDDING_MODEL,
            "summary": summary,
            "max_seq_length": max_seq_length,
            "over_window_chunks": len(token_over),
            "current_max_distance": MAX_DISTANCE,
            "suggested_max_distance": suggest_threshold(positive_distances, negative_distances),
            "rerank_enabled": bool(retrieval.RERANK_ENABLED),
            "fusion": {"enabled": bool(retrieval.FUSION_ENABLED),
                        "gate": retrieval.FUSION_GATE,
                        "rrf_k": retrieval.RRF_K},
            "answerability": (summarize_answerability(results) if judge is not None else None),
            "rerank": ({
                "model": retrieval.RERANK_MODEL,
                "candidates": retrieval.RERANK_CANDIDATES,
                "top_n": retrieval.RERANK_TOP_N,
                "min_score": retrieval.RERANK_MIN_SCORE,
                "positive_scores": score_band(positive_scores),
                "negative_scores": score_band(negative_scores),
                "suggested_min_score": suggest_score_threshold(positive_scores, negative_scores),
            } if args.calibrate and retrieval.RERANK_ENABLED else None),
            "results": results,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入：{args.json}")

    if args.hyde and args.hyde_cache:
        # 落盘时机放在最后：中途 return 2（自检失败等）不写，免得把半成品当基线复用
        save_hyde_cache(args.hyde_cache, hyde_cache)
        print(f"假设文本已缓存：{args.hyde_cache}（{len(hyde_cache)} 条；"
              f"下一轮换个开关跑会直接复用这批文本）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
