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

用法
----
    python scripts/eval_retrieval.py                        # 默认 no-hyde，打印指标
    python scripts/eval_retrieval.py --calibrate --verbose   # 附带距离分布与逐条明细
    python scripts/eval_retrieval.py --hyde                 # 走真实 HyDE（花 API 额度）
    python scripts/eval_retrieval.py --json out.json        # 结果另存 JSON

退出码：0 正常；2 输入（语料 / 标注）不合法。
"""
import argparse
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


def load_corpus(corpus_dir):
    """读目录下的 .md/.txt，按文件名排序，返回 {文件名: 原文}。"""
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
    return files


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
        if kind == "positive":
            if not gold_phrases:
                raise ValueError(f"第 {i} 条正样本缺少 gold_phrases")
            if not expect_files:
                raise ValueError(f"第 {i} 条正样本缺少 expect_files")
        queries.append({
            "id": str(item.get("id") or f"q{i:02d}"),
            "question": question,
            "kind": kind,
            "expect_files": expect_files,
            "gold_phrases": gold_phrases,
            "note": str(item.get("note") or ""),
        })

    if not queries:
        raise ValueError("评测标注是空的")
    return queries


def rank_and_file(hits, gold_phrases, expect_files):
    """同时返回首个命中位次、以及该命中是否来自预期文件。

    文件归属只在命中块真的带了来源元数据时才有意义；没有元数据（老索引）时返回 None，
    由上层跳过统计，而不是把「不知道」算成「归属错误」。
    """
    needles = [n for n in (normalize(g) for g in gold_phrases) if n]
    if not needles:
        return None, None
    for rank, (text, meta) in enumerate(hits, 1):
        haystack = normalize(text)
        if any(needle in haystack for needle in needles):
            name = (meta or {}).get("file_name")
            if not name or not expect_files:
                return rank, None
            return rank, name in expect_files
    return None, None


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


def summarize(results, k):
    """汇总指标。results 每条形如 {'kind', 'rank', 'returned'}。

    - recall@k：正样本里命中片段出现在前 k 个 chunk 的比例
    - MRR：首个命中位次倒数的均值（超出 k 的算 0）
    - 负样本误召回率：负样本里「检索返回了非空结果」的比例
    """
    positives = [r for r in results if r["kind"] == "positive"]
    negatives = [r for r in results if r["kind"] == "negative"]

    hit_flags = [r["rank"] is not None and r["rank"] <= k for r in positives]
    reciprocal = [(1.0 / r["rank"]) if (r["rank"] is not None and r["rank"] <= k) else 0.0
                  for r in positives]
    false_positives = [r for r in negatives if r.get("returned")]
    # 文件归属只在命中块真的带了元数据时才统计：老索引没有 metadatas 时 file_ok 是 None，
    # 不能把「不知道」算成「错了」，否则这个指标会骗人。
    attributed = [r for r in positives if r.get("file_ok") is not None]
    attributed_ok = [r for r in attributed if r["file_ok"]]

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "hits_at_k": sum(hit_flags),
        "recall_at_k": (sum(hit_flags) / len(positives)) if positives else 0.0,
        "mrr": (sum(reciprocal) / len(reciprocal)) if reciprocal else 0.0,
        "file_attribution_hits": len(attributed_ok),
        "file_attribution_total": len(attributed),
        "file_attribution_rate": (len(attributed_ok) / len(attributed)) if attributed else None,
        "negative_false_positives": len(false_positives),
        "negative_false_positive_rate":
            (len(false_positives) / len(negatives)) if negatives else 0.0,
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
        print("检索模式：HyDE（真实调用大模型，会花 API 额度）")

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
    started = time.perf_counter()
    for query in queries:
        question = query["question"]
        hits = retrieval.hyde_retrieve_hits(
            question, collection, full_text, use_vector=True, top_k=args.top_k, files=files)

        if query["kind"] == "positive":
            rank, file_ok = rank_and_file(hits, query["gold_phrases"], query["expect_files"])
            results.append({"id": query["id"], "kind": "positive", "rank": rank,
                            "returned": len(hits), "file_ok": file_ok})
        else:
            results.append({"id": query["id"], "kind": "negative",
                            "rank": None, "returned": len(hits)})

        if args.calibrate:
            # 原始查询：不经阈值过滤，用来看真实距离分布
            raw = collection.query(
                query_texts=[question], n_results=min(args.top_k, len(chunks)))
            docs = (raw.get("documents") or [[]])[0]
            dists = (raw.get("distances") or [[]])[0]
            if query["kind"] == "positive":
                gold_hits = [dist for doc, dist in zip(docs, dists)
                             if rank_of_first_hit([doc], query["gold_phrases"]) == 1]
                if gold_hits:
                    positive_distances.append(min(gold_hits))
            elif dists:
                negative_distances.append(min(dists))
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
    print(f"负样本误召回率 = {summary['negative_false_positive_rate']:.3f}"
          f"（{summary['negative_false_positives']}/{summary['negatives']}）")
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

    misses = [r for r in results if r["kind"] == "positive" and r["rank"] is None]
    false_positives = [r for r in results if r["kind"] == "negative" and r["returned"]]
    print()
    print(f"未命中正样本：{', '.join(r['id'] for r in misses) or '无'}")
    print(f"误召回负样本：{', '.join(r['id'] for r in false_positives) or '无'}")

    if args.verbose:
        print()
        print("--- 明细 ---")
        by_id = {q["id"]: q for q in queries}
        for r in results:
            query = by_id[r["id"]]
            if r["kind"] == "positive":
                status = f"rank={r['rank']}" if r["rank"] else "未命中"
            else:
                status = "误召回" if r["returned"] else "正确返回空"
            print(f"  [{r['id']}] {status} | {query['question']}")

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
            "results": results,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入：{args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
