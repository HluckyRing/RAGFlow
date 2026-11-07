#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量「不要独立判定、让 QA 自己拒答」这一版的质量与延迟（inline 方案）。

为什么单独一个脚本
------------------
可答性判断（`src/answerability.py`）每条提问固定 **+0.78 s**，而实测缩短输入没用
（只看 top1 1.01 s / top2 0.92 s / 全长 0.78 s）—— 延迟是 API 往返，不是 prefill。
要真省掉这次调用，只能让**生成那一次调用**自己说「材料里没有」。`QA_SYSTEM_PROMPT_TEMPLATE`
第 1 条本来就是这条纪律，这个脚本量它到底管不管用。

结论（2026-11-06 实测，59 条）：域内负样本正确拒答 **8/8**、远域 **2/2**、正样本误拒
**0/36** —— 与独立判定分毫不差，但**零额外调用**。所以 `ANSWERABILITY_CHECK` 默认改为 0
（inline 既省 0.78 s，拒答文案还更详细）；独立判定保留为「严格模式」可选项。

判据：
- 域内负样本 8 条：答完必须**拒答**（说没有相关内容），否则就是「答非所问」；
- 正样本 36 条：**一条都不许**拒答（拒答 = 误杀）。

用法（会真实调用大模型，约 46 次生成）：
    python scripts/measure_inline_abstention.py
    python scripts/measure_inline_abstention.py --json /tmp/inline.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_retrieval import (  # noqa: E402
    DEFAULT_CORPUS, DEFAULT_QUERIES, build_context, load_corpus, load_queries)
from src.prompts import REFUSAL_MARKER  # noqa: E402

# 判定「是不是拒答」用**结构化标记**，不是文案启发式（2026-11-06）：
# 提示词要求模型答不出时以 `REFUSAL_MARKER` 开头，服务端据此推 refused 事件；
# 这里量的是同一件事，所以用同一套契约 —— 顺带也量了「模型到底听不听话」。

def looks_like_abstention(answer):
    """这条回答算不算拒答。

    只看**开头**：提示词要求答不出时以 `REFUSAL_MARKER` 开头，正文中间的这几个字不算
    （那是正常回答在讲标记本身）。允许模型把标记放进 markdown 加粗里。
    """
    text = (answer or "").strip()
    if not text:
        return True
    head = text.lstrip("*# ").lstrip()
    return head.startswith(REFUSAL_MARKER)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="量「让 QA 自己拒答」的质量与延迟（会真实调用大模型）")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--hyde-cache", default=".agents/hyde-cache.json",
                        help="假设文本缓存（不给就每次重新生成，结果不可比）")
    parser.add_argument("--json", default=".agents/inline-abstain.json",
                        help="逐条明细写到这个 JSON")
    args = parser.parse_args(argv)

    import chromadb

    import src.llm as llm
    import src.retrieval as retrieval

    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    files = [(name, corpus[name]) for name in sorted(corpus)]
    full_text = "\n\n".join(text for _, text in files)
    pairs = retrieval._pairs_with_meta(files, "")
    chunks = [chunk for chunk, _ in pairs]
    collection = chromadb.EphemeralClient().get_or_create_collection(
        name="inline_kb", embedding_function=retrieval._get_embedding_fn())
    collection.add(documents=chunks, ids=[f"i{i}" for i in range(len(chunks))],
                   metadatas=[retrieval.chroma_meta(meta) for _, meta in pairs])

    cache_path = Path(args.hyde_cache)
    hyde = (json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.exists() else {})
    retrieval._generate_hyde = lambda q: hyde.get(q, q)
    print(f"假设文本缓存：{cache_path}（{len(hyde)} 条；缺的用原问题代替）")

    rows = []
    for query in queries:
        question = query["question"]
        _path, hits = retrieval.hyde_retrieve_hits_with_path(
            question, collection, full_text, use_vector=True, top_k=5, files=files)
        if not hits:
            rows.append({"id": query["id"], "kind": query["kind"],
                         "class": query.get("negative_class"), "returned": 0,
                         "abstained": None, "answer": "", "seconds": None})
            continue
        context = build_context(hits)
        started = time.perf_counter()
        first_at = None
        parts = []
        try:
            for token in llm.stream_answer(question, context, []):
                if first_at is None:
                    first_at = time.perf_counter() - started
                parts.append(token)
        except Exception as e:                      # noqa: BLE001
            print(f"[{query['id']}] 生成失败：{str(e)[:80]}")
            continue
        answer = "".join(parts)
        rows.append({"id": query["id"], "kind": query["kind"],
                     "class": query.get("negative_class"), "returned": len(hits),
                     "abstained": looks_like_abstention(answer), "answer": answer,
                     "seconds": time.perf_counter() - started,
                     "ttft": first_at, "context_chars": len(context)})

    def group(kind, cls=None):
        return [r for r in rows if r["kind"] == kind and r["returned"]
                and (cls is None or r["class"] == cls)]

    pos = group("positive")
    adj = group("negative", "adjacent")
    far = group("negative", "far")
    pos_bad = [r for r in pos if r["abstained"]]
    adj_ok = [r for r in adj if r["abstained"]]
    far_ok = [r for r in far if r["abstained"]]

    print()
    print("=== 让 QA 自己拒答（inline，零额外调用）===")
    print(f"域内负样本正确拒答 = {len(adj_ok)}/{len(adj)}")
    print(f"  仍然硬答的：{[r['id'] for r in adj if not r['abstained']]}")
    print(f"远域负样本正确拒答 = {len(far_ok)}/{len(far)}")
    print(f"正样本被误拒 = {len(pos_bad)}/{len(pos)}"
          f"{'（' + ', '.join(r['id'] for r in pos_bad) + '）' if pos_bad else ''}")
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    ttfts = [r["ttft"] for r in rows if r.get("ttft")]
    if secs:
        print(f"生成总耗时：中位 {statistics.median(secs):.2f}s / 平均 {sum(secs) / len(secs):.2f}s"
              f"（{len(secs)} 条）")
    if ttfts:
        print(f"首 token 延迟：中位 {statistics.median(ttfts):.2f}s / 平均 "
              f"{sum(ttfts) / len(ttfts):.2f}s —— 这是并行方案能藏住判定耗时的高度")

    Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"明细已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
