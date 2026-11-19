#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""范围词（`SCOPE_FILTER`）与候选池（`FUSION_CANDIDATES`）的对照实验 —— 只量数字、不改代码。

要回答的问题
------------
真实对话里那条「结合第2章课件，在同一训练集上不断提高多项式回归的阶数…」只捞到 1 块并拒答，
去掉前缀却能答对。两个候选修法各有代价，必须分别量：

| 臂 | `SCOPE_FILTER` | `FUSION_CANDIDATES` |
|----|----------------|---------------------|
| base | 0 | 0（现状） |
| pool | 0 | 20 |
| scope | 1 | 0 |
| both | 1 | 20 |

量什么
------
1. **质量**：`tests/fixtures/scope_queries.json` 的 12 条「结合第N章课件，…」提问里，
   事实核（`gold_phrases`）有没有进 context、命中块是不是落在问句点名的那份文件；
2. **回归**：`tests/fixtures/eval_queries.json` 的 `far` 负样本（与本语料零词面重叠）
   在各臂下的**误召回**有没有上升；
3. **代价**：检索耗时（HyDE 走缓存，只算检索侧）、命中块数、context 字符数。

安全约定
--------
- 语料读自 `state/*.json` 的 `files[].file_text`（用户上传件的解析文本，**只读、不落盘、
  不打印正文**）；向量库用 `chromadb.EphemeralClient`，绝不碰真实 `chroma_db/`；
- 事实核**离线校验**：每条 `gold_phrases` 必须逐字出现在它 `expect_files` 里的那份文件中，
  对不上直接退出码 2（量尺错了，数字再好也没意义）；
- 默认 `--no-hyde`（不花额度、可复现）；`--hyde` 才是线上口径，配合 `--hyde-cache` 复用。

用法
----
    python scripts/eval_scope_filter.py --verify-only          # 只校事实核，不检索
    python scripts/eval_scope_filter.py                        # 四臂，no-hyde，不花额度
    python scripts/eval_scope_filter.py --hyde --hyde-cache .agents/scope-hyde.json
    python scripts/eval_scope_filter.py --conv <对话 id>       # 语料有多个对话时指定

退出码：0 正常；2 输入不合法（语料/事实核/对话找不到）。
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

import chromadb  # noqa: E402

import src.retrieval as retrieval  # noqa: E402
from scripts.eval_retrieval import cached_generate, load_hyde_cache, save_hyde_cache  # noqa: E402
from src.config import logger  # noqa: E402

SCOPE_QUERIES = ROOT / "tests" / "fixtures" / "scope_queries.json"
NEGATIVE_QUERIES = ROOT / "tests" / "fixtures" / "eval_queries.json"

ARMS = (
    ("base", False, 0),
    ("pool", False, 20),
    ("scope", True, 0),
    ("both", True, 20),
)


def load_scope_queries(path=SCOPE_QUERIES):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"提问集不合法（应为非空数组）：{path}")
    for item in data:
        for key in ("id", "question", "expect_files", "gold_phrases"):
            if key not in item:
                raise SystemExit(f"提问 {item!r} 缺字段 {key}")
    return data


def load_far_negatives(path=NEGATIVE_QUERIES):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item for item in data
            if item.get("kind") == "negative" and item.get("negative_class") == "far"]


def conversations(state_dir):
    """读真实状态目录 → `[{"conv_id", "files": [(名, 正文)], "questions": [...]}, ...]`（只读）。"""
    from scripts.eval_real_corpus import conversations_from_state

    return conversations_from_state(state_dir)


def pick_corpus(convs, conv_id=None, required=()):
    """挑一份语料。

    给了 `conv_id` 就用它；否则**按事实核点名的文件名反查**：只有包含全部 `expect_files`
    的那份语料才配用来评这套提问（选错语料时事实核校验会全线报错，不如在这里就挑对）。
    """
    usable = [conv for conv in convs if conv.get("files")]
    if not usable:
        raise SystemExit("状态目录里没有带文件的对话，无从评测")
    if conv_id:
        for conv in usable:
            if conv["conv_id"] == conv_id:
                return conv["conv_id"], conv["files"]
        raise SystemExit(f"找不到对话 {conv_id}（有：{[conv['conv_id'] for conv in usable]}）")
    required = [name for name in required if name]
    matched = [conv for conv in usable
               if required and all(any(name == file_name for file_name, _text in conv["files"])
                                   for name in required)]
    if not matched:
        raise SystemExit("没有哪份语料同时包含这套提问点名的文件："
                         f"{required}；请用 --conv 指定（有：{[conv['conv_id'] for conv in usable]}）")
    best = sorted(matched, key=lambda conv: (-len(conv["files"]), conv["conv_id"]))[0]
    return best["conv_id"], best["files"]


def verify_gold(queries, files):
    """事实核离线校验：每条 `gold_phrases` 必须逐字出现在 `expect_files` 里，否则退出码 2。"""
    by_name = {name: text for name, text in files}
    problems = []
    for item in queries:
        for name in item["expect_files"]:
            if name not in by_name:
                problems.append(f"{item['id']}：语料里没有文件《{name}》")
        for phrase in item["gold_phrases"]:
            targets = item["expect_files"] or list(by_name)
            if not any(phrase in by_name.get(name, "") for name in targets):
                problems.append(f"{item['id']}：事实核「{phrase}」不在《{'、'.join(targets)}》里")
    return problems


def build_collection(files):
    """把语料灌进一个**临时**向量库（EphemeralClient，进程结束即消失）。"""
    client = chromadb.EphemeralClient()
    coll = client.create_collection("kb_eval_scope", embedding_function=retrieval._get_embedding_fn())
    pairs = retrieval._pairs_with_meta(files, "")
    coll.add(documents=[text for text, _meta in pairs],
             ids=[f"chunk_{index}" for index in range(len(pairs))],
             metadatas=[retrieval.chroma_meta(meta) for _text, meta in pairs])
    return coll, len(pairs)


def run_arm(question, coll, files, top_k, scope_filter, fusion_candidates):
    """跑一条提问（不含生成），返回 (hits, 检索路径, 检索耗时秒)。"""
    retrieval.SCOPE_FILTER = scope_filter
    retrieval.FUSION_CANDIDATES = fusion_candidates
    started = time.perf_counter()
    path, hits = retrieval.hyde_retrieve_hits_with_path(question, coll, "", use_vector=True,
                                                        top_k=top_k, files=files)
    return hits, path, time.perf_counter() - started


def score(hits, item):
    """一条提问的记账：事实核命中数 / 命中块数 / 是否命中目标文件 / context 字符数。"""
    texts = [text or "" for text, _meta in hits]
    files_hit = {(meta or {}).get("file_name") for _text, meta in hits}
    covered = sum(1 for phrase in item["gold_phrases"] if any(phrase in text for text in texts))
    return {
        "blocks": len(hits),
        "chars": sum(len(text) for text in texts),
        "gold_total": len(item["gold_phrases"]),
        "gold_covered": covered,
        "file_hit": bool(files_hit & set(item["expect_files"])),
    }


def summarize(rows):
    """把逐条记账汇总成一个臂的数字。"""
    if not rows:
        return {}
    gold_total = sum(row["gold_total"] for row in rows)
    gold_covered = sum(row["gold_covered"] for row in rows)
    return {
        "questions": len(rows),
        "gold_covered": gold_covered,
        "gold_total": gold_total,
        "gold_rate": gold_covered / gold_total if gold_total else 0.0,
        "answerable": sum(1 for row in rows if row["gold_covered"] > 0),
        "file_hit": sum(1 for row in rows if row["file_hit"]),
        "blocks_median": statistics.median([row["blocks"] for row in rows]),
        "chars_median": statistics.median([row["chars"] for row in rows]),
        "seconds_median": statistics.median([row["seconds"] for row in rows]),
    }


def evaluate(queries, negatives, coll, files, top_k, use_hyde, hyde_cache, hyde_cache_path):
    """四个臂 × (质量提问 + 负样本)，返回 {臂: {"pos": 汇总, "neg": 汇总, "rows": [...]}}。

    延时**单独计时一遍**：第一遍只为把 HyDE 灌进缓存（否则先跑的臂替后面所有臂付了 API 延迟，
    量出来的"检索耗时"根本不是检索）。第二遍才是纯检索，各臂同口径可比。
    """
    generate = cached_generate(retrieval._generate_hyde, hyde_cache)
    original = retrieval._generate_hyde
    results = {}
    try:
        for arm, scope_filter, fusion_candidates in ARMS:
            retrieval._generate_hyde = generate if use_hyde else (lambda q: q)
            pos_rows, neg_rows = [], []
            for item in queries:
                hits, path, _seconds = run_arm(item["question"], coll, files, top_k,
                                               scope_filter, fusion_candidates)
                row = score(hits, item)
                row.update({"id": item["id"], "path": path,
                            "file_names": sorted({(m or {}).get("file_name") for _t, m in hits})})
                pos_rows.append(row)
            for item in negatives:
                hits, path, _seconds = run_arm(item["question"], coll, files, top_k,
                                               scope_filter, fusion_candidates)
                neg_rows.append({"id": item["id"], "blocks": len(hits), "path": path,
                                 "false_recall": bool(hits),
                                 "vector_false_recall": bool(hits) and path == retrieval.PATH_VECTOR})
            # 第二遍：纯检索计时（HyDE 已缓存）
            for row, item in zip(pos_rows, queries):
                _hits, _path, seconds = run_arm(item["question"], coll, files, top_k,
                                                scope_filter, fusion_candidates)
                row["seconds"] = seconds
            results[arm] = {
                "pos": summarize(pos_rows),
                "neg": {
                    "questions": len(neg_rows),
                    "false_recall": sum(1 for row in neg_rows if row["false_recall"]),
                    "vector_false_recall": sum(1 for row in neg_rows
                                               if row["vector_false_recall"]),
                },
                "rows": pos_rows,
                "neg_rows": neg_rows,
            }
    finally:
        retrieval._generate_hyde = original
        retrieval.SCOPE_FILTER = False
        retrieval.FUSION_CANDIDATES = 0
        if hyde_cache_path is not None:
            save_hyde_cache(hyde_cache_path, hyde_cache)
    return results


def print_report(results, corpus_name, blocks, top_k, use_hyde):
    print(f"\n语料：{corpus_name}（{blocks} 块）｜top_k={top_k}｜HyDE={'开' if use_hyde else '关'}\n")
    header = (f"{'臂':<6}{'事实核':>11}{'整条答到':>10}{'目标文件':>10}{'块数中位':>10}"
              f"{'字符中位':>10}{'检索中位':>10}{'负样本误召回':>12}{'其中走向量':>11}")
    print(header)
    print("-" * len(header))
    for arm, _scope, _fusion in ARMS:
        row = results[arm]
        pos, neg = row["pos"], row["neg"]
        print(f"{arm:<6}"
              f"{pos['gold_covered']:>6}/{pos['gold_total']:<4}"
              f"{pos['answerable']:>6}/{pos['questions']:<3}"
              f"{pos['file_hit']:>6}/{pos['questions']:<3}"
              f"{pos['blocks_median']:>10.1f}"
              f"{pos['chars_median']:>10.0f}"
              f"{pos['seconds_median']:>9.3f}s"
              f"{neg['false_recall']:>8}/{neg['questions']:<3}"
              f"{neg['vector_false_recall']:>8}/{neg['questions']:<3}")
    print("\n逐条（事实核命中 / 块数 / 落到的文件）：")
    for arm, _scope, _fusion in ARMS:
        print(f"  [{arm}]")
        for row in results[arm]["rows"]:
            files = "、".join(name for name in row["file_names"] if name) or "（无来源）"
            print(f"    {row['id']}  {row['gold_covered']}/{row['gold_total']}  "
                  f"{row['blocks']} 块  {files}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="范围词 / 候选池对照实验")
    parser.add_argument("--state-dir", default=str(ROOT / "state"))
    parser.add_argument("--conv", default=None, help="状态目录里的对话 id（默认取文件最多的那份）")
    parser.add_argument("--top-k", type=int, default=10, help="线上默认 max(10, 文件数×3)")
    parser.add_argument("--hyde", action="store_true", help="走真实 HyDE（花 API 额度）")
    parser.add_argument("--hyde-cache", default=None, help="HyDE 缓存 json（复用/落盘）")
    parser.add_argument("--verify-only", action="store_true", help="只校验事实核，不检索")
    parser.add_argument("--json", default=None, help="把逐条结果写到这个路径")
    args = parser.parse_args(argv)

    queries = load_scope_queries()
    negatives = load_far_negatives()
    required = sorted({name for item in queries for name in item["expect_files"]})
    conv_id, files = pick_corpus(conversations(Path(args.state_dir)), args.conv, required)

    problems = verify_gold(queries, files)
    if problems:
        print("事实核校验未通过（量尺有问题，先修这个）：", file=sys.stderr)
        for problem in problems:
            print("  - " + problem, file=sys.stderr)
        return 2
    print(f"事实核校验通过：{len(queries)} 条提问 / "
          f"{sum(len(item['gold_phrases']) for item in queries)} 个事实核，全部逐字落在"
          f"《{'、'.join(name for name, _text in files)}》里")
    if args.verify_only:
        return 0

    coll, blocks = build_collection(files)
    hyde_cache_path = Path(args.hyde_cache) if args.hyde_cache else None
    hyde_cache = load_hyde_cache(hyde_cache_path) if hyde_cache_path else {}
    results = evaluate(queries, negatives, coll, files, args.top_k,
                       args.hyde, hyde_cache, hyde_cache_path)
    print_report(results, conv_id, blocks, args.top_k, args.hyde)

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n逐条结果已写到 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
