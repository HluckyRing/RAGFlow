#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用**存下来的回答**重新判分，不重跑模型。

为什么这个脚本值得入库
----------------------
分桶只依赖「回答文本 + 标注」两样东西，与提示词、检索、模型随机性**都无关**。所以：

- 改**判定口径**（比如 `classify_partial` 加了抹装饰字符、加了新桶）→ 拿旧明细重算即可，
  **不必**花一轮 API 额度；而且重算结果与旧结果是同一批回答，可比性更好（没有采样噪声）；
- 改**标注锚点**（`answer_terms` / `absent_terms` 收成最短事实核）→ 同理；
- 只有改**提示词**或改**检索**才必须真跑 `measure_inline_abstention.py`。

前提是每轮把 `answer` 原文写进 `--json` 的明细文件 —— 这正是那个文件存在的意义。

2026-11-08 就是这么发现「第一轮 5/17 是假失败」的：模型的回答在语义上全对，是量尺把
`保留 **30 天**` 这种带 Markdown 强调的写法判成了「漏答」。逐条看原文 + 离线重算，
才把「模型答错了」和「判定器坏了」分开 —— 这两件事在汇总数字里长得一模一样。

用法：
    python scripts/rejudge_abstention.py .agents/inline-shapes-v1.json
    python scripts/rejudge_abstention.py .agents/inline-shapes-v1.json --write
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_retrieval import DEFAULT_QUERIES, load_queries  # noqa: E402
from scripts.measure_inline_abstention import (  # noqa: E402
    classify_partial, part_coverage, post_gap_coverage, print_partial_summary)


def rejudge(rows, queries):
    """重算每一行的分桶与逐项覆盖，返回 (新 rows, 与原判不同的 id 列表)。

    `rows` 是 `--json` 写出的明细（每行至少要有 `id` / `kind` / `answer`）；
    `queries` 是 `load_queries()` 读出的标注（要带 `parts` / `answer_terms` / `absent_terms`）。
    没生成出回答的行（`answer` 为空、`returned` 为 0）原样保留、不参与重算 ——
    它们没被判过分，重算会把「检索没召回」误报成「模型拒答」。
    """
    by_id = {query["id"]: query for query in queries}
    changed = []
    rejudged = []
    for row in rows:
        query = by_id.get(row.get("id"))
        if query is None or query["kind"] != "partial" or not row.get("answer"):
            rejudged.append(row)
            continue
        answer = row["answer"]
        bucket = classify_partial(answer, query["answer_terms"], query["absent_terms"])
        parts = part_coverage(answer, query.get("parts"))
        if row.get("bucket") and row["bucket"] != bucket:
            changed.append(row["id"])
        row = {**row,
               "bucket": bucket,
               "parts": parts,
               "coverage": {"covered": sum(1 for part in parts
                                           if part["answerable"] and part["ok"]),
                            "total": sum(1 for part in parts if part["answerable"])},
               "post_gap": post_gap_coverage(answer, query.get("parts"))}
        rejudged.append(row)
    return rejudged, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="用存下来的回答重新判分（不调用模型）")
    parser.add_argument("detail", help="measure_inline_abstention.py --json 写出的明细")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--write", action="store_true",
                        help="把重算结果另存为 <明细>-rejudged.json")
    args = parser.parse_args(argv)

    detail_path = Path(args.detail)
    if not detail_path.exists():
        print(f"找不到明细文件：{detail_path}")
        return 2
    rows = json.loads(detail_path.read_text(encoding="utf-8"))
    queries = load_queries(args.queries)

    rows, changed = rejudge(rows, queries)
    for row in rows:
        if row.get("kind") != "partial" or not row.get("answer"):
            continue
        coverage = row.get("coverage") or {}
        post_gap = row.get("post_gap")
        print(f"{row['id']:4s} {row['bucket']:24s} "
              f"覆盖 {coverage.get('covered')}/{coverage.get('total')}"
              f" post={post_gap and (post_gap['covered'], post_gap['total'])}")
        for part in row.get("parts") or []:
            if not part["ok"]:
                print(f"       未命中：{part['text']}（是漏答，还是锚点太脆？看原文）")
    print()
    print("与原判不同的样本：", changed or "（无）")

    partial = [row for row in rows
               if row.get("kind") == "partial" and row.get("answer")]
    print_partial_summary(partial)

    if args.write:
        out = detail_path.with_name(detail_path.stem + "-rejudged.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print("重算明细已写入", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
