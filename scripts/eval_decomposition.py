#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拆问（query 分解）自己的测量：闸门准不准、调用多少钱、拆出来的条数对不对。

为什么只量「结构」不量「拆对率」
--------------------------------
「拆出来的子问和 gold 子问像不像」只能用词面相似度当尺子，而这个项目在这上面栽过三次
（`保留 **30 天**`、`可以折算` vs `可折算`、`病假的最长可请期限`）—— 太松会把失败藏起来，
太紧会冤枉正确回答。所以这里只量**机械可核**的东西：闸门判对没有、调用几次、延迟多少、
拆出几条（和 gold 的条数差）、单问有没有被误拆、解析失败几次。

**拆问质量的真口径在另一处**：把拆出来的子问真的拿去检索，看 gold 目标片还在不在
（`scripts/eval_compound_retrieval.py --route decompose`，命中口径是同一片/同文件同节，
逐字可核）。本脚本的条数指标只能说明「拆得太碎/太粗」，说明不了「拆对了」。

用法
----
    python scripts/eval_decomposition.py                 # 只量闸门（不花钱、瞬间）
    python scripts/eval_decomposition.py --live          # 再加真模型：24 条复合 + 12 条单问抽样
    python scripts/eval_decomposition.py --live --repeat 2 --cache .agents/decompose-cache.json

退出码：0 正常；2 找不到标注文件。
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

from scripts.eval_retrieval import load_queries  # noqa: E402  标注校验只有一份，别处不要重写

QUERIES = ROOT / "tests" / "fixtures" / "eval_queries.json"

# 哪些 shape 的「复合性」写在**问句文本**里（闸门该抓的）：
# `two_parts` / `three_parts` / `gap_in_middle` / `four_parts` / `multi_gap` 都是一个问句里
# 问了几件事；`followup` / `followup_partial` 的问句文本只有一问，复合性在 `history` 里
# （那是 `resolve_query` 的地盘），闸门**不该**为它们触发，也不该算进闸门的分母。
TEXT_COMPOUND_SHAPES = {"two_parts", "three_parts", "gap_in_middle", "four_parts", "multi_gap"}


def load_decompose_cache(path):
    """读回上一轮拆出来的子问。文件坏了就当空缓存（缓存坏掉不该让整轮评测失败）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): [str(part) for part in value]
            for key, value in data.items() if isinstance(value, list)}


def save_decompose_cache(path, cache):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cached_decompose(decompose, cache):
    """按问句缓存拆问结果：对比不同开关时必须用同一批拆问，否则比的是模型的随机性。"""
    def cached(question):
        if question not in cache:
            cache[question] = list(decompose(question))
        return list(cache[question])
    return cached


def _pct(value):
    return "无数据" if value is None else f"{value * 100:.1f}%"


def measure_gate(queries):
    """闸门的召回与误判。闸门只决定「要不要花一次调用」，不决定怎么拆。

    分母口径：**问句文本里就有多件事**的那些样本（见 `TEXT_COMPOUND_SHAPES`）；
    多轮样本的问句文本只有一问，它们的复合性在 `history` 里，不该算闸门的漏判。
    """
    compound = [query for query in queries
                if query.get("shape") in TEXT_COMPOUND_SHAPES]
    followup = [query for query in queries
                if query["kind"] == "partial" and query.get("shape") not in TEXT_COMPOUND_SHAPES]
    single = [query for query in queries if query["kind"] != "partial"]
    caught = [query for query in compound if decompose_mod.looks_compound(query["question"])]
    false_alarms = [query for query in single if decompose_mod.looks_compound(query["question"])]
    followup_triggered = [query for query in followup
                          if decompose_mod.looks_compound(query["question"])]
    return {
        "compound": len(compound),
        "compound_caught": len(caught),
        "compound_missed": [query["id"] for query in compound if query not in caught],
        "followup": len(followup),
        "followup_triggered": [query["id"] for query in followup_triggered],
        "single": len(single),
        "single_false_alarm": len(false_alarms),
        "single_false_alarm_ids": [query["id"] for query in false_alarms],
        "call_rate": (len(caught) + len(false_alarms) + len(followup_triggered))
                     / max(1, len(queries)),
    }


def measure_live(queries, repeats=1, cache=None, limit_single=12):
    """真跑模型：复合样本全跑，单问只抽样的那几条（单问本来就不该触发调用）。

    解析失败没有别的办法看见 —— `decompose_question()` 会把解析失败吞成「退回原问句」，
    所以这里把 `parse_parts` 包一层记录原始输出，才能算出「模型输出根本没法解析」的比例。
    """
    import src.decompose as decompose_mod

    raw_seen = {}
    original_parse = decompose_mod.parse_parts

    def recording_parse(raw, max_parts=None):
        result = original_parse(raw, max_parts=max_parts)
        raw_seen[raw] = result
        return result

    decompose_mod.parse_parts = recording_parse
    try:
        decompose = cached_decompose(decompose_mod.decompose_question, cache)
        compound = [query for query in queries if query.get("shape") in TEXT_COMPOUND_SHAPES]
        followup = [query for query in queries
                    if query["kind"] == "partial" and query.get("shape") not in TEXT_COMPOUND_SHAPES]
        single = [query for query in queries if query["kind"] != "partial"][:limit_single]

        rows, durations = [], []
        for query in compound + followup + single:
            started = time.perf_counter()
            parts = decompose(query["question"])
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            gold = [part.get("text") or "" for part in (query.get("parts") or [])]
            rows.append({
                "id": query["id"],
                "kind": query["kind"],
                "gate": decompose_mod.looks_compound(query["question"]),
                "n_gold": len(gold) if gold else 1,
                "n_got": len(parts),
                "elapsed": elapsed,
                "parts": parts,
            })
        # 稳定性：同一批问题再拆一遍，看 temperature=0 下是不是逐字一致
        unstable = []
        for _ in range(max(0, repeats - 1)):
            for query in compound:
                again = decompose(query["question"])
                first = next(row["parts"] for row in rows if row["id"] == query["id"])
                if again != first:
                    unstable.append(query["id"])
    finally:
        decompose_mod.parse_parts = original_parse

    compound_rows = [row for row in rows if row["kind"] == "partial"
                     and row["id"] in {query["id"] for query in compound}]
    followup_rows = [row for row in rows if row["id"] in {query["id"] for query in followup}]
    single_rows = [row for row in rows if row["id"] in {query["id"] for query in single}]
    diffs = [row["n_got"] - row["n_gold"] for row in compound_rows]
    return {
        "rows": rows,
        "duration_median": statistics.median(durations) if durations else None,
        "duration_p90": sorted(durations)[int(len(durations) * 0.9)] if durations else None,
        "compound_exact": sum(1 for diff in diffs if diff == 0),
        "compound_short": sum(1 for diff in diffs if diff < 0),
        "compound_long": sum(1 for diff in diffs if diff > 0),
        "compound_collapsed": sum(1 for row in compound_rows if row["n_got"] <= 1),
        "followup_over_split": sum(1 for row in followup_rows if row["n_got"] > 1),
        "followup_rows": len(followup_rows),
        "single_over_split": sum(1 for row in single_rows if row["n_got"] > 1),
        "single_rows": len(single_rows),
        "parse_failures": sum(1 for _raw, parts in raw_seen.items() if not parts),
        "raw_calls": len(raw_seen),
        "unstable": unstable,
    }


def print_gate(result):
    print("=== 闸门（looks_compound：只决定要不要花一次调用）===")
    print(f"  文本复合问句 {result['compound']} 条：判「可能是复合」{result['compound_caught']} 条"
          f"（{_pct(result['compound_caught'] / max(1, result['compound']))}）"
          + (f"；漏判 {result['compound_missed']}" if result["compound_missed"] else "；无漏判"))
    print(f"  单问 {result['single']} 条：误判 {result['single_false_alarm']} 条"
          f"（{_pct(result['single_false_alarm'] / max(1, result['single']))}）"
          + (f"；{result['single_false_alarm_ids']}" if result["single_false_alarm_ids"] else ""))
    print(f"  多轮样本 {result['followup']} 条（复合性在 history 里，不该触发闸门）："
          f"触发 {len(result['followup_triggered'])} 条"
          + (f"（{result['followup_triggered']}）" if result["followup_triggered"] else ""))
    print(f"  预计调用率：{_pct(result['call_rate'])}（每条白花的调用都是 0.7 s 上下的延迟）")


def print_live(result):
    print()
    print("=== 真模型拆问（只量结构，不量「拆对率」）===")
    print(f"  调用 {result['raw_calls']} 次；延迟中位 "
          f"{result['duration_median']:.2f} s / p90 {result['duration_p90']:.2f} s"
          if result["duration_median"] is not None else "  没有调用")
    print(f"  复合样本条数：与 gold 相同 {result['compound_exact']} 条、偏少 "
          f"{result['compound_short']} 条、偏多 {result['compound_long']} 条；"
          f"只拆出一条（等于没拆）{result['compound_collapsed']} 条")
    print(f"  单问被误拆：{result['single_over_split']}/{result['single_rows']}"
          f"；多轮样本被误拆：{result['followup_over_split']}/{result['followup_rows']}"
          f"（它们的问句文本只有一问，拆开就是多花钱）")
    print(f"  模型输出无法解析：{result['parse_failures']}/{result['raw_calls']} 次"
          f"（这些会静默退回原问句，所以单独数出来）")
    if result["unstable"]:
        print(f"  同一问题两次拆问不一致：{sorted(set(result['unstable']))}（temperature=0 却有抖动）")
    print()
    print("  逐条（gold 条数 → 拆出条数）：")
    for row in result["rows"]:
        flag = "" if row["n_got"] == row["n_gold"] else "  ← 条数不符"
        print(f"    {row['id']:<5} {row['kind']:<8} {row['n_gold']} → {row['n_got']}"
              f"  {row['elapsed']:.2f}s{flag}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="拆问（query 分解）的结构测量")
    parser.add_argument("--queries", default=str(QUERIES), help="标注文件，默认合成评测集")
    parser.add_argument("--live", action="store_true", help="真调模型（花 API 额度）")
    parser.add_argument("--repeat", type=int, default=1,
                        help="同一批复合问题重复拆几次，看 temperature=0 稳不稳，默认 1")
    parser.add_argument("--limit-single", type=int, default=12,
                        help="单问抽样几条一起跑（它们本不该触发调用），默认 12")
    parser.add_argument("--cache", default=None, help="拆问结果缓存 JSON（对比两轮时必须带）")
    parser.add_argument("--json", default=None, help="明细落盘")
    args = parser.parse_args(argv)

    global decompose_mod
    import src.decompose as decompose_mod

    path = Path(args.queries)
    if not path.exists():
        print(f"错误：标注文件不存在：{path}")
        return 2
    queries = load_queries(path)

    gate = measure_gate(queries)
    print_gate(gate)

    payload = {"gate": gate}
    if args.live:
        if decompose_mod.client is None:
            print("\n错误：没有配置 API_KEY，--live 跑不了（闸门部分已经在上面了）。")
            return 2
        cache = load_decompose_cache(args.cache) if args.cache else {}
        live = measure_live(queries, repeats=args.repeat, cache=cache,
                            limit_single=args.limit_single)
        print_live(live)
        payload["live"] = live
        if args.cache:
            save_decompose_cache(args.cache, cache)
            print(f"\n拆问结果已缓存：{args.cache}（{len(cache)} 条）")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"明细已落盘：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
