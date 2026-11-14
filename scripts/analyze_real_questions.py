#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实用户提问盘点：闸门触发 + 线上装配口径下的文件覆盖。

为什么单独一个脚本
------------------
前几轮所有「复合问」样本都是**自己造的**（先是模板拼的，后来是手写的）。真实用户到底怎么问，
一直是个空白 —— 于是本轮把 `state/` 里**真实产生过的**提问捞出来盘一遍，只回答三个能机械核对
的问题：

1. **真实流量里到底有多少复合问**：`decompose.looks_compound` 逐条过一遍。这条同时量了两件事
   —— 拆问功能在真实流量上有多大机会被用到（触发率），以及闸门有没有把单问误判成复合问
   （误判只要白花一次调用，漏判是整类提问静默失效）。
2. **真实流量的「多部分」需求是什么形状**：`looks_compound` 只认「两件事分别是什么」那种写法，
   而真实提问里更常见的是**枚举/汇总型**（「分别有哪些实验」「列出所有的实验」「用三点总结
   主要内容」「帮我整理第一章的内容」）。这类需求要的是**跨全部文件取样**，不是按子问分路 ——
   所以脚本单独数一遍「用了枚举/汇总措辞的提问有几条」。
3. **线上装配口径下装进去几个文件**：走 `llm.retrieve_and_build_context`（线上同一个函数、
   同一个 `MAX_CONTEXT_LENGTH` 上限、同一个 `dynamic_top_k`），只数「装配后的 context 里有几个
   文件」，不数检索返回了几个 —— 检索回来再多，装不下就等于没给模型。

刻意不做的事：不判回答对不对（真实日志没有 gold），不写盘（只读 `state/`，明细只进 `--json`）。

用法：
    python scripts/analyze_real_questions.py
    python scripts/analyze_real_questions.py --state-dir /tmp/state --json /tmp/real.json
    python scripts/analyze_real_questions.py --hyde-cache .agents/real-log-hyde.json  # 线上 HyDE 口径

退出码：0 正常；2 状态目录不存在 / 没有任何真实提问（不假装跑出一张空表）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_real_corpus import conversations_from_state  # noqa: E402

# 「枚举 / 汇总」措辞：真实流量里承担「多部分」需求的主力。
# 只做**检出**用（数条数、给人工复核一个抓手），不当正确性判据 —— 措辞表永远是脆的，
# 所以只用来分组，不用来判对错。
ENUMERATION_MARKERS = (
    "列出", "全部", "所有", "分别", "几个", "哪些", "总结", "整理", "归纳", "概述", "主要内容",
)


def real_questions(conversations):
    """把状态里的真实提问摊平：`[{"conv_id", "question"}, ...]`（保持原始顺序）。

    只收 `role == "user"` 的消息；空内容跳过。**不在这里去重** —— 同一个问题被问了两遍
    （真实日志里就有）本身是信息：说明上一次没答好。
    """
    rows = []
    for conv_id, conv in (conversations or {}).items():
        for message in conv.get("messages") or []:
            if (message.get("role") or "") != "user":
                continue
            question = str(message.get("content") or "").strip()
            if question:
                rows.append({"conv_id": str(conv_id), "question": question})
    return rows


def gate_report(questions, gate):
    """闸门在真实提问上的触发情况；返回 `(触发条数, 触发列表, 明细)`。"""
    rows = []
    for row in questions or []:
        try:
            hit = bool(gate(row["question"]))
        except Exception:                       # 闸门不许把整个盘点带走
            hit = False
        rows.append({"conv_id": row["conv_id"], "question": row["question"], "gate": hit})
    triggered = [row for row in rows if row["gate"]]
    return len(triggered), triggered, rows


def looks_like_enumeration(question, markers=ENUMERATION_MARKERS):
    """这条提问是不是「枚举 / 汇总」措辞。只用于分组统计，不参与判对错。"""
    text = str(question or "")
    return any(marker in text for marker in markers)


def assembly_report(used, file_names):
    """装配后的 context 覆盖了几个文件：`(文件数, 总文件数, 缺哪些, 块数)`。

    `used` 是 `retrieve_and_build_context` 返回的 used_blocks（每项带 `file_name`）。
    「装进去」才算数：检索返回 39 块但装配受 `MAX_CONTEXT_LENGTH` 截断时，
    模型看到的文件就是少的那几个。
    """
    names = sorted({entry.get("file_name") for entry in used or [] if entry.get("file_name")})
    total = sorted({name for name in file_names or [] if name})
    missing = [name for name in total if name not in names]
    return len(names), len(total), missing, len(used or [])


def summarize(rows, gate_rows):
    """一句话汇总（纯函数，单测用）：真实提问数、闸门触发数、枚举型条数、对话数。"""
    return {
        "questions": len(rows or []),
        "conversations": len({row["conv_id"] for row in rows or []}),
        "gate_hits": sum(1 for row in gate_rows or [] if row["gate"]),
        "enumeration": sum(1 for row in rows or [] if looks_like_enumeration(row["question"])),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实用户提问盘点（只读 state/，不写盘）")
    parser.add_argument("--state-dir", default=None,
                        help="状态目录，默认 src.state.STATE_DIR")
    parser.add_argument("--hyde-cache", default=None,
                        help="假设文本缓存 JSON；给了就走线上 HyDE 口径（会真实调用大模型补齐缺失项）")
    parser.add_argument("--json", default=None, help="明细写到这个 JSON（可选）")
    args = parser.parse_args(argv)

    from src import state as state_mod

    state_dir = Path(args.state_dir) if args.state_dir else Path(state_mod.STATE_DIR)
    if not state_dir.exists():
        print(f"错误：状态目录不存在：{state_dir}")
        return 2

    conversations = conversations_from_state(state_dir)
    if not conversations:
        print(f"错误：这个状态目录里没有任何带文件的对话：{state_dir}")
        return 2

    # `conversations_from_state` 给的是 [(文件名, 正文)]；这里还要 messages，所以自己再读一遍。
    raw = {}
    for path in sorted(state_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for conv_id, conv in (data.get("conversations") or {}).items():
            if isinstance(conv, dict):
                raw[str(conv_id)] = conv

    rows = real_questions(raw)
    if not rows:
        print(f"错误：这个状态目录里没有任何真实提问（只有上传的文件）：{state_dir}")
        return 2

    from src.decompose import looks_compound

    hit_count, triggered, gate_rows = gate_report(rows, looks_compound)
    stats = summarize(rows, gate_rows)
    print(f"状态目录：{state_dir}")
    print(f"真实提问 {stats['questions']} 条 / 对话 {stats['conversations']} 个")
    print(f"复合问闸门（looks_compound）触发 **{hit_count}/{stats['questions']}**"
          f" —— 这就是拆问在真实流量上的机会")
    print(f"枚举 / 汇总措辞的提问 {stats['enumeration']} 条"
          f"（这类要的是跨全部文件取样，不是按子问分路）")
    # 闸门靠措辞判断意图，换语言/换语料可能整片失效 —— 失效是 fail-safe 的（退回改造前行为），
    # 但**必须看得见**：HyDE 当年就是静默失效、很久没人发现。
    if stats["questions"] and hit_count == 0:
        print("  [警告] 复合问闸门在整批真实提问上 **0 触发** —— 若提问语言与 "
              "`GATE_PACK` 不匹配，拆问会静默不生效（行为退回改造前，但别以为它在工作）")
    if stats["questions"] and stats["enumeration"] == 0:
        print("  [警告] 枚举/汇总闸门同样 0 触发 —— 覆盖取样永远不会被激活")
    print()
    for row in gate_rows:
        mark = "复合" if row["gate"] else "单问"
        enum = "枚举/汇总" if looks_like_enumeration(row["question"]) else "       "
        print(f"  [{mark}] [{enum}] {row['question']}")

    # 线上装配口径下的文件覆盖（真跑检索，需要 embedding；HyDE 可选）
    import chromadb

    import src.llm as llm
    import src.retrieval as retrieval

    if args.hyde_cache:
        from scripts.eval_retrieval import cached_generate, load_hyde_cache, save_hyde_cache

        hyde_path = Path(args.hyde_cache)
        hyde = load_hyde_cache(hyde_path) if hyde_path.exists() else {}
        retrieval._generate_hyde = cached_generate(retrieval._generate_hyde, hyde)  # noqa: SLF001
    else:
        hyde, hyde_path = None, None
        retrieval._generate_hyde = lambda question: question  # noqa: SLF001  no-hyde 口径

    embedding_fn = retrieval._get_embedding_fn()
    coverage = []
    print()
    print("=== 线上装配口径下的文件覆盖（检索→按块边界装配→8000 字符上限）===")
    for index, (conv_id, conv) in enumerate(sorted(raw.items()), 1):
        files = sorted((f.get("file_name") or "", f.get("file_text") or "")
                       for f in conv.get("files") or [] if isinstance(f, dict))
        files = [(name, text) for name, text in files if text]
        if not files:
            continue
        collection = chromadb.EphemeralClient().get_or_create_collection(
            name=f"real_{index}", embedding_function=embedding_fn)
        pairs = retrieval._pairs_with_meta(files, "")
        collection.add(documents=[chunk for chunk, _ in pairs],
                       ids=[f"i{i}" for i in range(len(pairs))],
                       metadatas=[retrieval.chroma_meta(meta) for _, meta in pairs])
        file_names = sorted({(meta or {}).get("file_name") for _, meta in pairs})
        budget = max(10, len(files) * 3)          # 与 src/server.py 的 dynamic_top_k 同一个式子
        for row in [r for r in gate_rows if r["conv_id"] == conv_id]:
            _context, used = llm.retrieve_and_build_context(
                row["question"], "", collection, True, budget, files,
                decompose=None, history=None)
            seen, total, missing, blocks = assembly_report(used, file_names)
            coverage.append({"conv_id": conv_id, "question": row["question"],
                             "blocks": blocks, "files_seen": seen, "files_total": total,
                             "missing": missing})
            flag = "" if not missing else f"  ← 缺 {len(missing)} 份"
            print(f"  [{seen}/{total} 文件 · {blocks} 块] {row['question']}{flag}")

    if hyde is not None and hyde_path is not None:
        save_hyde_cache(hyde_path, hyde)
        print(f"\nHyDE 缓存：{hyde_path}（{len(hyde)} 条）")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": stats, "questions": gate_rows, "coverage": coverage},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
