#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真实语料的检索基线（只记数字）：切窗诊断 + 距离带。

为什么单独一个脚本
------------------
`scripts/eval_retrieval.py` 靠人工标注的 gold 算 recall / MRR / 归属正确率；**真实语料没有
gold 标注**，那几个指标到这里没有意义，硬套只会得到看起来像「检索坏了」的假数字。真实语料
能诚实测到的只有两类：

1. **切窗诊断**：真实正文按当前 `CHUNK_SIZE` 切完，有多少块超过 embedding 的 512 token ——
   超出的部分会被 tokenizer 静默截断，不参与向量；
2. **距离带**：拿**真实提问**（状态里用户真问过的话）做一次原始查询（不经阈值过滤），看最佳
   距离落在哪、有多少条超过 `MAX_DISTANCE`（这些在线上会被关键词兜底接走）。

顺带用评测标注里的负样本问题当「无关提问」的代理，给距离带的另一侧一个参照 —— 它是**代理**，
不是真实语料的负样本（真实语料没有 gold，也就没有真正的负样本）。

安全约定
--------
- **只读、不落文件、不打印正文**：语料读自 `state/*.json` 的 `files[].file_text`，向量库用
  `chromadb.EphemeralClient`（绝不碰真实 `chroma_db/`）。真实语料只记数字，不进仓库。
- **默认 `--no-hyde`**（可重复、不花额度）；`--hyde` 才走真实大模型，会花 API 额度。线上走
  的是 HyDE，两条路径的距离不可直接互换 —— 要重标 `MAX_DISTANCE` 得用 `--hyde` 那一版。

用法
----
    python scripts/eval_real_corpus.py                     # 只打印数字，不花额度
    python scripts/eval_real_corpus.py --hyde              # 线上口径（花 API 额度）
    python scripts/eval_real_corpus.py --state-dir /tmp/state

退出码：0 正常；2 输入不合法（扫不到状态文件 / 没有可测对话）。
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_retrieval import (  # noqa: E402
    DEFAULT_QUERIES, cached_generate, exceeds_model_window, hyde_self_check)
from src.config import logger  # noqa: E402


def conversations_from_state(state_dir):
    """读真实状态，返回 [{"conv_id", "files": [(名, 正文)], "questions": [str]}, ...]。

    只收**有文件的**对话（没文件的没有可检索的东西）；坏状态文件跳过并告警 —— 不能因为
    一个坏文件就漏掉别的对话。返回按 conv_id 排序，让输出可复现。
    """
    state_dir = Path(state_dir)
    paths = sorted(state_dir.glob("*.json")) if state_dir.exists() else []
    found = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("状态文件读取失败，跳过 %s：%s", path, e)
            continue
        convs = data.get("conversations") if isinstance(data, dict) else None
        if not isinstance(convs, dict):
            continue
        for conv_id, conv in convs.items():
            if not isinstance(conv, dict):
                continue
            files = [(str(f.get("file_name") or ""), str(f.get("file_text") or ""))
                     for f in (conv.get("files") or []) if isinstance(f, dict)]
            files = [(name, text) for name, text in files if text]
            if not files:
                continue
            questions = []
            for message in conv.get("messages") or []:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                text = str(message.get("content") or "").strip()
                if text:
                    questions.append(text)
            found.append({"conv_id": str(conv_id), "files": files, "questions": questions})
    return sorted(found, key=lambda item: item["conv_id"])


def negative_questions(queries_path):
    """从评测标注里取负样本问题，当真实语料的「无关提问」代理。

    这些合成负样本的本意是「语料里根本没有答案」，套到真实语料上只是给「无关那一侧的距离
    带」一个参照 —— 是代理，不是真实语料的负样本，对外必须这么讲。
    """
    data = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    out = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or item.get("kind") != "negative":
            continue
        text = str(item.get("question") or "").strip()
        if text:
            out.append(text)
    return out


def band_stats(distances, threshold):
    """距离带汇总：条数 / min / 中位 / max / 超过阈值的条数与比例。

    超阈值用的是线上口径：`dist <= MAX_DISTANCE` 保留，`> ` 才会被丢掉、落进关键词兜底。
    """
    values = [float(d) for d in distances if d is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None,
                "over_threshold": 0, "over_rate": None}
    over = [d for d in values if d > threshold]
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "over_threshold": len(over),
        "over_rate": len(over) / len(values),
    }


def best_distance(collection, query_text, top_k, count):
    """原始查询（不经阈值过滤）的最佳距离；查不到或出错返回 None。"""
    try:
        result = collection.query(query_texts=[query_text], n_results=min(top_k, count))
    except Exception as e:
        logger.warning("查询失败，跳过这条：%s", str(e)[:80])
        return None
    dists = (result.get("distances") or [[]])[0]
    return min(dists) if dists else None


def _num(value):
    return "无数据" if value is None else f"{value:.4f}"


def _print_band(title, stats, threshold):
    if stats["count"] == 0:
        print(f"{title}：无数据")
        return
    print(f"{title}：n={stats['count']} min={_num(stats['min'])} "
          f"中位={_num(stats['median'])} max={_num(stats['max'])}"
          f" | 超过 {threshold}：{stats['over_threshold']}/{stats['count']}"
          f"（{stats['over_rate'] * 100:.1f}%）")


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实语料检索基线（只读、只记数字）")
    parser.add_argument("--state-dir", default=None, help="状态目录，默认 state.STATE_DIR")
    parser.add_argument("--hyde", action="store_true",
                        help="走真实 HyDE（花 API 额度；线上就是这个口径）")
    parser.add_argument("--top-k", type=int, default=5, help="原始查询取几块，默认 5")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES),
                        help="无关提问代理的标注文件（只取负样本）")
    parser.add_argument("--rerank", action="store_true",
                        help="开启跨编码器精排（首次会下载约 1.1GB 权重）")
    parser.add_argument("--rerank-min-score", type=float, default=None,
                        help="覆盖 RERANK_MIN_SCORE")
    args = parser.parse_args(argv)

    from src import state as state_mod
    state_dir = Path(args.state_dir) if args.state_dir else Path(state_mod.STATE_DIR)

    conversations = conversations_from_state(state_dir)
    if not conversations:
        print(f"没有扫到任何有文件的对话：{state_dir}")
        print("   这通常说明 --state-dir 指错了。")
        return 2

    import chromadb
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_DISTANCE
    from src.pdf_ingestion import split_files
    import src.retrieval as retrieval

    if not args.hyde:
        # 与合成评测一致：把 HyDE 换成恒等函数，可重复、不花额度
        retrieval._generate_hyde = lambda question: question  # noqa: SLF001

    if args.rerank:
        retrieval.RERANK_ENABLED = True
        if args.rerank_min_score is not None:
            retrieval.RERANK_MIN_SCORE = args.rerank_min_score

    negatives = negative_questions(args.queries)
    embedding_fn = retrieval._get_embedding_fn()
    # 同一句话只生成一次假设文本、且跨对话复用（助手函数住在 eval_retrieval，别处也用它）
    hyde_cache = {}
    generate = cached_generate(retrieval._generate_hyde, hyde_cache)  # noqa: SLF001

    # 切窗诊断要 tokenizer；拿不到就跳过这一项，不让整次统计失败
    encode_len, max_seq_length = None, None
    try:
        model = embedding_fn._model          # SentenceTransformer（chromadb 的 EF 挂在 _model）
        tokenizer = model.tokenizer
        max_seq_length = int(getattr(model, "max_seq_length", 512))
        encode_len = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    except Exception as e:
        print(f"（切窗诊断不可用，只报距离带：{e}）")

    print(f"状态目录：{state_dir}")
    print(f"检索模式：{'HyDE（真实调用大模型）' if args.hyde else 'no-hyde（query = 原问题）'}")
    print(f"切片：CHUNK_SIZE={CHUNK_SIZE}（overlap={CHUNK_OVERLAP}）"
          f"；阈值 MAX_DISTANCE={MAX_DISTANCE}")
    if retrieval.RERANK_ENABLED:
        print(f"精排：开启（{retrieval.RERANK_MODEL}；候选 {retrieval.RERANK_CANDIDATES} → "
              f"保留 {retrieval.RERANK_TOP_N}；分数下限 {retrieval.RERANK_MIN_SCORE}）")
        print("  注意：下面的距离带描述的是**候选池**、不是最终进 context 的块；"
              "真实语料的精排分数带尚未实现（见 docs/retrieval-baseline-2026-10-06.md）。")
    print()

    if args.hyde:
        # 先花一次调用确认 HyDE 真的生效再往下跑：失败时它会静默退回原问题，那样整轮
        # 「线上口径」其实都是 no-hyde 的数字（2026-10-06 实测踩过）。
        probe = next((q for c in conversations for q in c["questions"]), None)
        if probe is None and negatives:
            probe = negatives[0]
        if probe is not None:
            ok, text = hyde_self_check(probe, generate)
            if not ok:
                print("错误：HyDE 没有真正生效 —— 生成的文本与原问题相同，"
                      "这一轮的距离不是线上口径，已中止。")
                print("   排查：上面有没有「HyDE 返回空内容 / 生成失败」告警？"
                      "推理模型的 max_tokens 被 reasoning 吃光就会这样。")
                return 2
            print(f"HyDE 自检通过：首次生成 {len(text)} 字，与问题不同")
            print()

    real_distances, proxy_distances, over_lengths = [], [], []
    total_chunks = total_over = 0
    for conv in conversations:
        pairs = split_files(conv["files"])
        chunks = [chunk for chunk, _ in pairs]
        if not chunks:
            print(f"[{conv['conv_id']}] 切片后没有 chunk，跳过")
            continue
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(name="real_eval",
                                                     embedding_function=embedding_fn)
        collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))],
                       metadatas=[retrieval.chroma_meta(meta) for _, meta in pairs])

        conv_over = ([length for length, _ in exceeds_model_window(chunks, max_seq_length,
                                                                  encode_len)]
                     if encode_len is not None else [])
        over_lengths += conv_over
        total_chunks += len(chunks)
        total_over += len(conv_over)

        # 真实提问与无关代理走同一条查询路径（--hyde 时都用 HyDE 文本查）
        def _ask(question):
            return best_distance(collection, generate(question), args.top_k, len(chunks))

        conv_real = [d for d in (_ask(q) for q in conv["questions"]) if d is not None]
        conv_proxy = [d for d in (_ask(q) for q in negatives) if d is not None]
        real_distances += conv_real
        proxy_distances += conv_proxy

        over_text = f"{len(conv_over)}/{len(chunks)}" if encode_len is not None else "未测"
        print(f"[{conv['conv_id']}] 文件={len(conv['files'])} 块={len(chunks)} 超窗={over_text}"
              f" 真实提问={len(conv['questions'])}")
        _print_band("    真实提问最佳距离", band_stats(conv_real, MAX_DISTANCE), MAX_DISTANCE)
        if negatives:
            _print_band("    无关代理最佳距离", band_stats(conv_proxy, MAX_DISTANCE), MAX_DISTANCE)

    print()
    print("=== 真实语料基线（只记数字）===")
    if encode_len is not None:
        print(f"chunk 总数：{total_chunks}；超过 {max_seq_length} token 的："
              f"{total_over}/{total_chunks}")
        if over_lengths:
            print(f"  超出长度：min={min(over_lengths)} max={max(over_lengths)} "
                  f"中位={statistics.median(over_lengths):.0f}")
    else:
        print(f"chunk 总数：{total_chunks}（切窗诊断不可用）")
    _print_band("真实提问最佳距离", band_stats(real_distances, MAX_DISTANCE), MAX_DISTANCE)
    if negatives:
        _print_band("无关提问代理距离", band_stats(proxy_distances, MAX_DISTANCE), MAX_DISTANCE)
        print(f"  （代理 = 评测标注里的 {len(negatives)} 条负样本，套到真实语料上；"
              f"真实语料没有 gold，这不是它的真实负样本）")
    if args.hyde:
        print(f"HyDE 调用：{len(hyde_cache)} 次（同一问题只生成一次；"
              f"{len(real_distances) + len(proxy_distances)} 条查询复用它）")
    else:
        print("提示：这版是 no-hyde 口径，而线上走 HyDE —— 两者距离不可直接互换，"
              "要重标 MAX_DISTANCE 得用 --hyde 重跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
