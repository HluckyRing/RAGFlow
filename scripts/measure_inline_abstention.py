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

复合问题的形状（2026-11-08 起）
--------------------------------
第一批 8 条只测了「一个问句两问」。产品里更难的是别的几种形状，靠 `shape` / `parts` /
`history` / `expect_refused` 四个字段驱动：
- `three_parts` / `four_parts`：三问、四问 —— 子问题一多，漏答其中一问的概率就上来了；
- `gap_in_middle`：三问、不可答那问在中间 —— 最容易「答完第一问就收尾」，
  用 `post_gap_coverage()` 量**缺口后面那一问**还活着没有；
- `multi_gap`：四问、两处以上缺口 —— 缺口不止一处时，容易只提第一个就把后面全丢掉；
- `followup` / `followup_partial`：多轮补问（后者是「上一轮答了一半、本轮接着追问」）——
  先过 `resolve_query` 做指代消解（与线上同一路径），再来判它该拒还是该答。
`classify_partial()` 因此从五桶扩到九桶（多了「漏答可答项」两种与「该拒却没写标记 / 直接硬答」两种），
`summarize_partial()` 按形状拆开算 —— 该答的样本和该拒的样本绝不混进同一个分母。

用法（会真实调用大模型，约 46 次生成）：
    python scripts/measure_inline_abstention.py
    python scripts/measure_inline_abstention.py --json /tmp/inline.json
    python scripts/measure_inline_abstention.py --fill-hyde   # 缓存里缺的假设文本现场补齐
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
    DEFAULT_CORPUS, DEFAULT_QUERIES, build_context, load_corpus, load_queries, normalize)
from src.prompts import REFUSAL_MARKER  # noqa: E402

# 两臂共用的检索预算：拆问臂「每问各查 top_k 条再轮流分名额到 top_k 块」，
# 整句臂就是「一次查 top_k 条」。预算不一样的话，比的是 token 不是检索质量。
TOP_K = 5

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


# 「这里没有」的常见说法：判「有没有指出缺口」时，主题词之外还得见到这些词之一。
# 刻意宽松（宁可把「其实没说清楚」算成说了），因为这一步只用来分桶，不当硬门禁。
ABSENCE_WORDS = (
    "没有", "没写", "未写", "未提", "没提", "未涉及", "没有涉及", "不涉及",
    "未说明", "没有说明", "未给出", "没有给出", "无法", "未提供", "没有提供",
    "不包含", "未包含", "未规定", "没有规定",
)

# 分桶口径（2026-11-08 起）。前半截是「整条被标成拒答」的两种，后半截是正常作答的四种。
BUCKETS = (
    "refused_all",             # 标了拒答标记，可答项也一条没答 —— 最糟
    "refused_despite_answer",  # 标了拒答标记，但正文答了可答项 —— 用户仍看到「知识库里没有」
    "refused_without_marker",  # 这条本来就答不了：正文说了「材料里没有」，却没写标记（前端不显示拒答标签）
    "answered_without_refusal",  # 这条本来就答不了：直接硬答/编造，连「没有」都没说
    "neither",                 # 有可答项，但一条没答到、也没提缺口 —— 答偏了
    "dropped_part_with_gap",   # 漏答了某个可答项，但确实点明了缺口
    "dropped_part_silent",     # 漏答了某个可答项，且对缺口只字不提
    "answered_with_gap",       # 可答项全答到 + 点明缺口 —— 理想
    "answered_silently",       # 可答项全答到，但对缺口只字不提（样本本身没有缺口时这也算达标）
)
# 该答却整条标成拒答 = 过度拒答
OVER_REFUSAL_BUCKETS = ("refused_all", "refused_despite_answer")
# 漏答了本可答的子问题
DROPPED_PART_BUCKETS = ("dropped_part_with_gap", "dropped_part_silent")
# 可答项全答到（且没被标成拒答）
FULL_COVERAGE_BUCKETS = ("answered_with_gap", "answered_silently")
# 该整条拒答的样本里，「行为对、契约也对」的那个桶
REFUSED_OK_BUCKETS = ("refused_all",)


# 判词面命中前要抹掉的**装饰性**字符：Markdown 强调、各种引号、括号。
# 2026-11-08 实测：模型明明答对了，却写成「保留 **30 天**」「窑温“通常分三段缓慢抬升”」
# 「需在**加班之日起 6 个月内**使用」—— 这些字符会把锚点切成两半，于是正确回答被判成漏答。
# 只抹装饰，**不做**同义改写（「可以折算」≠「可折算」仍算不同）：那会从「太脆」滑到「太松」，
# 而「太松」比「太脆」更危险 —— 它把失败藏起来。
DECORATION_CHARS = "*_`#>“”‘’「」『』《》〈〉\"'（）()［］[]【】"


def _norm(text):
    """判「回答里有没有这个片段」用的归一化。

    比 `eval_retrieval.normalize`（检索指标用的那套）多一步：抹掉装饰性字符。
    两套刻意分开 —— 改这里的口径不该影响检索基线里的历史数字。
    """
    united = normalize(text or "")
    return "".join(ch for ch in united if ch not in DECORATION_CHARS)


def contains(text, term):
    """回答里有没有这个片段（两侧归一化后再比）。

    直接 `term in answer` 会在「模型写『每年5天』、标注写『每年 5 天』」时判成漏答 ——
    这种假失败比漏测更糟（会逼着人去改提示词修一个并不存在的问题），所以空格/全半角先抹平。
    """
    needle = _norm(term)
    return bool(needle) and needle in _norm(text)


def flags_gap(text, absent_terms):
    """有没有指出缺口：主题词出现 **且** 出现「没有/未涉及/不包含…」之一。

    只提到主题词不算 —— 那可能是在编造（「异地机房的切换窗口建议安排在周末」）。
    """
    return (any(contains(text, term) for term in (absent_terms or []) if term)
            and any(word in _norm(text) for word in ABSENCE_WORDS))


def classify_partial(answer, answer_terms, absent_terms):
    """给「部分可答」样本的回答分桶。

    2026-11-08 的第一批只分五桶（那时一条样本只有一个可答项）。复合问题扩到「三问」「缺口在
    中间」之后，一条样本可以有**多个**可答项，于是多出两种真实的失败：

    - `dropped_part_with_gap` / `dropped_part_silent`：缺口点明了（或没点明），但**某个本可以答的
      子问题被吞了**。「答完第一问就收尾」正是缺口在中间那个形状的死法，五桶口径看不见它。
    - `refused_without_marker` / `answered_without_refusal`：这条样本本来就一条都答不了
      （`expect_refused: true`），正确行为是整条拒答。前者是「说了没有但没写标记」——
      行为对、契约没守，前端不会显示拒答标签；后者是直接硬答。

    判定顺序：先看有没有拒答标记（结构化契约，优先于一切文案判断），再看可答项覆盖，
    最后看缺口有没有点明。只有单个可答项的样本仍然只会落进原来那五个桶（向后兼容）。
    """
    text = (answer or "").strip()
    answer_terms = [t for t in (answer_terms or []) if t]
    absent_terms = [t for t in (absent_terms or []) if t]
    covered = [t for t in answer_terms if contains(text, t)]
    missing = [t for t in answer_terms if t not in covered]
    gap_flagged = flags_gap(text, absent_terms)
    if looks_like_abstention(text):
        return "refused_despite_answer" if covered else "refused_all"
    if not answer_terms:
        # 这条样本本来就「一条都答不了」，只判它有没有拒、以及拒得合不合契约
        return "refused_without_marker" if gap_flagged else "answered_without_refusal"
    if not covered:
        return "neither"
    if missing:
        return "dropped_part_with_gap" if gap_flagged else "dropped_part_silent"
    return "answered_with_gap" if gap_flagged else "answered_silently"


def part_coverage(answer, parts):
    """逐个子问题判有没有答到：可答项看词面命中，不可答项看有没有点明缺口。

    这是「三问 / 缺口在中间」唯一能量到细节的粒度 —— 整条消息的桶只说明有没有点明缺口，
    说明不了**第 3 问被吞了**没有。
    """
    text = (answer or "").strip()
    rows = []
    for part in parts or []:
        answer_terms = [t for t in (part.get("answer_terms") or []) if t]
        absent_terms = [t for t in (part.get("absent_terms") or []) if t]
        if answer_terms:
            hit = [t for t in answer_terms if contains(text, t)]
            rows.append({"text": str(part.get("text") or ""), "answerable": True,
                         "ok": bool(hit), "hit": hit})
        else:
            rows.append({"text": str(part.get("text") or ""), "answerable": False,
                         "ok": flags_gap(text, absent_terms), "hit": []})
    return rows


def post_gap_coverage(answer, parts):
    """缺口**之后**那些可答子问题还有没有答到 —— 「答完第一问就收尾」的唯一探针。

    「缺口在中间」这个形状之所以要单列，就是因为模型的收尾习惯：可答/不可答/可答 三问里，
    它很容易在第二问答完「材料里没有」之后就结束，把第三问静默丢掉。整条消息的桶看起来还是
    `answered_with_gap`（完全达标），只有这个数字会露馅。
    """
    rows = part_coverage(answer, parts)
    after = []
    seen_gap = False
    for row in rows:
        if not row["answerable"]:
            seen_gap = True
            continue
        if seen_gap:
            after.append(row)
    if not after:
        return None
    return {"covered": sum(1 for row in after if row["ok"]), "total": len(after),
            "parts": [row["text"] for row in after]}


def pick_hits(question, whole, decomposed=None, decompose=None):
    """取 context 用的检索块：**整句**或**每问分路**，两臂只差这一个开关。

    为什么要有这一层：第六轮把拆问接进线上之后，「复合问答得全不全」就只能在**生成侧**量，
    而这个脚本原来直接调 `hyde_retrieve_hits_with_path` —— 它压根没有拆问这条路，于是
    「装下之后答没答全」在线上口径下是空白。

    三条纪律：

    - **不在这里重新实现拆问逻辑**：`decomposed` 是线上 `llm._retrieve_hits` 的薄包装，
      「拆出 1 条就退回整句 / 分路全空退回整句 / 每条子问过 `resolve_query`」全部由那一份
      实现决定。判断写两份必然漂，漂了评测数字就不再说明线上行为。
    - **两臂除开这个开关以外完全相同**：同一个 `top_k`、同一份装配（`build_context`）、
      同一批样本、同一份 HyDE 缓存。
    - `decompose=None` 时**连 `decomposed` 都不看一眼** —— 关着就是改造前的行为，
      一次多余调用都不许有。
    """
    if decompose is None or decomposed is None:
        return whole(question)
    return decomposed(question)


def plan_search_queries(queries, resolve):
    """把「样本问句」翻译成「检索问句」。

    多轮补问（`shape: "followup"`）必须先做指代消解，走的必须是线上同一条路
    （`src.llm.resolve_query`，由调用方传进来）—— 抽成纯函数是为了能在单测里确认
    **history 真的被传进去了**，而不是被静默扔掉（扔掉时一切看起来都正常，只是检索变差）。
    """
    return [resolve(query["question"], query.get("history") or []) for query in queries]


def gap_coverage(answer, parts):
    """逐处缺口判有没有点明 —— 扁平字段看不见「只点了第一处、把第二处丢了」。

    **这是 `parts` 存在的第二个理由**（第一个是漏答可答项）：`classify_partial` 拿的是**平铺**的
    `absent_terms`，判据是「任一个主题词出现 + 有个缺席词」——所以「四处两缺口」的样本里，
    模型只点明第一处缺口，整条消息的桶仍然是 `answered_with_gap`（看着完全达标）。
    多缺口的漏答率只能按子问题逐处算，这里就是那把尺。
    """
    text = (answer or "").strip()
    gaps = []
    for part in parts or []:
        absent_terms = [t for t in (part.get("absent_terms") or []) if t]
        if not absent_terms:
            continue
        gaps.append({"text": str(part.get("text") or ""),
                     "ok": flags_gap(text, absent_terms)})
    if not gaps:
        return None
    return {"flagged": sum(1 for gap in gaps if gap["ok"]), "total": len(gaps),
            "unflagged": [gap["text"] for gap in gaps if not gap["ok"]]}


def summarize_partial(rows):
    """按形状汇总复合问题的结果。

    纯函数（只吃 `rows`，不碰模型、不读语料），所以能在单测里喂合成行。两种口径分开算：

    - **该答的**（`expect_refused: false`）：过度拒答 / 漏答可答项 / 可答项覆盖率；
    - **该拒的**（`expect_refused: true`）：真的拒了没有 / 说了没有但没写标记 / 直接硬答。
    混在一个分母里会让「该拒的拒了」和「该答的拒了」互相抵消成没有意义的数字。
    """
    shapes = {}
    for row in rows:
        shapes.setdefault(row.get("shape") or "two_parts", []).append(row)
    summary = {}
    for shape, group in shapes.items():
        should_answer = [row for row in group if not row.get("expect_refused")]
        should_refuse = [row for row in group if row.get("expect_refused")]
        covered = sum(int((row.get("coverage") or {}).get("covered") or 0) for row in group)
        total = sum(int((row.get("coverage") or {}).get("total") or 0) for row in group)
        post = [row.get("post_gap") or {} for row in group if row.get("post_gap")]
        # 「缺口全点明」只对**该答**的样本有意义：该拒的样本正确行为是整条拒答，
        # 此时模型往往只说「材料里没有涉及」而不复述主题词（那是**对**的），拿它算缺口漏提会冤枉它。
        gaps = [row.get("gaps") or {} for row in group
                if row.get("gaps") and not row.get("expect_refused")]
        summary[shape] = {
            "n": len(group),
            "should_answer": len(should_answer),
            "should_refuse": len(should_refuse),
            "over_refused": [row["id"] for row in should_answer
                             if row["bucket"] in OVER_REFUSAL_BUCKETS],
            "dropped_part": [row["id"] for row in group
                             if row["bucket"] in DROPPED_PART_BUCKETS],
            "full_coverage": [row["id"] for row in should_answer
                              if row["bucket"] in FULL_COVERAGE_BUCKETS],
            "neither": [row["id"] for row in group if row["bucket"] == "neither"],
            "refused_ok": [row["id"] for row in should_refuse
                           if row["bucket"] in REFUSED_OK_BUCKETS],
            "refused_without_marker": [row["id"] for row in should_refuse
                                       if row["bucket"] == "refused_without_marker"],
            "answered_without_refusal": [row["id"] for row in should_refuse
                                         if row["bucket"] == "answered_without_refusal"],
            "answer_terms_covered": covered,
            "answer_terms_total": total,
            "post_gap_covered": sum(int(item.get("covered") or 0) for item in post),
            "post_gap_total": sum(int(item.get("total") or 0) for item in post),
            "gaps_flagged": sum(int(item.get("flagged") or 0) for item in gaps),
            "gaps_total": sum(int(item.get("total") or 0) for item in gaps),
            "all_gaps_flagged": [row["id"] for row in group
                                 if (row.get("gaps") or {}).get("total")
                                 and (row["gaps"].get("flagged") == row["gaps"].get("total"))],
        }
    return summary


SHAPE_LABELS = {
    "two_parts": "两问",
    "three_parts": "三问",
    "gap_in_middle": "缺口在中间",
    "four_parts": "四问",
    "multi_gap": "缺口多处",
    "followup": "多轮补问",
    "followup_partial": "多轮·部分可答",
}


def print_partial_summary(rows, log=print):
    """把 `summarize_partial` 的结果按形状打出来；返回同一份汇总，方便调用方落盘。"""
    summary = summarize_partial(rows)
    if not summary:
        return summary
    log("")
    log("=== 复合问题的形状（按形状拆开看）===")
    for shape, stats in summary.items():
        label = SHAPE_LABELS.get(shape, shape)
        log(f"[{label}] n={stats['n']}"
            f"（该答 {stats['should_answer']} / 该拒 {stats['should_refuse']}）")
        if stats["should_answer"]:
            log(f"  过度拒答          = {len(stats['over_refused'])}/{stats['should_answer']}"
                f" {stats['over_refused']}")
            log(f"  可答项全答到      = {len(stats['full_coverage'])}/{stats['should_answer']}"
                f" {stats['full_coverage']}")
        if stats["answer_terms_total"]:
            rate = stats["answer_terms_covered"] / stats["answer_terms_total"]
            log(f"  可答项覆盖率      = {stats['answer_terms_covered']}"
                f"/{stats['answer_terms_total']}（{rate:.0%}）")
        if stats["dropped_part"]:
            log(f"  漏答可答项        = {len(stats['dropped_part'])} {stats['dropped_part']}")
        if stats["neither"]:
            log(f"  可答项一条没答到  = {len(stats['neither'])} {stats['neither']}")
        if stats["should_refuse"]:
            log(f"  该拒·真的拒了     = {len(stats['refused_ok'])}/{stats['should_refuse']}"
                f" {stats['refused_ok']}")
            if stats["refused_without_marker"]:
                log(f"  该拒·说了没有但没写标记 = {len(stats['refused_without_marker'])}"
                    f" {stats['refused_without_marker']}")
            if stats["answered_without_refusal"]:
                log(f"  该拒·直接硬答     = {len(stats['answered_without_refusal'])}"
                    f" {stats['answered_without_refusal']}")
        if stats["post_gap_total"]:
            log(f"  缺口后一问存活    = {stats['post_gap_covered']}/{stats['post_gap_total']}")
        if stats["gaps_total"]:
            log(f"  缺口全点明        = {stats['gaps_flagged']}/{stats['gaps_total']}"
                f"（按子问题逐处算，扁平字段看不见漏第二处；该拒的样本不计入）")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="量「让 QA 自己拒答」的质量与延迟（会真实调用大模型）")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--hyde-cache", default=".agents/hyde-cache.json",
                        help="假设文本缓存（不给就每次重新生成，结果不可比）")
    parser.add_argument("--json", default=".agents/inline-abstain.json",
                        help="逐条明细写到这个 JSON")
    parser.add_argument("--fill-hyde", action="store_true",
                        help="缓存里缺的假设文本现场生成并写回缓存（会真实调用大模型；"
                             "不填就退化成「假设文本=原问题」，融合只剩一路，结果不可比）")
    parser.add_argument("--decompose", action="store_true",
                        help="检索走线上的「每问分路」（= DECOMPOSE_ENABLED 开的口径）；"
                             "不给就是整句一次检索（改造前口径）。两臂只差这一个开关")
    parser.add_argument("--decompose-cache", default=".agents/inline-decompose-cache.json",
                        help="拆问结果缓存；对比两轮/两臂时必须指向同一个文件，"
                             "否则拆出来的子问会变，比的是模型随机性")
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

    # 检索用的问句与「喂给模型的问句」不是一回事：多轮补问要先过 resolve_query 做指代消解
    # （和线上 server.chat 一模一样），否则「这个要多久」这种样本连检索都检索不到东西。
    search_queries = plan_search_queries(queries, llm.resolve_query)

    cache_path = Path(args.hyde_cache)
    hyde = (json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path.exists() else {})
    real_generate_hyde = retrieval._generate_hyde
    missing = [q for q in dict.fromkeys(search_queries) if q not in hyde]
    if missing and args.fill_hyde:
        for question in missing:
            try:
                text = real_generate_hyde(question)
            except Exception as e:                      # noqa: BLE001
                print(f"[HyDE] 生成失败（{question[:14]}…）：{str(e)[:80]}")
                continue
            # 推理模型的小 max_tokens 会把内容吃光，空结果被 `or question` 悄悄兜成原问题 ——
            # 那就是「假设文本=原问题」，双路融合退化成一路。缓存里必须留下真生成物。
            if not text or not text.strip() or text.strip() == question.strip():
                print(f"[HyDE] 假设文本退化（{question[:14]}…）：为空或等于原问题，不入缓存")
                continue
            hyde[question] = text
        cache_path.write_text(json.dumps(hyde, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        print(f"已现场补 {len(missing)} 条假设文本到 {cache_path}")
    elif missing:
        print(f"[注意] {len(missing)} 条问题没有缓存假设文本，将退化成「假设文本=原问题」："
              f"{[q[:14] for q in missing]}（加 --fill-hyde 现场补齐，会真实调用大模型）")
    retrieval._generate_hyde = lambda q: hyde.get(q, q)
    degraded = [q for q in search_queries if hyde.get(q, q).strip() == q.strip()]
    if degraded:
        print(f"[注意] 这些问题的假设文本等于原问题（融合退化）：{[q[:14] for q in degraded]}")
    print(f"假设文本缓存：{cache_path}（{len(hyde)} 条；"
          f"本次命中 {len(search_queries) - len(degraded)}/{len(search_queries)}）")

    # 拆问臂：只有真要跑它时才装（否则一次 API 都不该花）
    decompose = None
    decompose_cache = {}
    save_cache = None
    if args.decompose:
        from scripts.eval_decomposition import (cached_decompose, load_decompose_cache,
                                                save_decompose_cache)
        import src.decompose as decompose_mod

        if decompose_mod.client is None:
            print("错误：--decompose 要调拆问大模型，但没有配置 API_KEY。"
                  "要么补 key，要么去掉 --decompose 跑整句口径。")
            return 2
        decompose_cache = load_decompose_cache(args.decompose_cache)
        decompose = cached_decompose(decompose_mod.decompose_question, decompose_cache)
        save_cache = save_decompose_cache
        print(f"检索口径：**每问分路**（线上拆问；缓存 {args.decompose_cache}）")
    else:
        print("检索口径：**整句一次检索**（改造前口径；加 --decompose 走每问分路）")

    arm_label = "decompose" if args.decompose else "whole"
    rows = []
    for query, search_query in zip(queries, search_queries):
        question = query["question"]
        history = query.get("history") or []
        hits = pick_hits(
            search_query,
            whole=lambda q: retrieval.hyde_retrieve_hits(
                q, collection, full_text, use_vector=True, top_k=TOP_K, files=files),
            decomposed=(lambda q, _history=history: llm._retrieve_hits(  # noqa: SLF001
                q, collection, full_text, True, TOP_K, files, decompose, history=_history))
            if decompose is not None else None,
            decompose=decompose)
        if not hits:
            rows.append({"id": query["id"], "kind": query["kind"], "arm": arm_label,
                         "class": query.get("negative_class"), "returned": 0,
                         "abstained": None, "answer": "", "seconds": None,
                         "shape": query.get("shape"),
                         "expect_refused": query.get("expect_refused", False),
                         "search_query": search_query, "history_turns": len(history)})
            continue
        context = build_context(hits)
        started = time.perf_counter()
        first_at = None
        parts = []
        try:
            for token in llm.stream_answer(question, context, history):
                if first_at is None:
                    first_at = time.perf_counter() - started
                parts.append(token)
        except Exception as e:                      # noqa: BLE001
            print(f"[{query['id']}] 生成失败：{str(e)[:80]}")
            continue
        answer = "".join(parts)
        row = {"id": query["id"], "kind": query["kind"], "arm": arm_label,
               "class": query.get("negative_class"), "returned": len(hits),
               "abstained": looks_like_abstention(answer), "answer": answer,
               "seconds": time.perf_counter() - started,
               "ttft": first_at, "context_chars": len(context),
               "shape": query.get("shape"),
               "expect_refused": query.get("expect_refused", False),
               "search_query": search_query, "history_turns": len(history)}
        if query["kind"] == "partial":
            row["bucket"] = classify_partial(
                answer, query["answer_terms"], query["absent_terms"])
            part_rows = part_coverage(answer, query.get("parts"))
            row["parts"] = part_rows
            row["coverage"] = {
                "covered": sum(1 for part in part_rows
                               if part["answerable"] and part["ok"]),
                "total": sum(1 for part in part_rows if part["answerable"]),
            }
            row["post_gap"] = post_gap_coverage(answer, query.get("parts"))
            row["gaps"] = gap_coverage(answer, query.get("parts"))
        rows.append(row)

    def group(kind, cls=None):
        return [r for r in rows if r["kind"] == kind and r["returned"]
                and (cls is None or r["class"] == cls)]

    pos = group("positive")
    adj = group("negative", "adjacent")
    far = group("negative", "far")
    partial = group("partial")
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

    if partial:
        buckets = {}
        for row in partial:
            buckets.setdefault(row["bucket"], []).append(row["id"])
        print()
        print(f"=== 部分可答 / 复合问题 n={len(partial)} · 分桶 ===")
        for bucket in BUCKETS:
            print(f"  {bucket:26s} = {len(buckets.get(bucket, []))}"
                  f"/{len(partial)} {buckets.get(bucket, [])}")
        print_partial_summary(partial)

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
    print(f"明细已写入 {args.json}（检索口径：{arm_label}）")
    if save_cache is not None and decompose_cache:
        save_cache(args.decompose_cache, decompose_cache)
        print(f"拆问结果已缓存：{args.decompose_cache}（{len(decompose_cache)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
