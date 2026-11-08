#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复合问题的「检索覆盖」实验（真实语料，gold 现场派生，只记数字）。

为什么要单独一个脚本
--------------------
合成评测里 `kind: "partial"` 的复合问题只覆盖 6 个文件，三个子问题经常落在同一两份
文件里。那种语料**量不到**产品真正会遇到的形状：一个问句里问了 3 件事、它们分散在
**不同的文件**里，`top_k=5` 的 context 装不装得下？

真实语料没有人工 gold，所以这里不套 recall/MRR，而是现场派生 gold：从每片正文里挑一个
**有辨识度的短语当锚点**，把锚点包成子问句，再从**不同文件**各取一个子问题拼成复合问句。
锚点就是那条子问题的「标准答案」——检索到的那块里有没有它，是可以逐字核验的。

量四件事
--------
1. **单问基线**：子问题**单独**检索时，目标片在不在 top-1。不在的子问题**不进**合并
   覆盖率的分母 —— 否则等于把「本来就检不到」记到「top_k=5 装不下」账上。
2. **合并覆盖@k**：复合问句检索出的 top-k 里，每个子问题的目标片在不在（k=5 与 k=10 各一遍）。
3. **k 的边际收益**：覆盖@5 → 覆盖@10 涨多少，用来回答「top_k=5 够不够」。
4. **每问分路检索的对照**（`--route`，见下面「分路对照」一节）：同一份 context 预算下，
   「整句一次检索」与「每问各查一次再拼」谁装得下更多子问题。上一轮实测覆盖@5 = 覆盖@10、
   未覆盖子问的目标片名次双峰（第 1 附近或 20 名开外），指向的就是这条对照。

为什么两个 k 都跑**完整检索路径**、而不是把 k=10 的结果截前 5 条
-------------------------------------------------------------
`hyde_retrieve_hits_with_path` 里 k 同时决定「向量召回条数」和「距离闸门的候选池」，
事后截断量与真的把 TOP_K 调到 5 跑一遍不是同一件事。只有真改 k 才叫线上口径。

覆盖判定口径（**写死在这里，改口径必须改测试**）
------------------------------------------------
一条子问题的目标块被「覆盖」，指检索到的某块与它**指向同一处证据**，二者其一：

- `same_chunk`：同一文件且 `chunk_index` 相同（同一片正文）；
- `same_section`：同一文件、`section` 非空且相同（同一页 / 同一工作表）。

同一文件但**不同节**（含任一侧 `section` 为空）**不算**：那只是「这份文件里检索到了别处」，
模型手里并没有那条事实。线上引用的粒度就是页，同一页被召回时读者能翻页核验；不同页不算。
另外**单独记一列严格口径**（`strict` = 锚点原文真的出现在某块正文里），因为 `same_section`
可能把「同一页的兄弟块」（里面没有那条事实）也算成命中，两列差值就是这种通胀的上限。

安全约定（沿用 AGENTS.md「真实语料只记数字不留文件」）
-----------------------------------------------------
- 只读 `state/*.json` 的 `files[].file_text`；`chromadb.EphemeralClient` 临时库，不写
  `chroma_db/`；**不打印正文、不打印文件名、不落语料**。
- `--json` 明细只有 id / 命中布尔 / 距离 / 路径；落盘前还有一道 leak 检查（出现锚点原文或
  文件名就中止，绝不写盘）。
- **默认不开 HyDE**（免费、可重复）；开了要配 `--hyde-cache`，否则两轮比的是大模型随机性。
- embedding 模型只在 `main()` 里加载：模块 import 阶段不碰 torch / 模型。

用法
----
    python scripts/eval_compound_retrieval.py                      # 只打印数字（含三臂对照）
    python scripts/eval_compound_retrieval.py --route off          # 只看原有口径，不跑分路对照
    python scripts/eval_compound_retrieval.py --route rrf --route-budget 5
    python scripts/eval_compound_retrieval.py --json .agents/x.json
    python scripts/eval_compound_retrieval.py --hyde --hyde-cache .agents/hyde.json
    python scripts/eval_compound_retrieval.py --state-dir /tmp/state --limit-files 5

退出码：0 正常；2 输入不合法（状态目录不存在 / 没有带文件的对话 / 一条样本都造不出来）。
"""
import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_real_corpus import conversations_from_state  # noqa: E402
from scripts.eval_retrieval import (  # noqa: E402  归一化口径只有一份，别处不要重写
    cached_generate, hyde_self_check, load_hyde_cache, normalize, save_hyde_cache)
from src.pdf_ingestion import split_files  # noqa: E402  与线上同一条切片路径

# 锚点长度区间：低于 8 字的短语在语料里太常见（钉不住一片正文），高于 14 字又会把
# 子问句撑成半句原文（问句和答案越长，测到的就越不是「检索」而是「抄写」）。
ANCHOR_MIN_LEN = 8
ANCHOR_MAX_LEN = 14
# 锚点在全语料里最多出现几次。1 太苛刻（真实课程资料里同义句、模板句很常见，会挑不出锚点）；
# 3 以上就会经常两个子问题的锚点落在同一片正文里，「多文件分散」这个前提就不成立了。
ANCHOR_MAX_CORPUS_HITS = 2
# 每片正文最多试多少个候选窗口（候选是随机顺序，试不完就放弃这一片）。
ANCHOR_TRIES = 80
# 复合问句的子问数量。2 问是「分散」的下限，4 问是「一条问句里能问清几件事」的上限。
SAMPLE_SIZES = (2, 3, 4)
# 实义片段分隔符：标点、空白、结构标记（`=== 第 N 页 ===` 里的 `=`）都算。锚点绝不跨标点，
# 否则包进问句就是半句话，读起来不像人问的。
_SEGMENT_SPLIT_RE = re.compile(r"[^\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9]+")


# ── 锚点：从一片正文里挑一个能把子问题钉死在这片上的短语 ──

def _is_cjk(char):
    return "\u3400" <= char <= "\u4dbf" or "\u4e00" <= char <= "\u9fff"


def _anchor_shape_ok(window):
    """窗口能不能当锚点：两端必须是汉字，且至少 6 个汉字。

    两端非汉字通常意味着把英文单词或数字切成了半截（`...thequi` 这种），包进「」里
    既不像话，也不利于 2-gram 关键词打分；汉字占比太低则多半是表头、编号或纯符号窗口。
    """
    if not window or not _is_cjk(window[0]) or not _is_cjk(window[-1]):
        return False
    return sum(1 for char in window if _is_cjk(char)) >= 6


def anchor_candidates(text, rng, min_len=ANCHOR_MIN_LEN, max_len=ANCHOR_MAX_LEN):
    """返回**随机顺序**的候选锚点窗口（已归一化）。

    只在「实义连续片段」内部取窗：跨标点的窗口是半句话；同时要求片段比窗口长至少 2 字，
    否则整行小标题会被当成锚点 —— 标题不是「事实」，检索命中它也不能说明那片正文被装下了。
    """
    normalized = normalize(text)
    windows = []
    for segment in _SEGMENT_SPLIT_RE.split(normalized):
        if not segment:
            continue
        for length in range(min_len, max_len + 1):
            if len(segment) - length < 2:
                continue
            for start in range(0, len(segment) - length + 1):
                window = segment[start:start + length]
                if _anchor_shape_ok(window):
                    windows.append(window)
    rng.shuffle(windows)
    return windows


def anchor_fits(anchor, file_text, corpus_texts, max_corpus=ANCHOR_MAX_CORPUS_HITS):
    """锚点够不够「钉死」：在**本文件里恰好出现一次**，且全语料里不超过 `max_corpus` 次。

    为什么要求本文件唯一：否则检索命中同一文件的另一片正文也算「命中」，整份指标就退化成
    「这份文件被召回了没有」，量不到「这一片在不在 context 里」。
    为什么限制全语料频次：锚点在别处也出现时，检出别处是**合理的**，但那不能算「这条子问题
    没被装下」—— 指标会掺进词面巧合的噪声。两个口径都在 `tests/` 里钉住。
    """
    needle = normalize(anchor)
    if not needle:
        return False
    if normalize(file_text).count(needle) != 1:
        return False
    return sum(normalize(text).count(needle) for text in corpus_texts or []) <= max_corpus


def pick_anchor(chunk_text, file_text, corpus_texts, rng,
                tries=ANCHOR_TRIES, max_corpus=ANCHOR_MAX_CORPUS_HITS):
    """给一片正文挑一个锚点（归一化短语）；挑不到返回空串。

    随机只发生在这里，而且用调用方传进来的 `random.Random(seed)` —— 同一 seed 必须挑出
    同一批锚点，否则两轮数字不可比（这个脚本存在的意义就是可重复地对开关做对比）。
    """
    for window in anchor_candidates(chunk_text, rng)[:max(1, tries)]:
        if anchor_fits(window, file_text, corpus_texts, max_corpus=max_corpus):
            return window
    return ""


def build_sub_question(anchor):
    """把锚点包成一个自然的中文子问句。

    问句里必须**逐字**含锚点：这是「子问题能通过锚点定位到那片正文」的前提，
    也是覆盖判定能逐字核验的来源。措辞刻意保持中性，避免问句本身携带额外的检索线索。
    """
    return f"材料里「{anchor}」讲的是什么"


def build_compound_question(sub_questions, style="qmark"):
    """把子问句拼成一个复合问句。

    三种拼法（`--join-style`），差别就在「边界显不显眼」：
    - `qmark`（默认）：用「？」拼 —— 与线上用户写法一致，但分隔符本身就是答案，
      所以它对这个拆问实验是**简单模式**；
    - `comma`：用「，」拼 —— 中文里最常见的连写方式，问号只剩最后一个；
    - `runon`：用「还有」拼 —— 没有任何标点提示，模型必须自己找边界。
    """
    pieces = [str(question).strip().rstrip("？?") for question in sub_questions or []]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return ""
    if style == "comma":
        return "，".join(pieces) + "？"
    if style == "runon":
        return "还有".join(pieces) + "？"
    return "？".join(pieces) + "？"


JOIN_STYLES = ("qmark", "comma", "runon")


# ── 探针与复合样本 ──

def build_probes(files, rng, chunks_per_file=1, splitter=None, picker=None):
    """从一组文件里造探针：每个文件最多 `chunks_per_file` 片，每片挑一个锚点。

    返回 `(probes, missing)`，`missing` 是「挑不出锚点的片数」（如实计数，不静默）。
    切片走 `src.pdf_ingestion.split_files`，与线上是同一条路径 —— 覆盖判定比的是
    `(file_name, chunk_index)`，切片口径不一致就全错。
    """
    splitter = splitter or split_files
    picker = picker or pick_anchor
    corpus_texts = [text for _name, text in files or []]
    probes, missing = [], 0
    for file_name, file_text in files or []:
        pairs = splitter([(file_name, file_text)])
        if not pairs:
            continue
        take = min(max(1, chunks_per_file), len(pairs))
        for index in sorted(rng.sample(range(len(pairs)), take)):
            chunk, meta = pairs[index]
            anchor = picker(chunk, file_text, corpus_texts, rng)
            if not anchor:
                missing += 1
                continue
            probes.append({
                "file": (meta or {}).get("file_name") or file_name,
                "anchor": anchor,
                "sub_question": build_sub_question(anchor),
                "target": meta,
                "chunk": chunk,
            })
    return probes, missing


def sample_files(sample):
    """一条样本里各子问题目标块所属的文件（按出现顺序，可能重复）。"""
    return [((part.get("target") or {}).get("file_name")) for part in sample.get("parts") or []]


def validate_sample(sample):
    """复合样本是否有效：每个子问题必须来自**不同文件**。返回 `(ok, 原因)`。

    判的是**目标块的文件**（覆盖判定真正用的那个字段），不是构造时的标签 ——
    两者不一致时按目标块算，宁可把样本判废也不能让「同一文件里的两个子问题」
    混进分母：那样测到的还是「单文件装不装得下」，正是这个实验要补的那块的隔壁。
    """
    names = [name for name in sample_files(sample) if name]
    if len(names) != len(sample.get("parts") or []):
        return False, "目标块缺少文件信息"
    if len(set(names)) != len(names):
        return False, "子问题挤在同一文件"
    return True, ""


def make_sample(probes, sample_id="s", join_style="qmark"):
    """把若干探针拼成一条复合样本（子问题必须来自不同文件是**调用方**的责任，这里只搬运）。"""
    parts = []
    for index, probe in enumerate(probes or []):
        parts.append({
            "id": f"{sample_id}/p{index}",
            "anchor": probe.get("anchor") or "",
            "sub_question": probe.get("sub_question") or "",
            "target": probe.get("target"),
            "baseline_top1": bool(probe.get("baseline_top1")),
            "baseline_kind": probe.get("baseline_kind"),
            "baseline_rank": probe.get("baseline_rank"),
            "file_ix": probe.get("file_ix"),
        })
    return {
        "id": sample_id,
        "n_parts": len(parts),
        "question": build_compound_question([part["sub_question"] for part in parts],
                                            style=join_style),
        "parts": parts,
    }


def build_samples(probes, size, prefix="s", join_style="qmark"):
    """把探针组合成「size 问」的复合样本；返回 `(samples, skipped)`。

    **每个文件只出一个子问题**：按文件名轮转取探针（第 1 轮每个文件取第 1 片、第 2 轮
    取第 2 片……），所以一条样本里的子问题天然分散在不同文件。轮转顺序按文件名排序，
    随机性只保留在「挑哪几片 / 挑哪个锚点」那一步（见 `pick_anchor`），这里不额外抛骰子。
    `skipped` 是校验判废的条数（`validate_sample`），必须是 0；不为 0 说明构造有 bug，
    所以单独计数报出来而不是静默丢。
    """
    by_file = {}
    for probe in probes or []:
        by_file.setdefault(probe.get("file"), []).append(probe)
    order = sorted(by_file)
    cursor = {name: 0 for name in order}
    samples, skipped = [], 0
    while True:
        picked = []
        for name in order:
            if len(picked) >= size:
                break
            index = cursor[name]
            if index < len(by_file[name]):
                picked.append(by_file[name][index])
                cursor[name] = index + 1
        if len(picked) < size:
            break
        sample = make_sample(picked, sample_id=f"{prefix}/n{size}/s{len(samples)}",
                             join_style=join_style)
        ok, _reason = validate_sample(sample)
        if not ok:
            skipped += 1
            continue
        samples.append(sample)
    return samples, skipped


# ── 覆盖判定（口径见模块 docstring，改它必须改测试）──

def hit_match(hit_meta, target):
    """一个检索块与目标块「指向同一处证据」的判定：`"same_chunk"` / `"same_section"` / None。"""
    if not isinstance(hit_meta, dict) or not isinstance(target, dict):
        return None
    if hit_meta.get("file_name") != target.get("file_name"):
        return None
    if hit_meta.get("chunk_index") == target.get("chunk_index"):
        return "same_chunk"
    hit_section, target_section = hit_meta.get("section"), target.get("section")
    if hit_section and target_section and hit_section == target_section:
        return "same_section"
    return None


def covered(hits, target):
    """口径 A：目标块被覆盖（同一片，或同文件同节）。"""
    return any(hit_match(meta, target) for _text, meta in hits or [])


def anchor_visible(hits, anchor):
    """口径 B（严格）：锚点原文真的出现在某块正文里 —— 事实确实喂到模型嘴边了。

    `same_section` 会把「同一页的兄弟块」也算命中，那些块里没有这条事实；这一列是它的
    上界对照，两列差值就是「页级引用对了、但答案不在 context 里」的那部分。
    """
    needle = normalize(anchor)
    if not needle:
        return False
    return any(needle in normalize(text) for text, _meta in hits or [])


def baseline_hit(hits, target, anchor):
    """单问基线：这条子问题**单独**检索时，top-1 有没有真的定位到目标片。

    用「同一片」或「锚点可见」判，**不用**同节口径：同页的兄弟块里没有那条事实，
    把它算成基线命中会让「本来就检不到」的子问题溜进合并覆盖率的分母。
    """
    if not hits:
        return False
    text, meta = hits[0]
    if hit_match(meta, target) == "same_chunk":
        return True
    return anchor_visible([(text, meta)], anchor)


def baseline_kind(hits, target, anchor):
    """单问基线命中的**性质**：`"same_chunk"` / `"anchor_only"` / None。

    为什么要分开：分母（「可用子问」）本身是**同一把检索器**定义的，而 `baseline_hit`
    还允许「锚点可见」这一条更宽的路 —— 锚点允许在全语料里出现 ≤2 次，所以一条子问
    可能只是把**另一份文件**里同一句模板话捞到了 top-1，就算「基线命中」。
    这种子问会让分路臂的数字被抬高（它重新查一次就能再捞到同一块）。
    所以除了原来的分母，还要能只数「top-1 真是目标片」的严格分母。
    """
    if not hits:
        return None
    text, meta = hits[0]
    if hit_match(meta, target) == "same_chunk":
        return "same_chunk"
    if anchor_visible([(text, meta)], anchor):
        return "anchor_only"
    return None


def measure_baseline(probes, retrieve):
    """给每个探针量一次单问基线（就地写回 `baseline_top1` 与 `baseline_kind`）。

    检索抛错的探针按「基线未命中」处理（保守：把它排除出合并覆盖率的分母），
    单独一条失败不该带走整轮统计。
    """
    for probe in probes or []:
        try:
            _path, hits = retrieve(probe["sub_question"], 1)
        except Exception as e:                                  # pragma: no cover - 兜底
            probe["baseline_top1"] = False
            probe["baseline_kind"] = None
            probe["baseline_error"] = str(e)[:80]
            continue
        probe["baseline_top1"] = baseline_hit(hits, probe.get("target"), probe.get("anchor"))
        probe["baseline_kind"] = baseline_kind(hits, probe.get("target"), probe.get("anchor"))
    return probes


def run_sample(sample, retrieve, top_k=5, extra_k=10):
    """跑一条复合样本：两个 k 各检索一次完整路径，逐子问题判覆盖。

    `retrieve(question, k) -> (path, hits)` 注入进来，单测就能用假检索把「覆盖怎么算」
    钉死，不必拉起 Chroma 与 embedding 模型。
    """
    record = {
        "id": sample.get("id"),
        "n_parts": sample.get("n_parts"),
        "top_k": top_k,
        "extra_k": extra_k,
        "path": None,
        "path_extra": None,
        "hits_top": None,
        "hits_extra": None,
        "best_distance": None,
        "parts": [],
    }
    judged = {}
    for slot, k in (("top", top_k), ("extra", extra_k)):
        path, hits = retrieve(sample.get("question") or "", k)
        record["path" if slot == "top" else "path_extra"] = path
        # 命中块数是解释「k 的边际收益为什么是 0」的关键：距离闸门常常先一步把
        # 候选砍到比 k 还少，那时把 TOP_K 调大根本不会有新块进来。
        record[f"hits_{slot}"] = len(hits)
        for part in sample.get("parts") or []:
            entry = judged.get(part["id"])
            if entry is None:
                target = part.get("target") or {}
                entry = {
                    "id": part["id"],
                    "eligible": bool(part.get("baseline_top1")),
                    "baseline_top1": bool(part.get("baseline_top1")),
                    "baseline_rank": part.get("baseline_rank"),
                    "file_ix": part.get("file_ix"),
                    "chunk_index": target.get("chunk_index"),
                }
                judged[part["id"]] = entry
            entry[f"covered_{slot}"] = covered(hits, part.get("target"))
            entry[f"strict_{slot}"] = anchor_visible(hits, part.get("anchor"))
    record["parts"] = list(judged.values())
    return record


# ── 每问分路检索（对照实验 A/B）──
#
# 上一轮量到的结论是「k 不是杠杆」：覆盖@5 与覆盖@10 完全相同，未覆盖子问的目标片名次
# 是双峰的（第 1 附近或 20 名开外）。名次落在 20 名开外说明**整句一起查的时候那一问
# 根本没被当回事** —— 一句复合问句被压成一个向量，几个子问题在向量空间里互相拉扯。
# 于是这里做对照：**同一份 context 预算**下，「整句一次检索」和「每问各查一次再拼」
# 谁装得下更多子问题。
#
# 三臂只在「怎么把预算分给各问」上不同，其余全部相同（同样的样本、同样 k 的召回深度、
# 同样的距离闸门/融合/兜底路径、同样的分母）：
#
# - `merged`：现状。整句一次检索，`budget` 就是 TOP_K。
# - `interleave`：每问各查 `budget` 条，然后**轮流分名额**（先每问第 1 名、再每问第 2 名…），
#   截到 `budget` 块。保证「某一问的第 2 名不许把另一问的第 1 名挤掉」。
# - `rrf`：每问各查 `budget` 条，按名次做 RRF（与线上 `retrieval._rrf_fuse` 同口径）
#   再截到 `budget` 块。让「被多问同时认可的块」往前排。
#
# 两只脚都要站稳，否则这个对照实验会骗人：
# 1. **上下文预算必须一样** —— 分路臂要是能塞 3 倍块，它赢的是 token 预算不是检索质量；
# 2. **分母必须一样** —— 都用「单问基线命中的子问题」，否则两臂衡量的是两批子问。
# 这两条都有测试钉住（tests/test_eval_compound_retrieval.py）。
#
# 本轮还有一条硬边界：**拆问用的是 gold 子问题**（样本本来就是由子问题拼出来的）。
# 所以这里量到的是「分路检索的上界」，线上还需要一个 query 分解步骤（LLM 一次调用），
# 那一层的质量与成本不在这张表里。

ROUTE_MERGED = "merged"
ROUTE_INTERLEAVE = "interleave"
ROUTE_RRF = "rrf"
ROUTE_DECOMPOSE = "decompose"
ROUTE_MODES = (ROUTE_MERGED, ROUTE_INTERLEAVE, ROUTE_RRF, ROUTE_DECOMPOSE)
ROUTE_LABELS = {
    ROUTE_MERGED: "现状（整句一次检索）",
    ROUTE_INTERLEAVE: "分路 + 轮流分名额（gold 拆问）",
    ROUTE_RRF: "分路 + 名次融合（gold 拆问）",
    ROUTE_DECOMPOSE: "分路 + 轮流分名额（**模型拆问**）",
}
# 一路走向量、另一路走关键词兜底时的汇总标签：两条路的分数含义完全不同，必须标出来。
PATH_MIXED = "mixed"


def hit_key(hit):
    """一个检索块的身份：优先 (文件名, 片号)；没有 meta 的老索引退回正文本身。

    去重必须按**块身份**而不是正文：同一片被两个子问题都召回时，正文一模一样，
    但「两片不同文件里恰好内容相同」也是可能的 —— 那种情况是两块，不能只留一块。
    """
    text, meta = hit
    if isinstance(meta, dict):
        name, index = meta.get("file_name"), meta.get("chunk_index")
        if name is not None and index is not None:
            return (name, index)
    return text


def interleave_hits(part_hits, budget):
    """每问轮流取名额，直到装满 `budget` 块。

    先给每个子问题它的第 1 名，再轮流给第 2 名……这样「某一问的第 2 名」永远不会把
    「另一问的第 1 名」挤掉。某问已经没料时名额自动留给还在出货的那些问，不空转。
    同一片被两问都召回时只占一个名额（按块身份去重）。
    """
    budget = max(0, int(budget or 0))
    ordered, seen = [], set()
    depth = max((len(hits or []) for hits in part_hits), default=0)
    for rank in range(depth):
        for hits in part_hits:
            if rank >= len(hits or []):
                continue
            key = hit_key(hits[rank])
            if key in seen:
                continue
            seen.add(key)
            ordered.append(hits[rank])
            if len(ordered) >= budget:
                return ordered
    return ordered


def rrf_merge_hits(part_hits, budget, k=None):
    """每问分路召回 → 按名次 RRF 融合 → 截到 `budget` 块。

    与线上 `retrieval._rrf_fuse` **同一套算法**（第 rank 位得 1/(k+rank)，跨路相加，
    同分按正文排），只是这里要保留块的 meta 以便判覆盖。同口径这件事由
    `test_rrf_agrees_with_the_online_fusion_rule` 钉住 —— 两边一旦漂开，
    这个实验比的就是两个不同的检索器。
    """
    if k is None:
        from src.config import RRF_K            # 惰性导入：模块加载阶段不碰配置/模型
        k = RRF_K
    budget = max(0, int(budget or 0))
    scores, hits_by_key, order = {}, {}, {}
    for ranked in part_hits:
        for rank, hit in enumerate(ranked or [], 1):
            key = hit_key(hit)
            if key not in hits_by_key:
                hits_by_key[key] = hit
                order[key] = len(order)
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)
    ranked_keys = sorted(hits_by_key,
                         key=lambda key: (-scores[key], hits_by_key[key][0], order[key]))
    return [hits_by_key[key] for key in ranked_keys[:budget]]


def _same_file_anchor_visible(hits, target, anchor):
    """锚点原文是不是出现在**同一文件**的某块里（哪怕不是目标片）。

    切片有 `CHUNK_OVERLAP` 重叠，一条事实经常同时落在相邻两片里：身份口径（同片 / 同节）
    会把「邻居片进了 context」判成没覆盖，可模型手里其实有那句话。这一列把这种
    「口径盲区」和「真的没进 context」分开 —— 两者在身份口径里长得一模一样。
    """
    name = (target or {}).get("file_name")
    needle = normalize(anchor)
    if not name or not needle:
        return False
    return any(isinstance(meta, dict) and meta.get("file_name") == name
               and needle in normalize(text) for text, meta in hits or [])


def _target_rank_in(hits, target):
    """目标片在这份 hits 里的名次（1 起），没有就 None。"""
    for rank, (_text, meta) in enumerate(hits or [], 1):
        if hit_match(meta, target):
            return rank
    return None


def _route_path_label(paths):
    """分路后的汇总路径标签：全向量 → vector；全兜底 → keyword；混着走 → mixed。"""
    labels = {path for path in paths or [] if path}
    if not labels:
        return PATH_MIXED
    if len(labels) == 1:
        return next(iter(labels))
    return PATH_MIXED


def route_retrieve(parts, question, retrieve, budget, mode=ROUTE_MERGED, decompose=None):
    """跑一个实验臂，返回 `(path, hits, diag)`。

    `merged` = 现状：整句一次检索。`interleave` / `rrf` = 每问分路（子问题用 **gold**）。
    `decompose` = 每问分路，但子问题由**模型现场拆**（线上唯一可行的形态）：
    它跟 `interleave` 的差就是「拆问那一步的损失」，两者用同一套名额分配口径。
    """
    if mode not in ROUTE_MODES:
        raise ValueError(f"未知的分路口径：{mode}")
    parts = list(parts or [])
    if mode == ROUTE_DECOMPOSE:
        if decompose is None:
            raise ValueError("decompose 臂必须注入一个拆问函数")
        generated = [str(text).strip() for text in (decompose(question or "") or [])]
        generated = [text for text in generated if text] or [question or ""]
        part_hits, paths = [], []
        for text in generated:
            part_path, hits = retrieve(text, budget)
            part_hits.append(list(hits or []))
            paths.append(part_path)
        merged = interleave_hits(part_hits, budget)
        # 拆漏：某个 gold 子问的目标片**任何一条**拆出来的问句都没召回到 → 拆分那一步就丢了它。
        # 这是机械可核的（比的是块身份），不靠「拆出来的问句像不像 gold」这种词面尺子。
        recalled = [any(covered(hits, part.get("target")) for hits in part_hits)
                    for part in parts]
        return _route_path_label(paths), merged, {
            "mode": mode,
            "retrievals": len(generated),
            "per_part": paths,
            "generated": len(generated),
            "fallback": len(generated) <= 1,
            "gold_recalled": recalled,
            "part_ranks": [None] * len(parts),
            "own_hits": [None] * len(parts),
            "merged": False,
        }
    if mode == ROUTE_MERGED or not parts:
        path, hits = retrieve(question or "", budget)
        hits = list(hits or [])
        return path, hits, {"mode": mode, "retrievals": 1, "per_part": [path],
                            "part_ranks": [None] * len(parts),
                            "own_hits": [None] * len(parts), "merged": bool(parts)}
    part_hits, paths = [], []
    for part in parts:
        part_path, hits = retrieve(part.get("sub_question") or "", budget)
        part_hits.append(list(hits or []))
        paths.append(part_path)
    if mode == ROUTE_INTERLEAVE:
        merged = interleave_hits(part_hits, budget)
    else:
        merged = rrf_merge_hits(part_hits, budget)
    return _route_path_label(paths), merged, {
        "mode": mode,
        "retrievals": len(parts),
        "per_part": paths,
        "part_ranks": [_target_rank_in(hits, part.get("target"))
                       for part, hits in zip(parts, part_hits)],
        "own_hits": [len(hits) for hits in part_hits],
        "merged": False,
    }


def run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_MERGED, decompose=None):
    """跑一条复合样本的一个实验臂，返回与 `run_sample` 同形的 record。

    刻意与 `run_sample` 用同一套字段名（`parts[].covered` 对应 `covered_top`）、
    同一套分母（`baseline_top1`），这样两边的数字可以直接放在一张表里比；
    `test_run_sample_arm_merged_reproduces_the_existing_measurement` 钉住
    「merged 臂逐位复现老口径」这件事。
    """
    record = {
        "id": sample.get("id"),
        "arm": mode,
        "n_parts": sample.get("n_parts"),
        "budget": budget,
        "path": None,
        "per_part_paths": [],
        "hits": None,
        "retrievals": 0,
        "generated": None,
        "decompose_fallback": None,
        "parts": [],
    }
    path, hits, diag = route_retrieve(sample.get("parts") or [], sample.get("question") or "",
                                      retrieve, budget, mode, decompose=decompose)
    record["path"] = path
    record["per_part_paths"] = list(diag["per_part"])
    record["hits"] = len(hits)
    record["retrievals"] = diag["retrievals"]
    record["generated"] = diag.get("generated")
    record["decompose_fallback"] = diag.get("fallback")
    for index, part in enumerate(sample.get("parts") or []):
        record["parts"].append({
            "id": part.get("id"),
            "eligible": bool(part.get("baseline_top1")),
            "baseline_top1": bool(part.get("baseline_top1")),
            "baseline_kind": part.get("baseline_kind"),
            "file_ix": part.get("file_ix"),
            "covered": covered(hits, part.get("target")),
            "strict": anchor_visible(hits, part.get("anchor")),
            # 「身份口径没覆盖、但事实在同文件的邻居片里」—— 切片重叠造成的口径盲区
            "same_file_anchor": _same_file_anchor_visible(hits, part.get("target"),
                                                         part.get("anchor")),
            # 拆问臂专用：这条 gold 子问有没有被**某条拆出来的问句**捞到过（拆漏诊断）
            "recalled_by_split": (diag["gold_recalled"][index]
                                  if index < len(diag.get("gold_recalled") or []) else None),
            # 这一问**自己**那份召回里目标片排第几（分路臂才有；1 就是第一名）
            "part_rank": diag["part_ranks"][index] if index < len(diag["part_ranks"]) else None,
            "own_hits": diag["own_hits"][index] if index < len(diag["own_hits"]) else None,
        })
    return record


# ── 汇总 ──

def _rate(hit, total):
    return None if not total else hit / total


def _group_stats(records, n_parts, path, top_k, extra_k):
    """一组样本的覆盖统计。

    分母只有**基线命中的子问题**（`eligible`）：基线没检到的子问题不是合并的错。
    样本级另有「一条样本的所有可用子问题都被覆盖」——那才是产品问的那句「装得下吗」。
    """
    parts = [(record, part) for record in records for part in (record.get("parts") or [])]
    eligible = [(record, part) for record, part in parts if part.get("eligible")]
    total_eligible = len(eligible)

    def _count(field, rows=None):
        return sum(1 for _record, part in (eligible if rows is None else rows) if part.get(field))

    covered_top = _count("covered_top")
    covered_extra = _count("covered_extra")
    strict_top = _count("strict_top")
    strict_extra = _count("strict_extra")

    def _all_covered(field):
        good = 0
        for record in records:
            usable = [part for part in record.get("parts") or [] if part.get("eligible")]
            if usable and all(part.get(field) for part in usable):
                good += 1
        return good

    all_top, all_extra = _all_covered("covered_top"), _all_covered("covered_extra")
    # 样本级全覆盖的分母只能是「至少有一个可用子问」的样本：一个可用子问都没有的样本
    # 按定义不可能算出「全覆盖」，把它们留在分母里等于让它们天然记成失败
    # （本语料 35 条里有 10 条就是这种）。两臂同口径，不影响对比。
    samples_with_eligible = sum(
        1 for record in records
        if any(part.get("eligible") for part in record.get("parts") or []))
    hits_top = [record.get("hits_top") for record in records if record.get("hits_top") is not None]
    hits_extra = [record.get("hits_extra") for record in records if record.get("hits_extra") is not None]
    # 「被排除的子问」多半不是检索坏，而是锚点本身没把这个子问钉到 top-1：
    # 目标片在原始向量查询（不经闸门）里排第几，能看出差的是「排序」还是「根本没召回」。
    excluded_ranks = [part.get("baseline_rank") for _record, part in parts if not part.get("eligible")]
    excluded_ranked = [rank for rank in excluded_ranks if rank]
    # 未覆盖的子问：目标片在**原始**排名里落在 k+1..extra_k 的才叫「调 k 能救」；
    # 落在更后面或压根没召回，调 k 也救不了 —— 这两类必须在同一张表里分开，
    # 否则「k 的边际收益为 0」会被误读成「k 已经够大」。
    uncovered_ranks = [part.get("raw_rank") for _record, part in eligible
                       if not part.get("covered_top")]
    rescuable = sum(1 for rank in uncovered_ranks if rank and top_k < rank <= extra_k)
    stats = {
        "n_parts": n_parts,
        "path": path,
        "samples": len(records),
        "parts": len(parts),
        "parts_eligible": total_eligible,
        "parts_excluded": len(parts) - total_eligible,
        "baseline_hit": _count("baseline_top1", parts),
        "baseline_rate": _rate(_count("baseline_top1", parts), len(parts)),
        "excluded_ranked": len(excluded_ranked),
        "excluded_rank_median": statistics.median(excluded_ranked) if excluded_ranked else None,
        "mean_hits_top": statistics.mean(hits_top) if hits_top else None,
        "mean_hits_extra": statistics.mean(hits_extra) if hits_extra else None,
        "uncovered_rescuable": rescuable,
        "uncovered_out_of_reach": len(uncovered_ranks) - rescuable,
        f"covered_at_{top_k}": covered_top,
        f"coverage_at_{top_k}": _rate(covered_top, total_eligible),
        f"covered_at_{extra_k}": covered_extra,
        f"coverage_at_{extra_k}": _rate(covered_extra, total_eligible),
        f"strict_covered_at_{top_k}": strict_top,
        f"strict_coverage_at_{top_k}": _rate(strict_top, total_eligible),
        f"strict_covered_at_{extra_k}": strict_extra,
        f"strict_coverage_at_{extra_k}": _rate(strict_extra, total_eligible),
        "samples_with_eligible": samples_with_eligible,
        f"samples_all_covered_at_{top_k}": all_top,
        f"samples_all_covered_rate_at_{top_k}": _rate(all_top, samples_with_eligible),
        f"samples_all_covered_at_{extra_k}": all_extra,
        f"samples_all_covered_rate_at_{extra_k}": _rate(all_extra, samples_with_eligible),

        "samples_path_changed_at_extra_k": sum(
            1 for record in records if record.get("path") != record.get("path_extra")),
    }
    if total_eligible:
        stats["gain"] = (covered_extra - covered_top) / total_eligible
        stats["strict_gain"] = (strict_extra - strict_top) / total_eligible
    else:
        stats["gain"] = None
        stats["strict_gain"] = None
    return stats


def summarize(results, top_k=5, extra_k=10):
    """按「子问数量 × 检索路径」汇总；空输入返回空结构而不是崩。

    为什么要按 `path` 拆开：关键词兜底的分母是 2-gram 词面分、不是向量距离，而且兜底
    只回 3 块（`retrieve_keyword` 的硬上限）—— 把它和向量路径混在一起，「top_k=5 够不够」
    就会被兜底的低天花板污染。分组的 `path` 取**主口径那一次（k=top_k）**的路径；
    `samples_path_changed_at_extra_k` 记下 k 变大后路径翻转的样本数，方便判断这一列还准不准。
    """
    records = list(results or [])
    groups = []
    sizes = sorted({record.get("n_parts") for record in records if record.get("n_parts")})
    for size in sizes:
        subset = [record for record in records if record.get("n_parts") == size]
        groups.append(_group_stats(subset, size, "all", top_k, extra_k))
        paths = sorted({record.get("path") for record in subset})
        for path in paths:
            rows = [record for record in subset if record.get("path") == path]
            groups.append(_group_stats(rows, size, path, top_k, extra_k))
    return {"groups": groups,
            "overall": _group_stats(records, None, "all", top_k, extra_k)}


def summarize_routes(records, budget=5):
    """按「实验臂 × 子问数量」汇总，另给每个臂一行「全部」。

    `n_parts is None` 的那一组是该臂的总体（跨 2/3/4 问合并）。分母与现状口径完全一致：
    **单问基线命中的子问题**（`eligible`），所以两臂放在一张表里比是同一个分母。
    成本也一起报（每样本的 context 块数、检索次数）—— 分路臂每问都要查一次，
    赢了覆盖率但没报成本，等于把一个要付钱的方案说成免费的。
    """
    records = list(records or [])
    groups = []
    sizes = sorted({record.get("n_parts") for record in records if record.get("n_parts")})
    for arm in ROUTE_MODES:
        rows = [record for record in records if record.get("arm") == arm]
        if not rows:
            continue
        for size in sizes + [None]:
            subset = [record for record in rows
                      if size is None or record.get("n_parts") == size]
            if not subset:
                continue
            parts = [part for record in subset for part in (record.get("parts") or [])]
            eligible = [part for part in parts if part.get("eligible")]
            covered = sum(1 for part in eligible if part.get("covered"))
            strict = sum(1 for part in eligible if part.get("strict"))
            # 身份口径没覆盖的格子再拆一次：事实在**同文件邻居片**里（切片重叠 → 口径盲区），
            # 还是压根没进 context。合并成一个数会把这两种完全不同的情况混为一谈。
            miss_sibling = sum(1 for part in eligible
                               if not part.get("covered") and part.get("same_file_anchor"))
            miss_elsewhere = sum(1 for part in eligible
                                 if not part.get("covered") and not part.get("same_file_anchor"))
            # 「锚点可见」这列会**跨文件**通胀：锚点允许在全语料里出现 ≤2 次，分路检索每次都
            # 拿带锚点的问句去查，很容易把另一份文件里的同一句模板话捞进来。它不算覆盖。
            strict_cross_file = sum(1 for part in eligible
                                    if part.get("strict") and not part.get("covered")
                                    and not part.get("same_file_anchor"))
            # 分母本身也是**同一把检索器**定义的：`baseline_hit` 允许「锚点可见」
            # 这条更宽的路（锚点可能来自别的文件），那样分路臂会白捡。所以另算一列
            # 只含「top-1 真是目标片」的严格分母，用来验证结论不是分母吹出来的。
            strict_denominator = [part for part in eligible
                                  if part.get("baseline_kind") == "same_chunk"]
            strict_den_covered = sum(1 for part in strict_denominator if part.get("covered"))
            # 拆问臂专用：**单问检索本来找得到**的 gold 子问里，有多少条拆完之后「任何一条
            # 拆出来的问句都捞不到」= 拆问那一步丢的。分母必须和覆盖口径一样（eligible），
            # 否则会把「本来就检不到」记到拆问头上 —— 这正是上一轮踩过的坑。
            split_rows = [part for part in eligible if part.get("recalled_by_split") is not None]
            split_recalled = sum(1 for part in split_rows if part.get("recalled_by_split"))
            fallbacks = sum(1 for record in subset if record.get("decompose_fallback"))
            generated = [record.get("generated") for record in subset
                         if record.get("generated") is not None]
            # 分路臂的漏格再拆：那一问自己就没召回到目标片，还是召回到了但名额被截掉。
            part_ranks = [part.get("part_rank") for part in eligible]
            rank_seen = [rank for rank in part_ranks if rank]
            all_covered = 0
            samples_with_eligible = 0
            for record in subset:
                usable = [part for part in record.get("parts") or [] if part.get("eligible")]
                if not usable:
                    continue                    # 没有可用子问的样本不进样本级分母
                samples_with_eligible += 1
                if all(part.get("covered") for part in usable):
                    all_covered += 1
            hits = [record.get("hits") for record in subset
                    if record.get("hits") is not None]
            retrievals = [record.get("retrievals") for record in subset
                          if record.get("retrievals") is not None]
            path_mix = {}
            for record in subset:
                path_mix[record.get("path")] = path_mix.get(record.get("path"), 0) + 1
            groups.append({
                "arm": arm,
                "n_parts": size,
                "samples": len(subset),
                "parts": len(parts),
                "parts_eligible": len(eligible),
                "parts_strict_denominator": len(strict_denominator),
                "covered_strict_denominator": strict_den_covered,
                "coverage_strict_denominator": _rate(strict_den_covered,
                                                     len(strict_denominator)),
                "parts_split_recalled": split_recalled,
                "parts_split_denominator": len(split_rows),
                "split_recall": _rate(split_recalled, len(split_rows)),
                "decompose_fallbacks": fallbacks,
                "mean_generated": statistics.mean(generated) if generated else None,
                "covered": covered,
                "coverage": _rate(covered, len(eligible)),
                "strict_covered": strict,
                "strict_coverage": _rate(strict, len(eligible)),
                "samples_all_covered": all_covered,
                "samples_with_eligible": samples_with_eligible,
                "samples_all_covered_rate": _rate(all_covered, samples_with_eligible),
                "miss_same_file_anchor": miss_sibling,
                "miss_elsewhere": miss_elsewhere,
                "strict_cross_file": strict_cross_file,
                "part_rank_seen": len(rank_seen),
                "part_rank_median": statistics.median(rank_seen) if rank_seen else None,
                "mean_hits": statistics.mean(hits) if hits else None,
                "mean_retrievals": statistics.mean(retrievals) if retrievals else None,
                "path_mix": path_mix,
            })
    return groups


# ── 落盘安全网 ──

def find_leaks(text, needles):
    """在要落盘的文本里找语料泄漏（锚点原文 / 文件名）。返回命中的**个数**，不返回原文。

    真实语料只许留数字：明细里一旦出现正文片段或文件名，就不是「指标」而是「语料副本」了。
    返回计数而不是内容，是为了连报错信息里都不带正文。
    """
    return sum(1 for needle in needles or [] if needle and needle in (text or ""))


# ── 打印 ──

def _pct(value):
    return "无数据" if value is None else f"{value * 100:.1f}%"


def _pp(value):
    return "无数据" if value is None else f"{value * 100:+.1f}pp"


def _print_group(stats, top_k, extra_k, indent="  "):
    label = f"{stats['n_parts']} 问" if stats["n_parts"] else "全部"
    if stats["path"] != "all":
        label += f" / {stats['path']}"
    print(f"{indent}{label}：样本 {stats['samples']}；子问 {stats['parts']}"
          f"（可用 {stats['parts_eligible']}，排除 {stats['parts_excluded']}）")
    print(f"{indent}  覆盖@{top_k} = {stats[f'covered_at_{top_k}']}/{stats['parts_eligible']}"
          f"（{_pct(stats[f'coverage_at_{top_k}'])}）"
          f"  覆盖@{extra_k} = {stats[f'covered_at_{extra_k}']}/{stats['parts_eligible']}"
          f"（{_pct(stats[f'coverage_at_{extra_k}'])}）"
          f"  边际收益 {_pp(stats['gain'])}")
    print(f"{indent}  严格口径（锚点真的进了正文）@{top_k} = {_pct(stats[f'strict_coverage_at_{top_k}'])}"
          f"  @{extra_k} = {_pct(stats[f'strict_coverage_at_{extra_k}'])}"
          f"  边际收益 {_pp(stats['strict_gain'])}")
    print(f"{indent}  样本级全覆盖@{top_k} = "
          f"{stats[f'samples_all_covered_at_{top_k}']}/{stats['samples_with_eligible']}"
          f"（{_pct(stats[f'samples_all_covered_rate_at_{top_k}'])}）"
          f"  @{extra_k} = {stats[f'samples_all_covered_at_{extra_k}']}/{stats['samples_with_eligible']}"
          f"（{_pct(stats[f'samples_all_covered_rate_at_{extra_k}'])}）"
          f"  路径翻转 {stats['samples_path_changed_at_extra_k']} 条"
          f"（分母只数有可用子问的样本 {stats['samples_with_eligible']}/{stats['samples']}）")
    mean_top = stats["mean_hits_top"]
    mean_extra = stats["mean_hits_extra"]
    print(f"{indent}  每问命中块数均值 @{top_k} = "
          f"{'无数据' if mean_top is None else f'{mean_top:.2f}'}"
          f"  @{extra_k} = {'无数据' if mean_extra is None else f'{mean_extra:.2f}'}"
          f"（闸门先砍到比 k 少时，调大 k 不会多出块来）")
    print(f"{indent}  没覆盖的子问 {stats['uncovered_rescuable'] + stats['uncovered_out_of_reach']} 条："
          f"目标片在原始排名 {top_k + 1}–{extra_k} 的 {stats['uncovered_rescuable']} 条（调 k 本来能救）"
          f"，更靠后或没召回的 {stats['uncovered_out_of_reach']} 条（调 k 救不回）")


def _print_routes(groups, budget):
    """对照表：每臂 × 子问数量的覆盖率、样本级全覆盖、成本。"""
    by_key = {(group["arm"], group["n_parts"]): group for group in groups}
    arms = [arm for arm in ROUTE_MODES if any(group["arm"] == arm for group in groups)]
    sizes = sorted({group["n_parts"] for group in groups if group["n_parts"] is not None})
    merged_overall = by_key.get((ROUTE_MERGED, None))
    print(f"=== 每问分路检索对照（同一 context 预算 {budget} 块；分母 = 单问基线命中的子问）===")
    for arm in arms:
        overall = by_key.get((arm, None))
        if not overall:
            continue
        detail = "｜".join(
            f"{size} 问 {_pct(by_key[(arm, size)]['coverage'])}"
            for size in sizes if (arm, size) in by_key)
        delta = ""
        if merged_overall and arm != ROUTE_MERGED and overall["coverage"] is not None \
                and merged_overall["coverage"] is not None:
            delta = f"｜相对现状 {_pp(overall['coverage'] - merged_overall['coverage'])}"
        print(f"  {ROUTE_LABELS.get(arm, arm)}：全部 {_pct(overall['coverage'])}"
              f"（{overall['covered']}/{overall['parts_eligible']}）"
              f"{'｜' + detail if detail else ''}{delta}")
    print()
    for arm in arms:
        overall = by_key.get((arm, None))
        if not overall:
            continue
        mean_hits = overall["mean_hits"]
        mean_retrievals = overall["mean_retrievals"]
        print(f"  {ROUTE_LABELS.get(arm, arm)}：样本级全覆盖 "
              f"{overall['samples_all_covered']}/{overall['samples_with_eligible']}"
              f"（{_pct(overall['samples_all_covered_rate'])}）"
              f"；每样本 context "
              f"{'无数据' if mean_hits is None else f'{mean_hits:.2f}'} 块 / 检索 "
              f"{'无数据' if mean_retrievals is None else f'{mean_retrievals:.2f}'} 次"
              f"；路径 {overall['path_mix']}")
        misses = overall["miss_same_file_anchor"] + overall["miss_elsewhere"]
        print(f"    身份口径漏掉的 {misses} 处：事实在同文件**邻居片**里 "
              f"{overall['miss_same_file_anchor']} 处（切片重叠 → 口径盲区）、"
              f"真的没进 context {overall['miss_elsewhere']} 处。"
              f"「锚点可见」那列 {_pct(overall['strict_coverage'])} 里有 "
              f"{overall['strict_cross_file']} 处是**从别的文件**捞到的同一句模板话，不算覆盖")
        print(f"    稳健性：只数「单问 top-1 真是目标片」的严格分母 {overall['parts_strict_denominator']} 个时，"
              f"覆盖 {overall['covered_strict_denominator']}/{overall['parts_strict_denominator']}"
              f"（{_pct(overall['coverage_strict_denominator'])}）")
        if overall["parts_split_denominator"]:
            mean_generated = overall["mean_generated"]
            print(f"    拆问：gold 子问被「某条拆出来的问句」捞到 "
                  f"{overall['parts_split_recalled']}/{overall['parts_split_denominator']}"
                  f"（{_pct(overall['split_recall'])}，拆漏 "
                  f"{overall['parts_split_denominator'] - overall['parts_split_recalled']} 条）；"
                  f"退回整句（只拆出一条）{overall['decompose_fallbacks']}/{overall['samples']} 条；"
                  f"平均拆出 {'无数据' if mean_generated is None else f'{mean_generated:.2f}'} 问")


# ── 检索 ──

def make_retrieve(collection, files, count):
    """把线上检索入口包成 `retrieve(question, k) -> (path, hits)`。

    刻意**不**自己实现一遍阈值/融合/兜底：这里量的必须是线上真跑的那条路，
    否则「k=10 有没有帮助」量的是我抄错的版本。
    """
    import src.retrieval as retrieval

    full_text = "\n\n".join(text for _name, text in files or [])

    def retrieve(question, k):
        return retrieval.hyde_retrieve_hits_with_path(
            question, collection, full_text, use_vector=True, top_k=k, files=files)

    return retrieve


def raw_ranking(collection, query_text, count, id_to_meta, targets, limit=None):
    """原始向量查询（不经距离闸门 / 融合 / 兜底）：`(最佳距离, [每个目标片的名次])`。

    为什么值得多查这一次：`hyde_retrieve_hits_with_path` 只回 hits 不回距离，而两个问题
    都要靠它回答 ——
    - **距离**：「这条为什么走了关键词兜底」；
    - **名次**：「k 的边际收益为什么是 0」。目标片若在第 20 名，TOP_K 调到 10 也救不回来；
      只有落在 k+1..extra_k 的那部分才算「调 k 能救」。把这两种「没覆盖」分开，
      才不会把「排序根本排不上来」记成「k 不够大」。

    `limit=None` 时取全量（名次是真名次），否则只在前 `limit` 名里找（找到就够用了）。
    """
    targets = list(targets or [])
    if not count or not id_to_meta:
        return None, [None] * len(targets)
    try:
        result = collection.query(query_texts=[query_text],
                                  n_results=count if limit is None else min(limit, count))
    except Exception as e:
        print(f"    （距离/名次诊断不可用：{str(e)[:60]}）")
        return None, [None] * len(targets)
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    ranks = []
    for target in targets:
        rank = None
        for position, chunk_id in enumerate(ids, 1):
            meta = id_to_meta.get(chunk_id) or {}
            if (meta.get("file_name") == (target or {}).get("file_name")
                    and meta.get("chunk_index") == (target or {}).get("chunk_index")):
                rank = position
                break
        ranks.append(rank)
    return (float(min(distances)) if distances else None), ranks


def target_rank(collection, question, target, id_to_meta, limit=10):
    """目标片在**原始**向量查询里的名次（1 起）；不在前 `limit` 名返回 None。

    单问基线只判 top-1，但「没命中」有两种完全不同的原因：目标片排在第 3（排序问题，
    调 k 或有救）和第 40（锚点根本没把它的正文召回来）。这一列把两者分开，
    免得把「锚点半身不灵」全记到「top_k=5 装不下」头上。
    """
    if not target or not id_to_meta:
        return None
    try:
        result = collection.query(query_texts=[question], n_results=min(limit, len(id_to_meta)))
    except Exception:
        return None
    for rank, chunk_id in enumerate((result.get("ids") or [[]])[0], 1):
        meta = id_to_meta.get(chunk_id) or {}
        if (meta.get("file_name") == target.get("file_name")
                and meta.get("chunk_index") == target.get("chunk_index")):
            return rank
    return None


# ── 主体 ──

def main(argv=None):
    parser = argparse.ArgumentParser(description="复合问题检索覆盖实验（真实语料、只记数字）")
    parser.add_argument("--state-dir", default=None, help="状态目录，默认 src.state.STATE_DIR")
    parser.add_argument("--top-k", type=int, default=5, help="主口径的 k，默认 5（线上默认 TOP_K 是 10）")
    parser.add_argument("--extra-k", type=int, default=10, help="对照口径的 k，默认 10（量边际收益）")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="每个对话最多用前 N 个文件（按文件名排序），默认不限制")
    parser.add_argument("--chunks-per-file", type=int, default=2,
                        help="每个文件最多取几片当探针，默认 2")
    parser.add_argument("--seed", type=int, default=0, help="挑锚点的随机种子（必须可复现）")
    parser.add_argument("--route", default="all", choices=("all", "off") + ROUTE_MODES,
                        help="分路检索对照：all=三臂都跑（默认）；off=只跑原有口径；"
                             "merged/interleave/rrf=只跑某一臂")
    parser.add_argument("--route-budget", type=int, default=None,
                        help="分路对照的 context 预算，默认等于 --top-k（两臂必须同预算才可比）")
    parser.add_argument("--join-style", default="qmark", choices=JOIN_STYLES,
                        help="复合问句怎么拼：qmark（？分隔，默认）/ comma（，）/ runon（还有）")
    parser.add_argument("--decompose-cache", default=None,
                        help="拆问结果缓存 JSON；跑 decompose 臂时必须带上，否则每次比的是大模型随机性")
    parser.add_argument("--json", default=None, help="明细落盘（只有 id/命中布尔/距离/路径）")
    parser.add_argument("--hyde", action="store_true",
                        help="走真实 HyDE（花 API 额度；线上口径）；默认不开")
    parser.add_argument("--hyde-cache", default=None,
                        help="假设文本缓存 JSON；对比两轮开关时必须带上，否则比的是模型随机性")
    args = parser.parse_args(argv)

    from src import state as state_mod

    state_dir = Path(args.state_dir) if args.state_dir else Path(state_mod.STATE_DIR)
    if not state_dir.exists():
        print(f"错误：状态目录不存在：{state_dir}")
        print("   --state-dir 指错了？没有真实语料就没有可测的东西，直接中止（不假装跑出 0 个样本）。")
        return 2

    conversations = conversations_from_state(state_dir)
    if not conversations:
        print(f"错误：这个状态目录里没有任何带文件的对话：{state_dir}")
        print("   0 个可测对话 → 0 条样本，跑完只会得到一张全是「无数据」的表，直接中止。")
        return 2

    import chromadb
    from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_DISTANCE
    import src.retrieval as retrieval

    if not args.hyde:
        # 与合成评测同一个套路：HyDE 换成恒等函数，可重复、不花额度。
        # 注意此时融合会退化成单探针（`_probe_queries` 见两路文本相同就不再查第二遍），
        # 所以本脚本 no-hyde 的数字是「单探针 + 距离闸门 + 关键词兜底」口径。
        retrieval._generate_hyde = lambda question: question  # noqa: SLF001
    else:
        hyde_cache = load_hyde_cache(args.hyde_cache) if args.hyde_cache else {}
        retrieval._generate_hyde = cached_generate(retrieval._generate_hyde, hyde_cache)  # noqa: SLF001

    embedding_fn = retrieval._get_embedding_fn()

    print(f"状态目录：{state_dir}")
    print(f"检索模式：{'HyDE（真实调用大模型）' if args.hyde else 'no-hyde（query = 原问题）'}"
          f"；CHUNK_SIZE={CHUNK_SIZE}（overlap={CHUNK_OVERLAP}）；MAX_DISTANCE={MAX_DISTANCE}")
    print(f"k：主口径 {args.top_k} / 对照 {args.extra_k}；种子 {args.seed}"
          f"；每个文件最多 {args.chunks_per_file} 片；锚点 {ANCHOR_MIN_LEN}–{ANCHOR_MAX_LEN} 字、"
          f"全语料最多 {ANCHOR_MAX_CORPUS_HITS} 次；复合问句拼法 {args.join_style}")
    print()

    if args.hyde:
        probe_question = next((q for c in conversations for q in c["questions"]),
                              "材料里的这段内容讲的是什么")
        ok, text = hyde_self_check(probe_question, retrieval._generate_hyde)
        if not ok:
            print("错误：HyDE 没有真正生效（生成的文本与原问题相同）—— 这一轮不是线上口径，已中止。")
            return 2
        print(f"HyDE 自检通过：首次生成 {len(text)} 字，与问题不同")
        print()

    rng = random.Random(args.seed)
    counts = {"conversations": len(conversations), "files_used": 0, "probes": 0,
              "samples": 0, "anchors_missing": 0, "samples_same_file": 0, "query_errors": 0}
    results = []
    route_results = []
    route_budget = args.route_budget if args.route_budget else args.top_k
    route_modes = ([] if args.route == "off"
                   else list(ROUTE_MODES) if args.route == "all" else [args.route])
    # 拆问臂：只有真的要用它时才装（否则一次 API 都不该花）
    decompose = None
    decompose_cache = {}
    if ROUTE_DECOMPOSE in route_modes:
        from scripts.eval_decomposition import cached_decompose, load_decompose_cache, save_decompose_cache
        import src.decompose as decompose_mod

        if decompose_mod.client is None:
            print("错误：decompose 臂要调大模型，但没有配置 API_KEY。"
                  "要么补 key，要么用 --route merged,interleave,rrf 里的其他口径。")
            return 2
        decompose_cache = load_decompose_cache(args.decompose_cache) if args.decompose_cache else {}
        decompose = cached_decompose(decompose_mod.decompose_question, decompose_cache)
        if not args.decompose_cache:
            print("提示：没带 --decompose-cache，拆问结果不落盘 —— 两轮之间会比到大模型的随机性。")

    # 落盘前的泄漏检查要用到「锚点原文 + 文件名」，这里就地攒一份（只在内存里，
    # 一个字节都不写盘 —— 见 find_leaks 的说明）。
    corpus_needles = [name for conv in conversations for name, _text in conv["files"]]
    for conv_ix, conv in enumerate(conversations, 1):
        files = sorted(conv["files"], key=lambda item: item[0])
        if args.limit_files:
            files = files[:max(1, args.limit_files)]
        pairs = split_files(files)
        if not pairs:
            print(f"[对话#{conv_ix}] 切片后没有块，跳过")
            continue
        client = chromadb.EphemeralClient()          # 临时库：绝不碰真实 chroma_db/
        collection = client.get_or_create_collection(name=f"compound_{conv_ix}",
                                                     embedding_function=embedding_fn)
        collection.add(documents=[chunk for chunk, _meta in pairs],
                       ids=[f"c{i}" for i in range(len(pairs))],
                       metadatas=[retrieval.chroma_meta(meta) for _chunk, meta in pairs])
        count = collection.count()

        probes, missing = build_probes(files, rng, chunks_per_file=args.chunks_per_file)
        counts["anchors_missing"] += missing
        file_ix = {name: index for index, (name, _text) in enumerate(files)}
        # chunk id → meta：目标片名次诊断要用它把 Chroma 回来的 id 换回 (文件, 片号)
        id_to_meta = {f"c{index}": meta for index, (_chunk, meta) in enumerate(pairs)}
        for probe in probes:
            probe["file_ix"] = file_ix.get(((probe.get("target") or {}).get("file_name")))
        counts["probes"] += len(probes)
        counts["files_used"] += len({probe["file"] for probe in probes})
        corpus_needles += [probe["anchor"] for probe in probes]

        retrieve = make_retrieve(collection, files, count)
        measure_baseline(probes, retrieve)
        for probe in probes:
            if not probe.get("baseline_top1"):
                _dist, ranks = raw_ranking(collection, probe["sub_question"], count,
                                           id_to_meta, [probe.get("target")], limit=10)
                probe["baseline_rank"] = ranks[0]
        baseline_hits = sum(1 for probe in probes if probe.get("baseline_top1"))
        ranked = sum(1 for probe in probes if probe.get("baseline_rank"))
        print(f"[对话#{conv_ix}] 文件 {len(files)}、块 {count}、探针 {len(probes)}"
              f"（挑不到锚点 {missing}）；单问基线（top-1 命中目标片）"
              f"{baseline_hits}/{len(probes)}；未命中的 {len(probes) - baseline_hits} 条里"
              f"有 {ranked} 条目标片仍在原始前 10")

        conv_results = []
        for size in SAMPLE_SIZES:
            samples, skipped = build_samples(probes, size, prefix=f"c{conv_ix}",
                                             join_style=args.join_style)
            counts["samples_same_file"] += skipped
            for sample in samples:
                try:
                    record = run_sample(sample, retrieve,
                                        top_k=args.top_k, extra_k=args.extra_k)
                except Exception as e:               # 单条失败不该带走整轮
                    counts["query_errors"] += 1
                    print(f"    样本检索失败，跳过：{str(e)[:60]}")
                    continue
                # 诊断列：用与检索路径同一个 query 文本再做一次**原始**全量查询，
                # 一次拿到「最佳距离」和「每个目标片的名次」—— 后者回答「k 调大有没有用」。
                search_text = retrieval._generate_hyde(sample["question"])  # noqa: SLF001
                targets = [(part.get("target") or {}) for part in sample["parts"]]
                distance, ranks = raw_ranking(collection, search_text, count,
                                              id_to_meta, targets)
                record["best_distance"] = distance
                for part, rank in zip(record["parts"], ranks):
                    part["raw_rank"] = rank
                conv_results.append(record)
                # 分路对照：同一条样本、同一份预算，只换「怎么把预算分给各问」。
                # 放在同一次循环里跑，样本与分母天然对齐（两臂之间不可能挑到不同样本）。
                for mode in route_modes:
                    try:
                        route_results.append(run_sample_arm(sample, retrieve,
                                                            budget=route_budget, mode=mode,
                                                            decompose=decompose))
                    except Exception as e:           # 单条失败不该带走整轮
                        counts["query_errors"] += 1
                        print(f"    分路检索失败（{mode}），跳过：{str(e)[:60]}")
        if conv_results:
            path_counts = {}
            for record in conv_results:
                path_counts[record.get("path")] = path_counts.get(record.get("path"), 0) + 1
            summary_text = "、".join(f"{path} {n} 条" for path, n in sorted(path_counts.items()))
            print(f"    样本 {len(conv_results)}：{summary_text}")
        results.extend(conv_results)
        counts["samples"] += len(conv_results)
        print()

    if not results:
        print("错误：一条样本都没造出来（语料太短或锚点全被过滤）。")
        print("   空结果不能当结论，直接中止；试试 --chunks-per-file 或换 --state-dir。")
        return 2

    stats = summarize(results, top_k=args.top_k, extra_k=args.extra_k)
    overall = stats["overall"]
    print("=== 复合问题检索覆盖（真实语料，gold 现场派生；只记数字）===")
    print(f"对话 {counts['conversations']} 个；参与文件 {counts['files_used']} 份；"
          f"探针 {counts['probes']} 个；挑不到锚点 {counts['anchors_missing']} 片；"
          f"因「子问题挤在同一文件」跳过的样本 {counts['samples_same_file']} 条；"
          f"检索失败 {counts['query_errors']} 条")
    print(f"单问基线（top-1 命中目标片）：{overall['baseline_hit']}/{overall['parts']}"
          f"（{_pct(overall['baseline_rate'])}）；"
          f"被排除出合并覆盖率分母的子问 {overall['parts_excluded']} 个")
    median_rank = overall["excluded_rank_median"]
    print(f"  被排除的子问里，目标片仍在原始向量前 10 的有 {overall['excluded_ranked']}"
          f"/{overall['parts_excluded']}（名次中位 "
          f"{'无数据' if median_rank is None else f'{median_rank:.1f}'}）"
          f"—— 这部分是「排序没排上来」，不是「锚点根本没把正文召回来」")
    print()
    print("按子问数量（每条样本的子问题来自不同文件）：")
    for group in stats["groups"]:
        if group["path"] == "all":
            _print_group(group, args.top_k, args.extra_k)
    print()
    print("按检索路径拆开（关键词兜底是 2-gram 词面分，且最多回 3 块，别和向量距离混看）：")
    for group in stats["groups"]:
        if group["path"] != "all":
            _print_group(group, args.top_k, args.extra_k)
    print()
    _print_group(overall, args.top_k, args.extra_k, indent="")
    print()
    print(f"结论口径提醒：覆盖@{args.top_k} 的分母是「单问基线命中的子问题」"
          f"（{overall['parts_eligible']} 个），不是全部子问；"
          f"k 从 {args.top_k} 涨到 {args.extra_k} 的收益 = {_pp(overall['gain'])}。")
    if not args.hyde:
        print("提示：这一版是 no-hyde 口径（融合退化为单探针），线上走 HyDE —— "
              "要线上口径请加 --hyde --hyde-cache。")

    route_groups = summarize_routes(route_results, budget=route_budget) if route_results else []
    if route_groups:
        print()
        _print_routes(route_groups, route_budget)
        merged_group = next((group for group in route_groups
                             if group["arm"] == ROUTE_MERGED and group["n_parts"] is None), None)
        if merged_group and route_budget == args.top_k:
            old_covered = overall[f"covered_at_{args.top_k}"]
            if merged_group["covered"] != old_covered:
                print(f"错误：merged 臂覆盖 {merged_group['covered']} 与原有口径 {old_covered} 不一致 —— "
                      f"两臂不再是同一条检索路径，这张对照表不可信，已中止。")
                return 2
            print(f"自检通过：merged 臂逐位复现了原有口径（覆盖@{args.top_k} = "
                  f"{old_covered}/{overall['parts_eligible']}），三臂可比。")
        print(f"边界：拆问用的是 gold 子问题（样本本来就由子问题拼出来），所以这张表是"
              f"「每问分路检索的上界」——线上还需要一次 query 分解，那层的拆错率与延迟不在这里。")
        if not args.hyde:
            print("成本口径：每问多一次 embedding + 一次向量查询（检索次数列已计入）；"
                  "线上还要加一次分解调用的延迟。")

    if args.json:
        payload = {
            "config": {"seed": args.seed, "top_k": args.top_k, "extra_k": args.extra_k,
                       "chunks_per_file": args.chunks_per_file,
                       "limit_files": args.limit_files, "hyde": bool(args.hyde),
                       "join_style": args.join_style, "route_budget": route_budget,
                       "anchor_len": [ANCHOR_MIN_LEN, ANCHOR_MAX_LEN],
                       "anchor_max_corpus_hits": ANCHOR_MAX_CORPUS_HITS},
            "counts": counts,
            "groups": stats["groups"],
            "overall": overall,
            "samples": results,
            "route": {"budget": route_budget, "groups": route_groups,
                      "samples": route_results},
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        # 安全网：明细里一旦出现锚点原文或文件名就中止，绝不把语料副本写进任何文件
        leaks = find_leaks(text, corpus_needles)
        if leaks:
            print(f"错误：明细里有 {leaks} 处语料泄漏（锚点或文件名），已中止，没有写盘。")
            return 2
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
        print(f"明细已落盘：{args.json}（{len(results)} 条样本，只有 id/布尔/距离/路径）")

    if args.hyde and args.hyde_cache:
        save_hyde_cache(args.hyde_cache, hyde_cache)
        print(f"假设文本已缓存：{args.hyde_cache}（{len(hyde_cache)} 条）")
    if args.decompose_cache and decompose_cache:
        save_decompose_cache(args.decompose_cache, decompose_cache)
        print(f"拆问结果已缓存：{args.decompose_cache}（{len(decompose_cache)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
