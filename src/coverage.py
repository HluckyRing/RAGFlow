# -*- coding: utf-8 -*-
"""跨文件取样：让「枚举 / 汇总型」提问不要再静默漏掉整份文件。

问题（2026-11-12 第十二轮实测）
------------------------------
真实用户问「列出所有的实验」「该知识库的主要内容是什么」时，检索按**与问句的相关度**排序、
再按 `MAX_DISTANCE` 砍，于是**相关度稍低的那份文件整份消失**：13 份文件的语料里，装配后的
context 只覆盖 12 份，用户和模型都不知道少了一份。这类提问要的不是「最相关的 k 块」，
而是**跨全部文件的代表性取样**。

三层职责，都在这个模块里，都是纯函数（不碰 Chroma、不花 API）：
1. `looks_enumerative()` —— 识别「枚举 / 汇总」措辞。**这是一张措辞表，永远是脆的**，
   所以它只决定「要不要多花一次补位查询」，绝不决定「答什么、算不算对」；
2. `sample_per_file()` —— 把已有命中按文件轮流取样：先保证每份文件各 1 块，再轮到第 2 块；
3. `missing_files()` / `coverage_report()` —— 数清楚还差哪几份（补位与告警都靠它）。

为什么「按文件轮流」而不是「再按相似度排一遍」：枚举型提问的每个文件都同等重要，
用相似度排序必然把「问句字面更像」的那几份排在前面，把边缘但必要的那几份挤掉。
"""
from collections import OrderedDict
import re

# 「枚举 / 汇总」措辞在 `src/gates/`（默认 zh，`GATE_PACK` 选包）：宁可漏判（少补一次位，
# 行为退回今天）也不要误判（多花一次查询）—— 但它本身不改变答案正确性。
import src.config as _config
from src.gates import load_pack as _load_pack

_gates = _load_pack(_config.GATE_PACK, logger=_config.logger)
ENUMERATION_MARKERS = _gates.enumeration_markers


def looks_enumerative(question, markers=ENUMERATION_MARKERS):
    """这条提问是不是「枚举 / 汇总型」。只用于「要不要做跨文件取样」。"""
    text = str(question or "")
    return any(marker in text for marker in markers)


def file_names_of(hits):
    """命中块涉及的文件名（去重、保持首次出现顺序）。没有来源的块不计。"""
    names = OrderedDict()
    for _text, meta in hits or []:
        name = (meta or {}).get("file_name")
        if name:
            names.setdefault(name, None)
    return list(names)


def missing_files(hits, file_names):
    """`file_names` 里哪些文件在 `hits` 里一块都没有。"""
    seen = set(file_names_of(hits))
    return [name for name in file_names or [] if name and name not in seen]


def sample_per_file(hits, budget):
    """按文件轮流取样：每份文件先各取 1 块，再轮流取第 2 块……直到 `budget` 块。

    - 顺序以**文件首次出现的先后**为准（也就是相似度最高的那份先取第 1 块），
      文件之间是公平的，但文件内部仍按相似度；
    - 没有来源（`meta` 为 None）的块归到 `None` 这一组，按原顺序排在最后参与轮转
      —— 老索引没有 metadatas 时不能让这些块整批消失。
    """
    buckets = OrderedDict()
    for text, meta in hits or []:
        name = (meta or {}).get("file_name")
        buckets.setdefault(name, []).append((text, meta))
    if not buckets:
        return []

    picked = []
    round_index = 0
    while len(picked) < max(0, budget or 0):
        added = False
        for bucket in buckets.values():
            if round_index < len(bucket):
                picked.append(bucket[round_index])
                added = True
                if len(picked) >= budget:
                    break
        if not added:
            break
        round_index += 1
    return picked


def coverage_report(hits, file_names):
    """覆盖盘点：`{"covered", "total", "missing", "blocks"}`。"""
    names = file_names_of(hits)
    total = [name for name in (file_names or []) if name]
    return {
        "covered": len([name for name in total if name in set(names)]),
        "total": len(total),
        "missing": missing_files(hits, total),
        "blocks": len(hits or []),
    }


# ── 导航页 / 行政页（2026-11-12 第十四轮实测出来的真病根）──
#
# 枚举 / 汇总型提问检索到的头部块里，绝大多数是**结构页而不是内容页**：
#   · 「第N章知识导图 第1章 | 操作系统引论 第2章 | 进程的描述与控制 …」——每章都有一页，
#     跟任何「讲的是什么 / 主要内容」的提问都极像，于是把正文页全挤掉；
#   · 「内容导航： 1.1 … 1.2 …」——目录页；
#   · 「第一次作业 1 简答题 2 3 4 … 标黄色为本次作业」——作业编号页，一个字的信息都没有。
# 模型拿到这些只能复述「材料只给了章节名」，于是「覆盖变全」并不等于「答得更全」。
#
# 这里刻意**只降权不删除**：小语料里那页可能就是唯一的答案（例如「有哪些题型」的题型清单
# 就长在提纲页上），删干净会让原本能答的变成答不出。顺序永远是「内容页在前、导航页在后」。
NAVIGATION_MARKERS = _gates.navigation_markers
# 「第1章 | 第2章 | …」这种把章节名排成一张表的页面：同一块里出现 ≥5 个「第N章」几乎只有导图页
CHAPTER_LIST_MIN = 5
_CHAPTER_RE = re.compile(r"第\s*[0-9一二三四五六七八九十]{1,3}\s*章")


def is_navigational(text, structural=None):
    """这一块是不是导航 / 行政页（目录、知识导图、作业编号页）。

    三条判据，按可靠性排：
    1. **跨文件近重复**（`structural` 参数，见 `structural_texts()`）—— 语言无关，
       这是主力判据；
    2. 命中 `NAVIGATION_MARKERS` 措辞表 —— 换语言就失效，只当兜底；
    3. 同一块里把章节名排成表（≥5 个「第N章」）—— 只对中文教材式 PPT 成立。

    只用于**排序**，不用于删除 —— 见本节开头那段的说明。
    """
    body = str(text or "")
    if not body.strip():
        return False
    if structural and _text_key(body) in structural:
        return True
    if any(marker in body for marker in NAVIGATION_MARKERS):
        return True
    return len(_CHAPTER_RE.findall(body)) >= CHAPTER_LIST_MIN


def prefer_content(hits):
    """稳定重排：内容页在前、导航页在后（各自保持原有相对顺序）。

    为什么是「稳定分区」而不是打分排序：块之间原本的名次是检索器给的，唯一该动的信息是
    「这块是不是结构页」。多打一层分就会引入新的口径，评测与线上都要跟着重校。
    """
    items = list(hits or [])
    structural = structural_texts(items)      # 先算「跨文件近重复」这一条语言无关的判据
    content, navigation = [], []
    for item in items:
        (navigation if is_navigational(item[0], structural) else content).append(item)
    return content + navigation


# ── 同一份文件内部跨节取样（2026-11-12 第十六轮）──
#
# 真实需求来自真实日志：「帮我整理第一章的内容」。这句话要的不是「跨文件的代表性」，
# 而是**把那一章（同一份文件里的 64 页）铺开讲**。文件级取样对它完全无效 —— 实测两臂都拒答，
# 因为第 1 章的正文块在检索结果里只剩几个，其余名额被别的文件占满。
#
# 两个关键设计：
# 1. **先定位是哪一份文件**：只认问句里的章号（「第一章」「第2章」），再拿章号去文件名里找。
#    不做「问问句像不像某个文件名」那种模糊匹配 —— 匹配错了会答成另一章，比不取样更糟。
# 2. **按节均匀铺开**，不是「取最相关的 k 块」：整理 / 摘要要覆盖整章，按相关度取必然抓头几条。
#    又因为装配阶段还有 `MAX_CONTEXT_LENGTH` 的字数上限（约 12 块），取样块数按**装配预算**
#    定而不是按检索预算定 —— 取 39 块再被截成 14 块，等于只覆盖了前四分之一章。

_CHINESE_NUMBERS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8,
                    "九": 9, "十": 10, "十一": 11, "十二": 12, "十三": 13, "十四": 14,
                    "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20}
_CHAPTER_MENTION_RE = re.compile(r"第\s*([0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*章")
_SECTION_PAGE_RE = re.compile(r"第\s*([0-9]{1,4})\s*页")


def chapter_numbers(question):
    """问句里提到的章号（「第一章」→ 1、「第2章」→ 2）。没提到就是空集合。"""
    numbers = set()
    for raw in _CHAPTER_MENTION_RE.findall(str(question or "")):
        if raw.isdigit():
            numbers.add(int(raw))
        else:
            value = _CHINESE_NUMBERS.get(raw)
            if value:
                numbers.add(value)
    return numbers


def pick_target_file(question, file_names):
    """问句指向哪一份文件：用**章号**去文件名里找。

    只认章号这一条硬信号，返回唯一匹配；找不到或有歧义都返回 `None`
    —— 定位错了会答成另一章，比不取样更糟。
    """
    numbers = chapter_numbers(question)
    if not numbers:
        return None
    matched = []
    for name in file_names or []:
        for number in numbers:
            if any(f"第{number}{sep}章" in name for sep in ("", " ")) or \
                    any(f"第{sep}{number}章" in name for sep in (" ",)):
                matched.append(name)
                break
    return matched[0] if len(matched) == 1 else None


def intra_file_report(picked, total_candidates, file_name, pool=None):
    """整章取样的**覆盖账**：这次到底摘了几段、目标文件一共几段（PDF/PPT 上一段 = 一页）。

    为什么需要它（2026-11-15）：内置上下文预算 `MAX_CONTEXT_LENGTH=8000` 装不下 100 页的课件
    （整章 10 541–15 616 字符），取样必然只覆盖一部分页。**模型和用户都不知道少了什么** ——
    回答看起来像「整章都整理完了」，实际上章尾整段没读到。有了这本账才能：
    ① 在 context 里告诉模型「你只看到 N/M 段，别假装看全了」；
    ② 通过 SSE 告诉用户同一件事（前端渲染成一个小标签）。

    只用计数、不用比例：一段一页的语料上两者等价，而 md 语料一段 480 字，
    报「N/M 段」永远为真，报「页」会撒谎。

    `pool` 可选：给了**本文件的全部候选块**时，额外算出「哪些页标题一次都没读到」
    （见 `topic_gaps`）—— 光有总账说不出「缺的是什么」，这一列补的就是那句。
    """
    covered = len(picked or [])
    total = max(int(total_candidates or 0), covered)
    report = {"kind": "intra_file", "file_name": file_name or "",
              "covered": covered, "total": total}
    if isinstance(pool, (list, tuple)):
        report.update(topic_gaps(pool, picked))
    return report


# ── 缺口清单：缺的是哪几段、那几段讲什么（2026-11-20 第二十轮）──
#
# 为什么不是「按小节编号归属」：原型实测三份语料三种坏法（负结果记在
# `.agents/intra-sections-spec.md`）——大纲页**一次性列全章小节**、不带「哪一节从哪页开始」
# 的信息；靠「页首标题 ≈ 小节标题」去猜，第1章 `1.6 操作系统的结构设计` 的正文页首行是
# 「操作系统结构设计」（差一个「的」）就锚不上，第4章还会算出 `p12-0` 这种负区间，
# ML 的 PDF 干脆没有编号大纲、正文算式反被当成标题。**猜错的「缺失小节」比不说更糟。**
#
# 所以退一步，只报**能验证的东西**：没进 context 的块，它们的页标题是什么。每个标题都来自
# 真实块文本，判据是「这个标题在 picked 块里一次都没出现过」——不需要知道小节边界，也不会编。
#
# 第二十三轮（2026-11-25）拿真实 state/ 语料（36 份文件 / 2886 块）量了第一版的三个毛病，
# 全部在这里治掉：
#   ① **行政页霸榜**：缺口条目频次前三名是「课程思政小讲堂」19 次、「学 习 进 步 ！」19 次、
#      「第二次作业」17 次 —— 每章 PPT 末尾都有一页，内容为零。原因很蠢：噪音表用的是
#      **子串**匹配，而真实文本是「学 习 进 步 ！」（逐字加空格），一个都没命中。改成**整行
#      去空白归一化后精确匹配**一张行政页表（不用子串：子串会把「作业调度算法」「两级文件目录」
#      这种真内容一起误杀，实测行政词子串命中 24 个条目里只有 4 个是真行政页）。
#   ② **前 N 条只覆盖文件开头**：缺口按文件顺序取前 4 条，122 块的文件报的永远是开头那几页，
#      章末整段从不出现。改成**按位置均匀铺开（首尾必含）**——与 `sample_evenly` 同一个哲学：
#      说得出「首中尾都缺」，而不是「开头缺几页」。
#   ③ **「4 条」是拍的数**：真实语料里 122 块的文件有 85–102 处缺口，4 条 = 4%。改成 **12 条上限**
#      —— 这个数不是从算法推的，是**可读性**上限（前端 `coverage-tag` 要把清单塞进一个标签里）+
#      一个可证明的字数上界：标题最多 18 字（`_TITLE_MAX`），12 × 19 = **228 字**，占 context 预算
#      8000 的 2.9%；真实语料实测每份清单中位 122 字、最长 191 字。曾试过「按字符预算动态定条数」，
#      但上限 12 × 标题上界 18 字 = 228 < 300 的预算 —— **那一层永远不生效，是死代码**，故删除。
# 另外补一条：**解析不出标题的页不再静默消失**。34.6% 的块没有可解析标题（998/2886），其中
# 628 块的 `section` 里有页码。这些页现在用「第 N 页」诚实命名 —— 但**排在标题条目之后**
# （稳定分区，与 `prefer_content` 同一个哲学）：真实语料里 PPT 的无标题页大多是封面/教材信息/
# 知识导图页，若与内容标题混排会把清单头部占满（实测 12 条里 2 条是「第 2 页」这种噪音）。
# 结构页（知识导图 / 内容导航 / 目录）与行政页既不命名也不计入缺口。
_NOISE_TITLES = {"date", "signature", "内容提要"}
_STRUCTURAL_TITLES = {"内容导航：", "内容导航", "目录", "/content/"}
_NOISE_HINTS = ("主讲", "主审", "作者：", "版权", "谢谢", "学习进步")
# 行政页（整页只有一个行政标题、内容为零）。**精确匹配**，不做子串 —— 见上面 ①。
_ADMIN_TITLES = frozenset({
    "课程思政小讲堂", "学习进步", "学习进步！", "学习进步!", "谢谢", "谢谢！", "谢谢!",
    "第一次作业", "第二次作业", "第三次作业", "作业", "上机练习", "上机练习：",
})
_BULLET_PREFIX = "⚫➢•·●■◆◇*＊-—–〉》"
_PAGE_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[/／]\s*")
_TITLE_MAX = 18
_TITLE_SCAN_LINES = 3        # 只在前 3 行里找标题：再往下就是正文了
_TITLE_RAW_MAX = 24         # 长过这个数的一律当正文，不许截断后冒充标题
TOPIC_LIMIT = 12            # 可读性上限（不是算法推出来的）：字数上界 12 × (18+1) = 228 字


def _normalize_line(line):
    """比对用的归一化：去掉全部空白（含全角空格）+ 转小写。"""
    return re.sub(r"\s+", "", str(line or "")).lower()


def _scan_title(text):
    """扫块的「页标题」，返回 `(kind, label)`。

    kind ∈ `"title"`（拿到标题）/ `"structural"`（目录、知识导图）/ `"admin"`（行政页，
    如「第二次作业」）/ `"untitled"`（前 3 行全是正文或空）。三种非 title 的区别很重要：
    结构页与行政页**什么都不报**，untitled 还能用页码兜底（见 `page_label`）。
    """
    for index, raw in enumerate(str(text or "").split("\n")):
        if index >= _TITLE_SCAN_LINES:
            break
        line = _PAGE_PREFIX_RE.sub("", raw.strip()).strip()
        line = line.lstrip(_BULLET_PREFIX).strip()
        if not line or line.isdigit() or len(line) < 2:
            continue
        if len(line) > _TITLE_RAW_MAX:
            return "untitled", ""            # 这是正文，不是标题 —— 别再往下抓一行短的凑数
        # 结构页（目录 / 知识导图）就地判死：它们后面的行是全章清单，当标题会张冠李戴。
        if line.lower() in _STRUCTURAL_TITLES or (line.startswith("第") and line.endswith("知识导图")):
            return "structural", ""
        if _normalize_line(line) in _ADMIN_TITLES:
            return "admin", ""
        if line.lower() in _NOISE_TITLES or any(hint in line for hint in _NOISE_HINTS):
            continue
        return "title", (line if len(line) <= _TITLE_MAX else line[:_TITLE_MAX] + "…")
    return "untitled", ""


def page_title(text):
    """块的「页标题」：前 3 行里第一个像标题的行（去掉 `67 / ` 前缀、丢掉噪音行），截到 18 字。

    三条边界都是被真实语料逼出来的：
    ① 只扫前 3 行 —— 正文里随便一行拿来当标题比不报更糟；
    ② 原始行超过 24 字当正文（PPT 正文行普遍 25 字以上），不许「截断一个长句子」冒充标题；
    ③ 噪音行（`Date` / `Signature` / 版权行）跳过，结构页与行政页直接判死。
    拿不到就返回空串 —— 单看它无法区分「正文开头」和「行政页」，要区分用 `_scan_title`。
    """
    kind, label = _scan_title(text)
    return label if kind == "title" else ""


def page_label(text, meta=None):
    """缺口条目的名字，返回 `(label, is_page)`。

    优先级：页标题 → 「第 N 页」（`section` 里的页码，如 `"第 3 页"`）→ 空串（不报）。
    结构页 / 行政页一律空串：它们不是内容，报出来只是噪音（见本节开头 ①）。
    页码兜底是给「页首就是正文长句」的页留的活口 —— 这类页占真实语料 34.6% 的块。
    """
    kind, label = _scan_title(text)
    if kind == "title":
        return label, False
    if kind in ("structural", "admin"):
        return "", False
    match = _SECTION_PAGE_RE.search(str((meta or {}).get("section") or ""))
    if match:
        return f"第 {int(match.group(1))} 页", True
    return "", False


def _even_indices(total, count):
    """`[0, total)` 里均匀取 `count` 个下标，**首尾必含**。"""
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    step = (total - 1) / float(count - 1)
    picked, seen = [], set()
    for index in range(count):
        position = int(round(index * step))
        if position not in seen:
            seen.add(position)
            picked.append(position)
    return picked


def _spread(items, limit):
    """按位置均匀铺开取 `limit` 条，**首尾必含**。

    为什么是均匀而不是「按文件顺序取前 N」：真实语料里 122 块的文件有 84–102 处缺口，
    取前 N 永远停在文件开头，章末整段一次都不会被点到 —— 用户会以为「只缺开头几页」。
    """
    return [items[index] for index in _even_indices(len(items), limit)]


def topic_gaps(pool, picked):
    """缺口清单：`pool` 里有、但 `picked` 里一份都没读到的页。

    判据刻意写成「标题在 picked 里一次都没出现」而不是「块没被选中」：同一标题常横跨好几页
    （「岭回归」连排 3 页），读了 1 页就算这个标题摸到了 —— 报成「缺」会夸大缺口。

    返回 `{"topics_total": int, "topics_missing": [...], "missing_count": int}`：
    `topics_missing` 是有标题的缺口（按位置均匀铺开，最多 `TOPIC_LIMIT` 条）+ 缺口太大时
    补上的「第 N 页」；总字数上界 `TOPIC_LIMIT × (_TITLE_MAX + 1)` = 228 字。
    `missing_count` 是真实总数，与列了几条无关。
    一个名字都解析不出来时返回空 dict（调用方据此退回老话术，不编）。
    """
    picked_keys = {(str(text or ""), (meta or {}).get("chunk_index"))
                   for text, meta in (picked or [])}
    covered, titles, pages = set(), [], []
    for text, meta in pool or []:
        label, is_page = page_label(text, meta)
        if not label:
            continue
        if (str(text or ""), (meta or {}).get("chunk_index")) in picked_keys:
            covered.add(label)
            continue
        bucket = pages if is_page else titles
        if label not in bucket:
            bucket.append(label)
    if not covered and not titles and not pages:
        return {}
    missing_titles = [title for title in titles if title not in covered]
    missing_pages = [page for page in pages if page not in covered]
    selected = _spread(missing_titles, TOPIC_LIMIT)
    if len(selected) < TOPIC_LIMIT and missing_pages:
        # 页码条目排在内容标题之后：它们只在标题不够填满上限时才补位（见本节开头那段的实测）
        selected += _spread(missing_pages, TOPIC_LIMIT - len(selected))
    return {"topics_total": len(covered) + len(missing_titles) + len(missing_pages),
            "topics_missing": selected,
            "missing_count": len(missing_titles) + len(missing_pages)}


def intra_file_notice(report):
    """把覆盖账翻译成**给模型看的一句话**；全覆盖（或没有账）返回空串。

    两句话：总账（摘了 N/M 段）+ 缺口清单（哪些页一次都没读到）。
    解析不出任何页名（既无标题也无页码）的语料上只说总账 —— 与 2026-11-15 那版逐字一致，不许编。

    缺口清单用「这些页」而不是「这些页标题」：清单里可能混着「第 3 页」这种页码条目
    （页首就是正文长句、没有标题可解析的页），叫「页标题」会把页码说成标题。
    """
    if not report or report.get("kind") != "intra_file":
        return ""
    covered, total = int(report.get("covered") or 0), int(report.get("total") or 0)
    if covered <= 0 or total <= 0 or covered >= total:
        return ""
    name = report.get("file_name") or "这份文件"
    missing = list(report.get("topics_missing") or [])
    count = int(report.get("missing_count") or 0)
    if missing:
        listed = "、".join(missing)
        more = f" 等 {count} 处" if count > len(missing) else ""
        return (f"【覆盖说明】本次只从《{name}》里摘录了 {covered}/{total} 段（按页均匀取样），"
                f"这些页一次都没读到：{listed}{more}。"
                f"回答时请如实说明「本次只覆盖了这份文件的一部分」，不要假装看过全部内容。")
    return (f"【覆盖说明】本次只从《{name}》里摘录了 {covered}/{total} 段（按小节均匀取样），"
            f"未被摘录的小节可能正是你要的细节。回答时请如实说明「本次只覆盖了这份文件的部分"
            f"小节」，不要假装看过全部内容。")


def section_order(meta):
    """排序键：能解析出「第 N 页」就按页号，否则退回 chunk_index（保证稳定）。"""
    meta = meta or {}
    match = _SECTION_PAGE_RE.search(str(meta.get("section") or ""))
    if match:
        return (0, int(match.group(1)), int(meta.get("chunk_index") or 0))
    return (1, int(meta.get("chunk_index") or 0), 0)


def sample_evenly(hits, blocks):
    """在**同一份文件内部**按节均匀取样：`blocks` 块铺满整份文件的节区间。

    为什么均匀而不是按相似度：整理 / 摘要类需求要覆盖整章；按相关度取必然抓头几条、
    尾部整段丢掉。返回顺序按节号升序（装配进 context 也是这个顺序）。
    """
    ordered = sorted(hits or [], key=lambda item: section_order(item[1]))
    if not blocks or blocks <= 0 or len(ordered) <= blocks:
        return ordered
    step = len(ordered) / float(blocks)
    picked, seen = [], set()
    for index in range(blocks):
        item = ordered[int(index * step)]
        if id(item) in seen:
            continue
        seen.add(id(item))
        picked.append(item)
    return picked


# 每块进 context 时还要带一行「【来源：文件名 · 第 N 页】」，估算字数时得把它算进去。
SOURCE_LABEL_OVERHEAD = 40


def blocks_that_fit(hits, max_blocks, max_chars=8000):
    """按**字符预算**反推能取几块：短页多取、长块少取。

    为什么不能写死块数：同一套代码在 PPT（每页 ~140 字）与 md（每块 480 字）上，
    「12 块」分别是 1.7k 与 5.8k 字符 —— 写死要么浪费预算、要么被装配截断后只覆盖前半段。
    """
    items = list(hits or [])
    if not items:
        return 0
    average = sum(len(text or "") for text, _meta in items) / float(len(items))
    per_block = average + SOURCE_LABEL_OVERHEAD
    if per_block <= 0:
        return max(1, int(max_blocks or 1))
    fits = int(max_chars // per_block)
    return max(1, min(int(max_blocks or 1), fits))


# ── 语言无关的「结构页」检测（2026-11-12 第十七轮）──
#
# 措辞表换语言就失效，但**结构页有多硬的语言无关特征**：同一段结构说明（目录、知识导图、
# 章节清单）在不同文件里几乎逐字重复。中文语料里「第N章知识导图」在 12 份 PPT 里各有一页，
# 英文语料里同样会是「Table of contents / Chapter 1 …」各有一页。
#
# 做法：字符 4-gram 的 Jaccard 相似度 ≥ 0.6 视为「近重复」，且这份近重复要**跨 ≥3 份文件**
# 才算结构页 —— 两个人引用同一句话是内容，每份文件都抄一遍同一张目录才是结构。
# 只在**候选块之间**比较（≤ 40 块，O(n²) 也在毫秒级），不需要额外检索。
_STRUCTURAL_SHINGLE = 4
_STRUCTURAL_JACCARD = 0.6
_STRUCTURAL_MIN_FILES = 3


def _shingles(text):
    body = re.sub(r"\s+", "", str(text or ""))
    if len(body) < _STRUCTURAL_SHINGLE:
        return {body} if body else set()
    return {body[i:i + _STRUCTURAL_SHINGLE]
            for i in range(len(body) - _STRUCTURAL_SHINGLE + 1)}


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _text_key(text):
    return re.sub(r"\s+", "", str(text or ""))


def structural_texts(hits, threshold=_STRUCTURAL_JACCARD, min_files=_STRUCTURAL_MIN_FILES):
    """挑出「跨 ≥`min_files` 份文件近重复」的块文本 —— 语言无关的结构页判据。

    返回**归一化文本**的集合（`prefer_content` 直接拿它比对）。
    """
    items = list(hits or [])
    if len(items) < min_files:
        return set()
    shingle_cache = [_shingles(text) for text, _meta in items]
    flagged = set()
    for index, (text, meta) in enumerate(items):
        name = (meta or {}).get("file_name")
        neighbours = set()
        for other_index, (_other_text, other_meta) in enumerate(items):
            if other_index == index:
                continue
            other_name = (other_meta or {}).get("file_name")
            if not other_name or other_name == name:
                continue                    # 同一份文件内重复不算（同一章的页眉页脚会重复）
            if _jaccard(shingle_cache[index], shingle_cache[other_index]) >= threshold:
                neighbours.add(other_name)
        if len(neighbours) + 1 >= min_files:
            flagged.add(_text_key(text))
    return flagged
