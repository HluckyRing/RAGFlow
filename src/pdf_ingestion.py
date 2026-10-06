import re
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_long_text(text, chunk_size, overlap):
    """将超长文本按句子边界切成不超过 chunk_size 的小段"""
    sentences = re.split(r'(?<=[。！？.!?\n])\s*', text)
    pieces = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current:
                pieces.append(current.strip())
            # 如果单个句子仍超长，硬切
            safe_overlap = min(overlap, chunk_size - 1)
            while len(sent) > chunk_size:
                pieces.append(sent[:chunk_size])
                sent = sent[chunk_size - safe_overlap:]
            current = sent
    if current.strip():
        pieces.append(current.strip())
    return pieces


def split_text(text, chunk_size=None, overlap=None):
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 先把超长段落拆成可管理的小段
    flat = []
    for para in paragraphs:
        if len(para) > chunk_size:
            flat.extend(_split_long_text(para, chunk_size, overlap))
        else:
            flat.append(para)

    chunks = []
    current_chunk = ""
    for piece in flat:
        if len(current_chunk) + len(piece) <= chunk_size:
            current_chunk += piece + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            # 保证「重叠 + 新段」不超 chunk_size：否则整块会超出 embedding 的 token 窗口
            # （硬切出来的整长句最容易撞上这条）。宁可不重合这一段，也不能让块超窗。
            if len(overlap_text) + len(piece) > chunk_size:
                overlap_text = ""
            current_chunk = overlap_text + piece + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ── 来源元数据（P2）──

# loader 留下的结构标记：pptx 与 pdf 是页码，xlsx 是工作表名。只认「整行就是标记」，
# 避免正文里恰好写到「=== 第 3 页 ===」这几个字时被误判成结构。
_MARKER_LINE_RE = re.compile(
    r'^=== (?:(?P<page>第 \d+ 页)|(?P<sheet>.+?)) ===$', re.MULTILINE)


def _section_spans(text):
    """返回 [(start, end, label)]：从每个标记行本身开始、到下一个标记行之前的区间。

    没有标记的格式（docx / csv / txt / md）返回空列表，调用方据此把 section 留空。
    """
    spans = []
    current_label = None
    current_start = 0
    for match in _MARKER_LINE_RE.finditer(text):
        if current_label is not None:
            spans.append((current_start, match.start(), current_label))
        current_label = match.group("page") or match.group("sheet")
        current_start = match.start()      # 区间从标记行本身开始：块首就是标记时也算这一节
    if current_label is not None:
        spans.append((current_start, len(text), current_label))
    return spans


def _split_by_sections(text):
    """按结构标记把正文切成 [(section, 该节正文), ...]，标记行本身不进正文。

    每一节**独立切片**，所以一个块永远不会跨两页/两个工作表 —— 页码才可能是准的。
    这是 2026-07-24 那次「引用不准确」的直接修法：当时按整篇切，块与页没有确定的
    对应关系，硬贴一个页码必然会说错。
    """
    spans = _section_spans(text)
    if not spans:
        return [(None, text)]
    parts = []
    if spans[0][0] > 0:                       # 第一个标记之前的内容没有归属
        head = text[:spans[0][0]].strip()
        if head:
            parts.append((None, head))
    for start, end, label in spans:
        body = _MARKER_LINE_RE.sub("", text[start:end], count=1).strip()
        if body:
            parts.append((label, body))
    return parts


def split_files(files, chunk_size=None, overlap=None):
    """按文件、按节切片，返回 [(chunk, meta), ...]。

    与 `split_text(把多文件拼成一段)` 的区别：**块不跨文件、也不跨页/工作表**。
    引用要可核验，前提是每个块只有唯一来源。

    meta 形如 `{"file_name": ..., "chunk_index": 0, "section": "第 3 页" | None}`；
    section 只在 loader 留下了结构标记时才有值（pdf/pptx 是页码、xlsx 是工作表名）。
    """
    out = []
    for file_name, text in files or []:
        if not text:
            continue
        index = 0
        for section, body in _split_by_sections(text):
            for chunk in split_text(body, chunk_size=chunk_size, overlap=overlap):
                out.append((chunk, {
                    "file_name": file_name or "未知文件",
                    "chunk_index": index,
                    "section": section,
                }))
                index += 1
    return out
