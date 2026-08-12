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
            current_chunk = overlap_text + piece + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
