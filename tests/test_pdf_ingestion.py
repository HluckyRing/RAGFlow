# -*- coding: utf-8 -*-
"""切片回归（P1）：块长必须留在 embedding 的 token 窗口内。

背景：`BAAI/bge-small-zh-v1.5` 的 `max_seq_length=512`，中文约 1 字 1 token。
`CHUNK_SIZE` 原来是 600，超出窗口的部分会被 tokenizer 静默截掉。缩到 480 只是
第一步 —— 更关键的是 `split_text` 自己不能切出比 `chunk_size` 还长的块，
否则缩了也白缩。
"""
from src.config import CHUNK_SIZE
from src.pdf_ingestion import split_text

EMBEDDING_MAX_TOKENS = 512   # bge-small-zh-v1.5 的 max_seq_length


def test_default_chunk_size_fits_embedding_window():
    """默认切片长度必须给特殊 token（[CLS]/[SEP]）留出位置。"""
    assert CHUNK_SIZE + 2 <= EMBEDDING_MAX_TOKENS, (
        "中文约 1 字 1 token；CHUNK_SIZE 超出模型窗口的部分会被静默截断")


def test_no_chunk_exceeds_chunk_size():
    """长句被硬切、段落被拼装 —— 两条路径都不许产出超长的块。

    这里刻意用「一整段没有任何标点的长文本」触发硬切分支：硬切出来的片段本身就
    有 chunk_size 长，再拼上 overlap 就会超窗（修复前会切出 overlap + chunk_size
    那么长的块）。
    """
    text = "一" * 1500 + "\n\n" + "短段落。" * 100

    chunks = split_text(text, chunk_size=100, overlap=30)

    assert chunks, "不该切出空结果"
    assert max(len(c) for c in chunks) <= 100, \
        "块长超过 chunk_size 就会被 embedding 截断"


def test_long_sentence_keeps_overlap_between_chunks():
    """硬切长句时块之间仍要重叠，否则答案正好跨在切口上就断了。"""
    text = "".join(f"{i:03d}" for i in range(150))   # 450 字、无标点，必走硬切

    chunks = split_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)
    assert chunks[0][-10:] == chunks[1][:10], "硬切的相邻块必须保留 overlap"


def test_short_paragraphs_are_packed_and_nothing_is_lost():
    """普通短段落：能装就装、装不下才分块，块长不超限，内容不丢。"""
    para = "这是一段正好二十个字的测试文本。"
    text = "\n\n".join([para] * 6)

    chunks = split_text(text, chunk_size=60, overlap=10)

    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    assert "".join(chunks).count("测试文本") >= 6, "每个段落都要有归属，不能丢内容"
