# -*- coding: utf-8 -*-
"""切片回归（P1）：块长必须留在 embedding 的 token 窗口内。

背景：`BAAI/bge-small-zh-v1.5` 的 `max_seq_length=512`，中文约 1 字 1 token。
`CHUNK_SIZE` 原来是 600，超出窗口的部分会被 tokenizer 静默截掉。缩到 480 只是
第一步 —— 更关键的是 `split_text` 自己不能切出比 `chunk_size` 还长的块，
否则缩了也白缩。
"""
from src.config import CHUNK_SIZE
from src.pdf_ingestion import split_files, split_text

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


# ── 来源元数据（P2）──

def test_split_files_keeps_each_chunk_to_one_file():
    """块不跨文件：引用要可核验，前提是每个块只有一个来源。"""
    pairs = split_files([("a.md", "甲甲甲"), ("b.md", "乙乙乙")])

    assert [(text, meta["file_name"]) for text, meta in pairs] == [
        ("甲甲甲", "a.md"), ("乙乙乙", "b.md")]
    assert [meta["chunk_index"] for _, meta in pairs] == [0, 0], "块序号按文件各自从 0 开始"


def test_split_files_numbers_chunks_within_each_file():
    pairs = split_files([("a.md", "甲" * 1000), ("b.md", "乙" * 1000)])

    for name in ("a.md", "b.md"):
        indices = [meta["chunk_index"] for _, meta in pairs if meta["file_name"] == name]
        assert len(indices) > 1, "长文本该切成多块，这条断言才有意义"
        assert indices == list(range(len(indices))), "每个文件的块序号连续、从 0 开始"


def test_split_files_never_mixes_two_pages_into_one_chunk():
    """页码准确的前提：一个块只能属于一页 —— 短页也不能和下一页揉在一起。"""
    text = "=== 第 1 页 ===\n甲甲\n\n=== 第 2 页 ===\n乙乙"

    pairs = split_files([("x.pdf", text)])

    assert [(chunk, meta["section"]) for chunk, meta in pairs] == [
        ("甲甲", "第 1 页"), ("乙乙", "第 2 页")]
    assert all("===" not in chunk for chunk, _ in pairs), "标记行是结构，不该混进正文"


def test_split_files_labels_every_chunk_of_a_long_page():
    """一页太长被切成多块时，每一块都带同一页码（它们确实都在那一页）。"""
    pairs = split_files([("x.pdf", "=== 第 7 页 ===\n" + "甲" * 1000)])

    assert len(pairs) > 1
    assert {meta["section"] for _, meta in pairs} == {"第 7 页"}
    assert {meta["file_name"] for _, meta in pairs} == {"x.pdf"}


def test_split_files_uses_sheet_names_for_xlsx_markers():
    text = "=== 成绩单 ===\n张三 | 90\n\n=== 出勤 ===\n李四 | 全勤"

    pairs = split_files([("x.xlsx", text)])

    assert [meta["section"] for _, meta in pairs] == ["成绩单", "出勤"]


def test_split_files_section_is_none_without_markers():
    """docx/csv/txt/md 没有结构标记：只给文件名，不给 section。"""
    pairs = split_files([("a.md", "甲甲甲\n\n乙乙乙")])

    assert pairs and all(meta["section"] is None for _, meta in pairs)


def test_split_files_skips_empty_files():
    pairs = split_files([("empty.md", ""), ("a.md", "内容")])

    assert [(chunk, meta["file_name"]) for chunk, meta in pairs] == [("内容", "a.md")]


def test_split_files_respects_chunk_size_override():
    """评测脚本要用 --chunk-size 覆盖，所以切片参数必须能透传。"""
    pairs = split_files([("a.md", "甲" * 100)], chunk_size=40, overlap=0)

    assert len(pairs) > 1
    assert all(len(chunk) <= 40 for chunk, _ in pairs)
