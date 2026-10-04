# -*- coding: utf-8 -*-
"""文本清洗：孤立代理字符必须被剥离。

来由（2026-10-03 真实 500）：pypdf 解析字体映射损坏的 PDF 时会产出孤立代理字符
（U+D800–U+DFFF）。它会让两处直接炸掉：
- chromadb → sentence-transformers 的 tokenizer：
  TypeError: TextEncodeInput must be Union[...]
- state 的 json.dump(ensure_ascii=False)：
  UnicodeEncodeError: 'utf-8' codec can't encode character '\\ud800': surrogates not allowed
"""
from src.text_utils import strip_surrogates, strip_surrogates_counted


def test_strip_surrogates_removes_lone_surrogates():
    assert strip_surrogates("正常\ud800文本") == "正常文本"
    assert strip_surrogates("\udc00") == ""
    assert strip_surrogates("\ud800\ud800x") == "x"


def test_strip_surrogates_keeps_valid_text_untouched():
    """别把正常文本改坏：emoji、非字符、NUL、换行都应当原样保留。"""
    for text in ["", "普通中文 ABC 123", "\U0001F600 表情", "\ufffe\uffff", "\x00", "\n\n", "😀"]:
        assert strip_surrogates(text) == text


def test_strip_surrogates_returns_unchanged_object_when_nothing_to_do():
    """没有代理字符时返回原对象，避免大文本被无谓复制一遍。"""
    text = "很长的正常文本" * 100
    assert strip_surrogates(text) is text


def test_strip_surrogates_logs_a_warning_when_it_removes_something(caplog):
    """不许静默丢字符 —— 剥掉了什么要能从日志里看见。"""
    with caplog.at_level("WARNING", logger="ai_rag"):
        strip_surrogates("坏\ud800字")
    assert "代理" in caplog.text


def test_strip_surrogates_counted_reports_how_many_were_removed():
    """上传要能告诉用户「这份文件有几个字符没解析出来」，所以带计数的版本。"""
    cleaned, count = strip_surrogates_counted("正常\ud800\udc00文本")
    assert cleaned == "正常文本"
    assert count == 2


def test_strip_surrogates_counted_returns_same_object_and_zero_when_clean():
    text = "很长的正常文本" * 100
    cleaned, count = strip_surrogates_counted(text)
    assert cleaned is text, "干净文本不该被复制一遍"
    assert count == 0
