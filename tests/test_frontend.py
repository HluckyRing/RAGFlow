# -*- coding: utf-8 -*-
"""前端源码静态守卫（P1-15）。

只做源码级断言，不引入 JS 测试依赖 —— 项目前端是零依赖原生 JS，pytest 之外
没有 JS 测试基建。开发期另用 node 真正执行过 renderSidebar（见提交说明）。

守的是这一类回归：**动态数据（对话名/文件名）绝不能拼进内联事件处理器**。
内联处理器是 HTML 属性，浏览器会当代码解析 —— 只要名字里带一个单引号，
就能闭合字符串执行任意 JS（改造前 index.html:260 的 confirmDelete('...','名称')）。
"""
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "src" / "templates" / "index.html"


def _html():
    return INDEX.read_text(encoding="utf-8")


def _sidebar_source():
    html = _html()
    start = html.index("function renderSidebar")
    end = html.index("function renderChat")
    return html[start:end]


def test_esc_escapes_single_quote():
    """esc() 之前漏了单引号；只要还有任何地方把 esc() 的结果放进单引号里就再次可注入。"""
    line = re.search(r"function esc\(s\).*", _html()).group(0)
    assert "&#39;" in line, f"esc() 必须转义单引号: {line}"


def test_no_template_interpolation_in_inline_handlers():
    """内联处理器里出现 ${...} 就是把动态数据当代码，必红。"""
    bad = re.findall(r'\son\w+\s*=\s*"[^"]*\$\{', _html())
    assert not bad, f"内联事件处理器里有模板插值：{bad[:3]}"


def test_no_escaped_dynamic_data_inside_inline_handlers():
    """esc() 只挡 HTML 元字符，挡不住 JS 字符串闭合，所以动态数据一律不许进内联处理器。"""
    bad = re.findall(r'\son\w+\s*=\s*"[^"]*esc\(', _html())
    assert not bad, f"内联事件处理器里拼了动态数据：{bad[:3]}"


def test_sidebar_uses_dataset_and_addEventListener():
    src = _sidebar_source()
    assert "dataset" in src, "对话项要用 dataset 传 id，而不是拼进 onclick"
    assert "addEventListener" in src, "事件要用 addEventListener 绑定"
    assert "onclick=" not in src, "renderSidebar 里不该再有内联 onclick"
    assert "textContent" in src, "对话名要用 textContent 写入，而不是拼 HTML"


def test_file_input_allows_multiple():
    tag = re.search(r'<input[^>]*id="fileInput"[^>]*>', _html()).group(0)
    assert "multiple" in tag, "file input 必须允许一次选多个文件"


def test_upload_sends_plural_files_field():
    """前端必须用复数 files 字段一次性提交，避免逐文件触发全量重建索引。"""
    html = _html()
    assert "form.append('files'" in html or 'form.append("files"' in html


def test_file_list_deletion_is_wired():
    """单文件删除要绑在 dataset 上并调用后端 DELETE，不能拼内联处理器。"""
    html = _html()
    assert "dataset.fileName" in html
    assert re.search(r"method:\s*'DELETE'", html), "缺少 DELETE 调用"
    assert "/files/" in html
