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


def _source_between(start_marker, end_marker):
    html = _html()
    start = html.index(start_marker)
    end = html.index(end_marker)
    assert start < end, f"{start_marker} 应在 {end_marker} 之前"
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


# ── 2026-10-03 前端改版新增守卫 ──
# 约定不变：零依赖、动态数据不进内联处理器（上面三条通用 regex 已覆盖全文件，
# 这里只补新功能的「接线是否真的接上了」）。


def test_theme_follows_system_and_remembers_manual_choice():
    html = _html()
    assert "prefers-color-scheme" in html, "首次访问要跟随系统"
    assert "ragflow_theme" in html, "手动切换过就要记住，下次不再跟系统"
    assert "data-theme" in html, "主题要落在 html[data-theme] 上供 CSS 覆盖"
    assert 'id="themeBtn"' in html, "要有手动切换入口"


def test_responsive_drawer_has_breakpoint_and_backdrop():
    html = _html()
    assert "820px" in html, "移动端断点"
    assert 'id="sidebarBackdrop"' in html, "抽屉打开时要有遮罩可点关闭"
    assert "translateX" in html, "移动端侧边栏用位移收起，不是压宽度"


def test_stop_generation_uses_abort_controller():
    html = _html()
    assert "AbortController" in html
    assert ".abort()" in html
    assert "signal" in html, "fetch 要带 signal，否则中断不了"


def test_regenerate_and_edit_send_truncate_to():
    """前端必须把 truncate_to 发给服务端，否则服务端只会追加，历史重复。"""
    html = _html()
    assert "truncate_to" in html
    assert "regenerateMsg" in html
    assert "startEditMsg" in html


def test_export_conversation_as_markdown():
    html = _html()
    assert "new Blob" in html
    assert "text/markdown" in html
    assert "download" in html


def test_conversation_search_and_shortcut():
    html = _html()
    assert 'id="convSearch"' in html
    assert "ctrlKey" in html or "metaKey" in html, "Ctrl/Cmd+K 聚焦搜索"


def test_markdown_renderer_supports_tables_lists_quotes():
    html = _html()
    assert "thead" in html, "表格要渲染成真表格"
    assert "<li>" in html, "列表要渲染成真列表"
    assert "blockquote" in html, "引用块要渲染"
    assert "inlineMd" in html, "行内规则要复用，别每处各写一套"


def test_message_actions_are_bound_with_listener():
    html = _html()
    assert "mact" in html, "消息级操作按钮"
    assert "copyText" in html
    assert "fmtTime" in html, "时间戳格式化"


def test_quick_prompts_offered_when_knowledge_base_is_ready():
    html = _html()
    assert "QUICK_PROMPTS" in html, "知识库就绪但没有消息时给几个示例问题"
    assert "renderEmptyChat" in html


def test_switching_conversation_clears_edit_state():
    """编辑态绑在「某个对话的第 N 条」上。

    换对话不清掉的话，下一次发送会把那个下标当成 truncate_to 用到新对话上，
    把新对话的历史截掉一部分。
    """
    html = _html()
    start = html.index("async function switchConv")
    end = html.index("function startEdit(")
    assert "cancelEditMsg" in html[start:end]


def test_file_input_accepts_pptx():
    """PPT 支持链上，前端 accept 是最容易漏的一环：漏了用户在选择器里根本选不到。"""
    tag = re.search(r'<input[^>]*id="fileInput"[^>]*>', _html()).group(0)
    assert ".pptx" in tag, "file input 要接受 .pptx"
    assert ".pptm" in tag, "file input 要接受 .pptm"


def test_upload_surfaces_stripped_chars_to_user():
    """后端把「剥离了几个脏字符」放在 stripped_chars 里，前端必须读它并提示用户，
    否则「解析残缺」只写在后端日志里，用户完全无从得知。"""
    html = _html()
    assert "stripped_chars" in html, "前端要读后端返回的 stripped_chars"
    assert "无法解析" in html, "要用人话告诉用户发生了什么"


def test_toasts_stack_so_two_messages_do_not_overlap():
    """上传成功与「字符无法解析」会同时弹出，必须有容器让它们纵向排开。"""
    html = _html()
    assert "toast-wrap" in html
    assert "flex-direction:column" in html.replace(" ", "")


# ── 「查看文件」列表的刷新守卫 ──
# 两个真实 bug：①上传解析期间点开「查看文件」，上传成功后列表停在旧内容（没有新
# 上传的文件）；②在 A 知识库点开「查看文件」，切到 B 知识库后列表仍显示 A 的文件。
# 共同根因：只有 toggleFileList 会 renderFileList，showChat（上传成功 / 切对话都会
# 走它）只更新角标和 fileBar，不管已经展开的列表。


def test_open_file_list_follows_active_conversation():
    """已展开的文件列表必须用当前对话的文件重绘，且只重绘「展开」状态的列表。"""
    src = _source_between("function showChat", "function renderSidebar")
    assert "$('fileList')" in src, "showChat 要检查文件列表当前是否展开"
    assert "hidden" in src, "只在列表展开时重绘，别把用户收起的列表又弹开"
    assert "renderFileList()" in src, "展开的列表要用当前对话的文件重绘"


def test_upload_success_and_conversation_switch_refresh_show_chat():
    """两条触发路径都必须经过 refreshView→showChat，才能让上面那条重绘生效。"""
    upload = _source_between("async function uploadFiles", "function renderFileList")
    assert "refreshView()" in upload, "上传成功要刷新视图"
    switch = _source_between("async function switchConv", "function startEdit(")
    assert "refreshView()" in switch, "切换对话要刷新视图"


# ── 文件预览（右侧抽屉）守卫 ──
# 预览读的是后端 files[].file_text（抽取正文，服务端不保存原件）。约定不变：
# 动态数据只走 textContent，绝不 innerHTML；预览本身只读，不能碰索引。


def test_file_list_rows_have_preview_button():
    src = _source_between("function renderFileList", "const PREVIEW_MAX_CHARS")
    assert "openFilePreview" in src, "文件行要绑定预览入口"
    assert "预览" in src, "要有可见的「预览」按钮"


def test_file_preview_drawer_markup_exists():
    html = _html()
    for marker in ('id="filePreview"', 'id="filePreviewBody"',
                   'id="filePreviewClose"', 'id="filePreviewMask"'):
        assert marker in html, f"缺少预览抽屉元素 {marker}"


def test_file_preview_fetches_text_and_writes_with_textContent():
    src = _source_between("const PREVIEW_MAX_CHARS", "function confirmDeleteFile")
    assert "/files/" in src, "预览要调用单文件接口"
    assert "file_text" in src, "要读取后端返回的正文"
    assert "textContent" in src, "正文必须用 textContent 写入"
    assert "innerHTML" not in src, "预览正文不得走 innerHTML，否则文件名/正文可注入"
    assert "filePreviewBody" in src


def test_file_preview_truncates_huge_text_with_notice():
    src = _source_between("const PREVIEW_MAX_CHARS", "function confirmDeleteFile")
    assert "PREVIEW_MAX_CHARS" in src
    assert "200000" in src, "阈值写死 20 万字符，便于核对"
    assert "仅显示" in src, "截断后要提示用户只看到一部分"


def test_escape_closes_file_preview():
    src = _source_between("function onGlobalKey", "function onDrop")
    assert "closeFilePreview" in src, "Esc 要能关掉预览抽屉"


# ── 原文件预览（本地 vendor 渲染库）守卫 ──
# 用户要求「原文件原预览」：原件要持久化，PDF/文本直接看原件，DOCX/XLSX 用本地
# 内置的渲染库（非 CDN），PPTX 只下载。动态数据仍然只走 textContent。

VENDOR_LIBS = ('/vendor/jszip.min.js', '/vendor/docx-preview.min.js', '/vendor/xlsx.core.min.js')


def test_vendor_renderers_are_local_not_cdn():
    html = _html()
    for lib in VENDOR_LIBS:
        assert lib in html, f"缺少本地 vendor 库 {lib}"
    assert not re.search(r'<script[^>]+src="https?://', html), "不能再引入外网 CDN 依赖"


def test_preview_drawer_has_tabs_frame_and_doc_stage():
    html = _html()
    for marker in ('id="filePreviewTabs"', 'id="fpTabOriginal"', 'id="fpTabText"',
                   'id="filePreviewFrame"', 'id="filePreviewDoc"'):
        assert marker in html, f"缺少原件预览元素 {marker}"


def test_original_preview_maps_formats_to_renderers():
    src = _source_between("function previewKind", "function closeFilePreview")
    assert "renderAsync" in src, "DOCX 要交给 docx-preview 渲染"
    assert "XLSX.read" in src, "XLSX 要用 SheetJS 解析"
    assert "sheet_to_json" in src, "XLSX 取出行数据后自己建表"
    assert "sheet_to_html" not in src, "不能把库生成的 HTML 直接塞进 DOM"
    assert "下载原件" in src, "PPTX/渲染失败要有下载兜底"
    assert "filePreviewFrame" in src, "PDF 用 iframe 看原件"
    assert "原件未保存" in src, "升级前的旧文件要标注没有原件"


def test_xlsx_preview_builds_table_with_textcontent_only():
    src = _source_between("function renderXlsx", "async function renderOriginal")
    assert "createElement" in src, "要自己建 table"
    assert "textContent" in src, "单元格文本必须用 textContent 写入"
    assert "innerHTML" not in src, "单元格是用户文件内容，绝不能当 HTML 解析"


def test_preview_open_reads_original_metadata():
    src = _source_between("async function openFilePreview", "function closeFilePreview")
    assert "has_original" in src, "抽屉要根据后端返回的 has_original 选视图"
    assert "previewKind" in src, "按扩展名分派渲染器"
