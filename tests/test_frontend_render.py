# -*- coding: utf-8 -*-
"""前端拒答标签的**渲染**测试（不只是「代码里有这段字符串」）。

背景（2026-11-08 待办③）：`tests/test_frontend.py` 那些静态守卫只能证明「index.html 里有这行」，
证明不了「渲染出来是什么样」。这个文件补上真渲染：

- 从 `src/templates/index.html` 里**抽出真实的函数源码**（`buildRefusedTag` / `showRefusedTag`），
- 用 Node 内置 `vm` 跑一遍，配一个极小的 DOM 替身（`document.createElement` + 节点树），
- 断言它建出来的节点的 class / 文本、以及「拒绝使用 innerHTML」。

不引 npm 依赖、不装 headless 浏览器：前端本来就是零依赖原生 JS，测试也不该给它加工具链。
本机没有 `node` 时整组用例 skip（CI 的 ubuntu runner 自带 node，会真跑）。
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "src" / "templates" / "index.html"


def _extract_function(source, name):
    """从 HTML 里的 JS 源码切出 `function name(...) { ... }` 整段（含嵌套花括号）。

    只做花括号配对 —— 这两个函数里唯一的 `{` `}` 来自模板字符串 `${searched}`，是配平的。
    切不出来就直接报错，绝不静默返回半截（半截会让断言变成假的通过）。
    """
    marker = f"function {name}("
    start = source.index(marker)
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"没能从 index.html 里切出完整函数 {name}")


HARNESS = r"""
const results = {};
// 极简 DOM 替身：只实现这两个函数用到的东西。
function createElement(tag) {
    const el = {
        tag: tag,
        className: '',
        children: [],
        _text: '',
        get textContent() { return this._text; },
        set textContent(value) { this._text = String(value); },
        get innerHTML() { return ''; },
        set innerHTML(_value) { throw new Error('拒答标签不许用 innerHTML'); },
        appendChild(child) { this.children.push(child); return child; },
        querySelector(selector) {
            const want = selector.replace(/^\./, '');
            for (const child of this.children) {
                if (String(child.className || '').split(/\s+/).includes(want)) return child;
            }
            return null;
        },
    };
    return el;
}
const document = { createElement };

__FUNCTIONS__

results.noCount = buildRefusedTag(0);
results.count2 = buildRefusedTag(2);
results.stringCount = buildRefusedTag('2');
results.undefinedCount = buildRefusedTag();
const host = createElement('div');
showRefusedTag(host, 3);
results.hostChildren = host.children.length;
results.hostTag = host.children[0];
showRefusedTag(host, 3);
results.hostChildrenAfterSecondCall = host.children.length;
process.stdout.write(JSON.stringify({
    noCount: { className: results.noCount.className, text: results.noCount.textContent },
    count2: { className: results.count2.className, text: results.count2.textContent },
    stringCount: results.stringCount.textContent,
    undefinedCount: results.undefinedCount.textContent,
    hostChildren: results.hostChildren,
    hostChildrenAfterSecondCall: results.hostChildrenAfterSecondCall,
    hostTagClassName: results.hostTag.className,
    hostTagText: results.hostTag.textContent,
}));
"""


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    node = shutil.which("node")
    if node is None:
        pytest.skip("本机没有 node，跳过前端渲染测试（CI 的 ubuntu runner 自带 node）")
    source = INDEX.read_text(encoding="utf-8")
    functions = "\n\n".join([_extract_function(source, "buildRefusedTag"),
                             _extract_function(source, "showRefusedTag")])
    script = tmp_path_factory.mktemp("render") / "harness.js"
    script.write_text(HARNESS.replace("__FUNCTIONS__", functions), encoding="utf-8")
    proc = subprocess.run([node, str(script)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert proc.returncode == 0, f"node 跑挂了：{proc.stderr[:800]}"
    return json.loads(proc.stdout)


BASE_TEXT = "⚠️ 知识库里没有能回答这个问题的内容"


def test_refused_tag_without_a_count(rendered):
    assert rendered["noCount"]["className"] == "refused-tag"
    assert rendered["noCount"]["text"] == BASE_TEXT, "没给计数时不要出现空括号"


def test_refused_tag_shows_the_file_count(rendered):
    assert rendered["count2"]["text"] == BASE_TEXT + "（已检索 2 个文件）"


def test_refused_tag_ignores_a_non_numeric_count(rendered):
    """形状不对就当没有（服务端契约变了自己也不该把脏数据渲染出来）。"""
    assert rendered["stringCount"] == BASE_TEXT
    assert rendered["undefinedCount"] == BASE_TEXT


def test_show_refused_tag_appends_once(rendered):
    assert rendered["hostChildren"] == 1
    assert rendered["hostChildrenAfterSecondCall"] == 1, "重复调用不许叠出第二个标签"
    assert rendered["hostTagClassName"] == "refused-tag"
    assert rendered["hostTagText"] == BASE_TEXT + "（已检索 3 个文件）"
