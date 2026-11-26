#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""README 里「能从仓库算出来」的部分：一个脚本生成 + `--check` 给 CI 当闸门。

为什么要有它（2026-11-19 第十九轮）
-----------------------------------
一次检查发现 README 里同时躺着四个过期/互相矛盾的数字：`750 passed`（实际 767）、
基线文档 `1998 行` / `1600+ 行` / `约 1650 行`（实际 2691）。这些**全都能从仓库算出来**，
却一直被手写 —— 每次加测试、改文档就再漂一次。所以：

- **能算的**（测试条数、基线行数、Python 版本、文件清单、支持格式）交给这个脚本；
- **算不出来的**（为什么这么做、哪条路被证伪了、实测数字 11/14 → 13/14）仍然人写，
  脚本一个字都不碰。

做法：HTML 注释当标记，脚本只重写标记**之间**的内容
--------------------------------------------------
行内标记（散文原样保留，渲染时看不见）：

    `tests/` 里是 <!-- AUTO:test-count -->767<!-- /AUTO:test-count --> 条测试

块标记（内容会带换行）：

    <!-- AUTO:structure -->
    ```text
    ...
    ```
    <!-- /AUTO:structure -->

可用的标记名：`test-count` / `docs-lines` / `python-min`（行内）与 `structure` / `formats`（块）。

`structure` 的说明文字来自本文件的 `DESCRIPTIONS`，**不是**模块 docstring —— 一半模块
（`config.py` / `llm.py` / `retrieval.py` / `server.py` …）根本没有 docstring。新增文件却没补
说明时脚本**直接报错退出**：逼着人写一句，而不是让 README 悄悄漏掉它。

用法
----
    python scripts/sync_readme.py --write                  # 重写标记块；写完 git diff 看一眼再提交
    python scripts/sync_readme.py --check                  # 只比对，不一致退出码 1（CI 用）
    python scripts/sync_readme.py --check --no-test-count  # 跳过 pytest 收集（pytest 里的守卫用）

`test-count` 要跑一次 `pytest --collect-only`（约 5–10 s），所以只在 CI 与 `--write` 时算；
pytest 套件里那条守卫带 `--no-test-count`，免得「在测试里再跑一遍测试」。

退出码：0 一致 / 已写入；1 `--check` 发现漂移；2 输入不合法（标记缺失、文件没写说明等）。
"""
import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                # 从 scripts/ 直接跑时也能 import src.*
    sys.path.insert(0, str(ROOT))
README = ROOT / "README.md"
BASELINE = ROOT / "docs" / "retrieval-baseline-2026-10-06.md"
PYPROJECT = ROOT / "pyproject.toml"

# ── src/ 与 scripts/ 的说明（README 项目结构树用）──
# 只写一句、说清「这个文件负责什么」。**新增文件必须补一行**，否则脚本报错。
DESCRIPTIONS = {
    "src/server.py": "FastAPI 应用定义（用 python server.py 启动）",
    "src/state.py": "会话状态持久化（按 session_id 隔离）",
    "src/config.py": "环境配置、日志、OpenAI 客户端",
    "src/loaders.py": "多格式文件加载器",
    "src/pdf_ingestion.py": "文本切片",
    "src/text_utils.py": "疑问词剥离 + 2/3-gram 抽词（检索与指代消解共用）",
    "src/retrieval.py": "向量检索 + 关键词回退 + HyDE + 双路融合 + 候选池",
    "src/scope.py": "范围词 → 文件过滤器（\"结合第2章课件，…\" → where=file_name）",
    "src/coverage.py": "覆盖取样（跨文件 / 文件内跨节）+ 覆盖账与【覆盖说明】",
    "src/decompose.py": "问句拆解（廉价闸门 + JSON 解析 + 任一故障退回原问句）",
    "src/hits_merge.py": "多路检索结果的合并口径（线上与评测共用同一份实现）",
    "src/answerability.py": "可答性判断（严格模式；默认由生成侧自己拒答）",
    "src/rerank.py": "跨编码器精排（默认关：CPU 3.9 s/问）",
    "src/security.py": "对外监听 fail-closed（无 AUTH_TOKEN 拒绝启动）+ /api/chat 限流",
    "src/llm.py": "指代消解 + 多轮对话 + 每问分路编排 + 流式答案生成",
    "src/prompts.py": "Prompt 模板（HyDE / 拆问 / 可答性 / QA）",
    "src/gates/__init__.py": "闸门措辞包的选择入口（GATE_PACK 决定用哪一套）",
    "src/gates/zh.py": "闸门措辞表：中文包（默认）",
    "src/__init__.py": "包标记",
    "scripts/eval_retrieval.py": "合成语料检索评测（recall/MRR/归属，默认不花额度）",
    "scripts/eval_real_corpus.py": "真实语料基线（只读、只记数字：切窗 + 距离带）",
    "scripts/eval_compound_retrieval.py": "复合问句检索覆盖 + 分路对照（含模型拆问臂）",
    "scripts/eval_decomposition.py": "拆问本身的结构测量（闸门/延迟/条数/误拆）",
    "scripts/eval_scope_filter.py": "范围词 / 候选池四臂对照（事实核离线校验）",
    "scripts/measure_inline_abstention.py": "拒答与部分可答的端到端测量",
    "scripts/rejudge_abstention.py": "用存下来的回答离线重算判定口径",
    "scripts/analyze_real_questions.py": "盘真实提问（只读 state/）：闸门触发率、复合问占比",
    "scripts/reindex_all.py": "按当前切片重建所有对话索引（默认干跑）",
    "scripts/cleanup_orphan_collections.py": "孤儿向量集合清理（默认干跑）",
    "scripts/check_release_notes.py": "发版守卫：标签必须有对应的 docs/release-notes-<版本>.md（缺了让工作流红）",
    "scripts/sync_readme.py": "README 的机械部分生成 / 校验（本文件）",
    "scripts/update_deploy.py": "Docker 部署更新：git pull → 重建镜像 → 等健康检查",
    "scripts/backup_data.py": "Docker 数据卷备份（停容器打包 + 保留最近 N 份）",
}

# 结构树里除 src/ scripts/ 之外的部分（都是稳定条目，手写在脚本里，不散落到 README）
STRUCTURE_TAIL = """├── tests/                # pytest 测试（pytest -q 全绿才算完）
├── docs/
│   ├── retrieval-baseline-2026-10-06.md # 逐条实测基线，所有默认值的依据
│   ├── session-log/      # 每一轮开发的复盘（数字、决策、踩过的坑）
│   └── code-review-2026-10-01.md       # 2026-10-01 代码审查原始清单
├── .github/workflows/    # CI（pytest + README 同步校验）与 Release
├── state/                # 会话状态，每个会话一个 JSON 文件（gitignore）
├── uploads/              # 上传原件，按会话/对话分目录存放（gitignore）
├── legacy/               # 历史版本归档
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml"""

# 支持格式表：显示名 / 扩展名 / 依赖。必须与 loaders.SUPPORTED_EXTENSIONS **完全一致**，
# 多一个少一个都会报错 —— 加格式时逼着人来这里写清依赖。
FORMAT_ROWS = (
    ("PDF", (".pdf",), "pypdf"),
    ("Word", (".docx",), "python-docx"),
    ("Excel", (".xlsx",), "openpyxl"),
    ("PPT", (".pptx", ".pptm"), "python-pptx"),
    ("CSV", (".csv",), "标准库"),
    ("文本", (".txt", ".md"), "无"),
)

BLOCK_NAMES = ("structure", "formats")
INLINE_NAMES = ("test-count", "docs-lines", "python-min")


def marker(name):
    """标记对：`<!-- AUTO:name -->…<!-- /AUTO:name -->`。"""
    return re.compile(r"(<!-- AUTO:%s -->)(.*?)(<!-- /AUTO:%s -->)" % (name, name), re.S)


# ── 各个值怎么算 ──

def count_tests():
    """跑 `pytest --collect-only -q` 数测试条数（约 5–10 s，故只在 CI 与 --write 时算）。"""
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    text = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"(\d+)\s+tests?\s+collected", text)
    if not match:
        raise SystemExit(f"数不出测试条数（pytest 输出里没有 'N tests collected'）：{text[-300:]}")
    return int(match.group(1))


def docs_lines():
    return len(BASELINE.read_text(encoding="utf-8").splitlines())


def python_min():
    match = re.search(r'requires-python\s*=\s*">=\s*([0-9.]+)"',
                      PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit('pyproject.toml 里没有 requires-python = ">=X.Y"')
    return match.group(1)


def _tracked_files(pattern_dir, suffixes=(".py",)):
    """`src/` 或 `scripts/` 下真正该写进 README 的文件（跳过 __pycache__）。"""
    base = ROOT / pattern_dir
    found = []
    for path in sorted(base.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix not in suffixes:
            continue
        found.append(path.relative_to(ROOT).as_posix())
    return found


def _describe(rel_path):
    """取一句说明：DESCRIPTIONS 优先，其次模块 docstring 首行；都没有就报错。"""
    if rel_path in DESCRIPTIONS:
        return DESCRIPTIONS[rel_path]
    try:
        doc = ast.get_docstring(ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))) or ""
    except SyntaxError:
        doc = ""
    first = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    if not first:
        raise SystemExit(f"{rel_path} 没有说明，也没 docstring：请在 scripts/sync_readme.py 的 "
                         f"DESCRIPTIONS 里补一句话（README 的项目结构树靠它生成）")
    return first.rstrip("。")


def structure_block():
    """整棵项目结构树（src/ 与 scripts/ 的文件清单由目录实际内容生成）。"""
    src_files = _tracked_files("src")
    script_files = _tracked_files("scripts")

    missing = sorted(set(DESCRIPTIONS) - set(src_files) - set(script_files))
    if missing:
        raise SystemExit(f"DESCRIPTIONS 里这些文件已经不存在了，删掉对应条目：{missing}")

    top_src = [path for path in src_files
               if path.count("/") == 1 and not path.endswith("__init__.py")]
    gates = [path for path in src_files if path.startswith("src/gates/")]

    lines = ["RAGFlow/", "├── src/"]
    width = max(len(Path(path).name) for path in top_src) + 4
    for path in top_src:
        lines.append("│   ├── " + f"{Path(path).name:<{width}s}# {_describe(path)}")
    lines.append("│   ├── " + f"{'gates/':<{width}s}# 闸门措辞表（GATE_PACK 可插拔，失效可见）")
    gate_width = max(len(Path(path).name) for path in gates) + 4
    for index, path in enumerate(gates):
        branch = "└──" if index == len(gates) - 1 else "├──"
        lines.append("│   │   " + branch + " " + f"{Path(path).name:<{gate_width}s}"
                     f"# {_describe(path)}")
    lines.append("│   └── " + f"{'templates/':<{width}s}"
                 "# 前端界面（vendor/ 为本地内置的 Office 预览库）")
    lines.append("├── scripts/")
    script_width = max(len(Path(path).name) for path in script_files) + 4
    for index, path in enumerate(script_files):
        branch = "└──" if index == len(script_files) - 1 else "├──"
        lines.append("│   " + branch + " " + f"{Path(path).name:<{script_width}s}"
                     f"# {_describe(path)}")
    lines.append(STRUCTURE_TAIL)
    return "```text\n" + "\n".join(lines) + "\n```"


def formats_block():
    """支持格式表：与 `loaders.SUPPORTED_EXTENSIONS` 必须完全一致（多退少补都会报错）。"""
    from src.loaders import SUPPORTED_EXTENSIONS

    declared = {ext for _label, exts, _dep in FORMAT_ROWS for ext in exts}
    supported = set(SUPPORTED_EXTENSIONS)
    if declared != supported:
        missing = sorted(supported - declared)
        extra = sorted(declared - supported)
        raise SystemExit("支持格式表与 loaders.SUPPORTED_EXTENSIONS 不一致："
                         f"代码支持但表里没有 {missing}；表里有但代码不支持 {extra}"
                         "（请在 scripts/sync_readme.py 的 FORMAT_ROWS 里对齐）")

    lines = ["| 格式 | 扩展名 | 依赖 |", "|------|--------|------|"]
    for label, exts, dep in FORMAT_ROWS:
        lines.append(f"| {label} | {' '.join('`' + ext + '`' for ext in exts)} | {dep} |")
    return "\n".join(lines)


def render(name):
    """一个标记的**新内容**（行内标记返回单行，块标记返回带换行的整段）。"""
    if name == "test-count":
        return str(count_tests())
    if name == "docs-lines":
        return str(docs_lines())
    if name == "python-min":
        return python_min()
    if name == "structure":
        return "\n" + structure_block() + "\n"
    if name == "formats":
        return "\n" + formats_block() + "\n"
    raise SystemExit(f"不认识的标记名：{name}")


def apply_blocks(text, only=None, quiet=False):
    """把标记之间的内容换成生成内容，返回 `(新文本, 变了的标记名列表)`。

    `only` 给了就只处理那几个（`--no-test-count` 用它跳过要跑 pytest 的那个）。
    """
    names = [name for name in INLINE_NAMES + BLOCK_NAMES if only is None or name in only]
    changed = []
    for name in names:
        pattern = marker(name)
        matches = pattern.findall(text)
        if not matches:
            raise SystemExit(f"README 里找不到标记 <!-- AUTO:{name} --> … <!-- /AUTO:{name} -->")
        value = render(name)
        if any(current != value for _open, current, _close in matches):
            changed.append(name)
        text = pattern.sub(lambda m, v=value: m.group(1) + v + m.group(3), text)
    if changed and not quiet:
        print("会改动的标记：" + "、".join(changed))
    return text, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成 / 校验 README 的机械部分")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="按当前仓库状态重写 README 的标记块")
    group.add_argument("--check", action="store_true", help="只比对，不一致退出码 1（CI 用）")
    parser.add_argument("--no-test-count", action="store_true",
                        help="跳过 test-count（要跑 pytest 收集；pytest 里的守卫用这个）")
    args = parser.parse_args(argv)

    only = [name for name in INLINE_NAMES + BLOCK_NAMES
            if not (args.no_test_count and name == "test-count")]
    text = README.read_text(encoding="utf-8")
    new_text, changed = apply_blocks(text, only=only)

    if args.write:
        if changed:
            README.write_text(new_text, encoding="utf-8")
            print(f"README.md 已更新（{len(changed)} 处）：" + "、".join(changed))
            print("记得 git diff 看一眼再提交 —— 块外的散文脚本不会碰。")
        else:
            print("README.md 已是最新，没有改动。")
        return 0

    if changed:
        print("README 与仓库实际不一致（跑 python scripts/sync_readme.py --write 修）",
              file=sys.stderr)
        return 1
    print("README 与仓库实际一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
