# -*- coding: utf-8 -*-
"""仓库卫生守卫（P2-18 / P2-20）。

这些不是运行时行为，而是「别把已经修掉的问题带回来」的断言：
- CI 必须存在且真的跑 pytest（P2-18 的 CI 部分）
- .gitignore 的取反规则不能是笔误，且 .env.example 必须保持不被忽略（P2-20）
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_exists_and_runs_pytest():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "P2-18 要求补 CI（原报告：无测试/CI）"
    text = workflow.read_text(encoding="utf-8")
    assert "pytest" in text, "CI 必须真的跑测试"
    assert "requirements-dev.txt" in text, "CI 要按仓库声明的 dev 依赖安装"


def test_gitignore_has_no_negation_typo():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!?" not in text, "`!?.env.example` 是笔误，应为 `!.env.example`"
    assert "!.env.example" in text


def test_env_example_is_not_ignored():
    """功能断言：git 确实不会忽略 .env.example，模板文件必须留在版本库里。"""
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("需要 git 仓库环境")
    proc = subprocess.run([git, "check-ignore", "--quiet", str(ROOT / ".env.example")],
                          cwd=ROOT, capture_output=True)
    # git check-ignore：退出码 0 = 被忽略，1 = 未被忽略
    assert proc.returncode != 0, ".env.example 不能被 .gitignore 忽略"


# 用户数据目录：**必须**被忽略。这条是功能断言（问 git 本人），不是「.gitignore 里有这行字符串」。
# 为什么钉它（2026-11-15）：`.agents/` 里存的是用户上传文件的**解析文本**（.agents/ml-*/、
# .agents/real-*/、cc-corpus/）、模型回答全文与各轮原始 JSON。它以前只靠「约定不入库」
# —— 未被跟踪但也没被忽略 —— 一次 `git add -A` 就会把这些推进公开仓库。
USER_DATA_PATHS = [
    "uploads/abc.pdf",
    "uploads/36e93a78c0dcc1e11b0afc1dd13f21d9/29e53011fa75/abc.pdf",
    "state/36e93a78c0dcc1e11b0afc1dd13f21d9.json",
    "data/某课件.pdf",
    "chroma_db/chroma.sqlite3",
    ".agents/ml-29e53011/01-第1章.md",
    ".agents/intra-on39.json",
    "MEMORY.md",
]


@pytest.mark.parametrize("rel", USER_DATA_PATHS)
def test_user_data_paths_are_gitignored(rel):
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("需要 git 仓库环境")
    proc = subprocess.run([git, "check-ignore", "--quiet", rel],
                          cwd=ROOT, capture_output=True)
    assert proc.returncode == 0, f"{rel} 必须被 .gitignore 忽略（用户数据/评测原始数据不进公开仓库）"


def test_agents_workspace_is_not_tracked():
    """`.agents/` 里是用户语料的解析文本 —— 除了要忽略，还不许已经被跟踪（忽略对已跟踪文件无效）。"""
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("需要 git 仓库环境")
    proc = subprocess.run([git, "ls-files", ".agents"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.stdout.strip() == "", f".agents/ 下不该有已跟踪文件：{proc.stdout[:200]}"


# ── README / .env.example 与代码不许漂（2026-11-19）──
# 起因：README 里「750 passed」「1998 行基线」「这个服务没有鉴权」三处全部过期 —— 文档漂了没人发现。
# 这两条守卫只管**能机械核对的**部分：变量名必须真实存在。默认值/行数/测试条数这类数字仍然要人肉
# 更新（改默认值时应同步 README 的「📊 实测数据」与 .env.example）。

# 只认「NAME=…」这种**赋值形态**（.env 文件里就是这么写的），不碰正文里的大写缩写（MRR / RRF / PPT）
_ASSIGN_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=")
_BACKTICK_RE = re.compile(r"`([A-Z][A-Z0-9_]{2,})`")


def _source_corpus():
    """`src/` 下所有 Python 源码拼起来 —— 变量可能在 config.py 之外被读（例如 STATE_DIR 在 state.py）。"""
    parts = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_env_example_only_documents_real_variables():
    """.env.example 里写的每个变量，`src/` 里必须真的有人读它。

    这条防的是「删了代码、模板还留着」：用户照着 `.env.example` 配了半天，程序根本不读。
    """
    source = _source_corpus()
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(_ASSIGN_RE.findall(text))
    missing = sorted(name for name in documented if f'"{name}"' not in source)
    assert not missing, f".env.example 里这些变量在 src/ 里没人读：{missing}"


def test_readme_only_mentions_real_variables():
    """README 里提到的变量名，必须能在 `src/` 里找到。

    这条防的是 README 把开关名写错（例如把 `SCOPE_FILTER` 写成 `SCOPE`）—— 用户照抄就静默失效。
    """
    source = _source_corpus()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    mentioned = set(_ASSIGN_RE.findall(readme)) | set(_BACKTICK_RE.findall(readme))
    unknown = sorted(name for name in mentioned if f'"{name}"' not in source)
    assert not unknown, f"README 里这些变量在 src/ 里找不到：{unknown}"


# ── README 的机械部分由脚本生成，不许漂（2026-11-19）──
# `scripts/sync_readme.py` 负责「能从仓库算出来」的东西：测试条数、基线行数、Python 版本、
# 项目结构树、支持格式表。CI 里跑 `--check`（含 test-count）；这里跑便宜版（跳过要收集测试的
# test-count），免得「在测试里再跑一遍测试」。
def test_readme_mechanical_blocks_are_in_sync():
    from scripts import sync_readme

    assert sync_readme.main(["--check", "--no-test-count"]) == 0, (
        "README 的自动生成块与仓库不一致：跑 `python scripts/sync_readme.py --write` 后看 git diff"
    )
