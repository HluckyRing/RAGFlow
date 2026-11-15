# -*- coding: utf-8 -*-
"""仓库卫生守卫（P2-18 / P2-20）。

这些不是运行时行为，而是「别把已经修掉的问题带回来」的断言：
- CI 必须存在且真的跑 pytest（P2-18 的 CI 部分）
- .gitignore 的取反规则不能是笔误，且 .env.example 必须保持不被忽略（P2-20）
"""
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
