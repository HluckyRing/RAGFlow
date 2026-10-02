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
