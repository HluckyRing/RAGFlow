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
    """**真实读取配置的地方**全拼起来：`src/` 的 Python 源码 + compose + Dockerfile。

    变量可能在 config.py 之外被读（`STATE_DIR` 在 state.py、`CHUNK_SIZE` 在 pdf_ingestion.py），
    2026-11-25 起 `HOST_PORT` 这种**只给 Docker 用**的变量也要算数 —— 它由 compose 的
    `${HOST_PORT:-8080}` 读取，Python 侧根本不碰。所以语料要把 compose / Dockerfile 一起算上，
    否则守卫会逼着人去 `src/` 里加一行假读取。
    """
    parts = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        parts.append(path.read_text(encoding="utf-8"))
    for name in ("docker-compose.yml", "Dockerfile"):
        candidate = ROOT / name
        if candidate.exists():
            parts.append(candidate.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_env_example_only_documents_real_variables():
    """.env.example 里写的每个变量，必须真的有人读它（`src/` 或 compose / Dockerfile）。

    这条防的是「删了代码、模板还留着」：用户照着 `.env.example` 配了半天，程序根本不读。
    """
    source = _source_corpus()
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(_ASSIGN_RE.findall(text))
    missing = sorted(name for name in documented if f'"{name}"' not in source and name not in source)
    assert not missing, f".env.example 里这些变量没人读（src/ 或 compose 里都找不到）：{missing}"


def test_readme_only_mentions_real_variables():
    """README 里提到的变量名，必须能在 `src/` 或 compose / Dockerfile 里找到。

    这条防的是 README 把开关名写错（例如把 `SCOPE_FILTER` 写成 `SCOPE`）—— 用户照抄就静默失效。
    """
    source = _source_corpus()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    mentioned = set(_ASSIGN_RE.findall(readme)) | set(_BACKTICK_RE.findall(readme))
    unknown = sorted(name for name in mentioned if f'"{name}"' not in source and name not in source)
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

# ── 发版正文来自仓库里的手写 notes，不许再发空正文（2026-11-23）──
#
# 病根：`--generate-notes` 在「直推 main、没有 PR」的仓库里抓不到东西，v1.1.0 / v1.2.0 /
# v1.2.1 三次发出来的正文都只有一行 Full Changelog，只能事后 `gh release edit --notes-file` 补。
# 做法：把「读哪份 notes」搬进 `scripts/check_release_notes.py`（可被 pytest 直接测），
# 工作流只负责调用它 + `gh release create --notes-file`。

def _git_tracked_workflow():
    raw = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    # 只看生效的行：注释里可以解释「为什么不用自动生成」，那不算用了它
    return "\n".join(line for line in raw.splitlines() if not line.strip().startswith("#"))


def test_publish_workflow_takes_release_notes_from_the_repo():
    text = _git_tracked_workflow()
    assert "scripts/check_release_notes.py" in text, "发版前必须过 notes 守卫脚本"
    assert "--notes-file" in text, "发版正文必须来自仓库里的手写 notes"
    assert "--generate-notes" not in text, \
        "直推 main 的仓库里 --generate-notes 只会产出空正文，别再退回去"


def test_release_notes_guard_accepts_the_current_version():
    import scripts.check_release_notes as guard

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"', pyproject, re.M)
    assert match, "pyproject.toml 里找不到 version"
    ok, message, path = guard.check(f"v{match.group(1)}")
    assert ok, message
    assert path.name == f"release-notes-{match.group(1)}.md"


def test_release_notes_guard_rejects_a_missing_file(tmp_path):
    """缺 notes 就必须红：宁可停在这一步，也不要再发一个空正文的 Release。"""
    import scripts.check_release_notes as guard

    ok, message, path = guard.check("v9.9.9", root=tmp_path)
    assert not ok and "release-notes-9.9.9.md" in message


def test_release_notes_guard_rejects_a_too_short_file(tmp_path):
    import scripts.check_release_notes as guard

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "release-notes-9.9.9.md").write_text("# v9.9.9\n太短\n", encoding="utf-8")
    ok, message, _path = guard.check("v9.9.9", root=tmp_path)
    assert not ok and "太短" in message or "至少" in message


def test_release_notes_guard_writes_notes_env(tmp_path):
    """通过时把 NOTES 写进 $GITHUB_ENV，后续 gh release create 才拿得到路径。"""
    import scripts.check_release_notes as guard

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "release-notes-1.2.1.md").write_text(
        "# v1.2.1\n" + "正文" * 300, encoding="utf-8")
    env_file = tmp_path / "github_env"
    code = guard.main(["--tag", "v1.2.1", "--root", str(tmp_path),
                       "--github-env", str(env_file)])
    assert code == 0
    assert env_file.read_text(encoding="utf-8").strip() == "NOTES=docs/release-notes-1.2.1.md"


def test_release_notes_guard_exits_nonzero_when_missing(tmp_path):
    import scripts.check_release_notes as guard

    assert guard.main(["--tag", "v9.9.9", "--root", str(tmp_path)]) == 1


# ── Docker 交付面（2026-11-25）──
# 这组守卫不是「文件存在就算过」，而是钉住几个**会出事**的性质：
# ① 用户数据（`.env` / state / uploads / chroma_db）绝对不许烤进镜像；
# ② 容器必然对外监听，所以启动闸门（绑非回环必须有 AUTH_TOKEN）必须在镜像里照旧生效；
# ③ 数据必须挂 volume，否则 `docker rm` 等于删库。

def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_keeps_the_fail_closed_listen_address():
    text = _text("Dockerfile")
    assert "python:3.11" in text, "基础镜像要跟上 requires-python>=3.11"
    assert "HOST=0.0.0.0" in text, "容器里必须绑 0.0.0.0，否则端口映射接不住"
    # 绑 0.0.0.0 时没有 AUTH_TOKEN 会拒绝启动（src/security.py）。镜像里**不许**给 AUTH_TOKEN
    # 任何默认值 —— 一旦有默认值，fail-closed 就变成「默认用一把人尽皆知的钥匙」。
    assert not re.search(r"^\s*AUTH_TOKEN\s*=", text, re.M), \
        "Dockerfile 不许给 AUTH_TOKEN 设默认值：那会绕过启动闸门"


def test_dockerfile_runs_as_non_root_and_has_healthcheck():
    text = _text("Dockerfile")
    assert re.search(r"^USER\s+(?!root)", text, re.M), "容器要以非 root 运行"
    assert "HEALTHCHECK" in text and "/api/session" in text, "要有健康检查（打 /api/session）"
    assert 'VOLUME ["/data"]' in text, "数据目录必须是 volume"


def test_dockerignore_keeps_user_data_and_secrets_out_of_the_image():
    text = _text(".dockerignore")
    entries = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    for needed in (".env", ".venv", "state", "uploads", "chroma_db", ".git"):
        assert needed in entries, f".dockerignore 必须排除 {needed}（别把用户数据/密钥烤进镜像）"


def test_compose_project_persists_data_and_listens_in_container():
    text = _text("docker-compose.yml")
    assert "/data" in text, "state/uploads/chroma 要落到挂载点"
    assert re.search(r"^\s*-?\s*ragflow-data:/data", text, re.M) or "ragflow-data:/data" in text, \
        "要有命名卷，否则删容器就删库"
    assert "HOST: 0.0.0.0" in text, "compose 要覆盖成容器内监听"
    assert "127.0.0.1:" in text, "默认只发布到宿主机回环，别主动暴露到局域网"


def test_compose_host_port_is_one_knob():
    """改端口只该动一个地方：`.env` 里的 HOST_PORT。

    容器内端口必须固定（健康检查与 `scripts/update_deploy.py` 的冒烟都按 8080 走），
    所以 compose 发布端口要写成 `${HOST_PORT:-8080}:8080` —— 左边可配、右边固定。
    """
    text = _text("docker-compose.yml")
    assert "${HOST_PORT:-8080}:8080" in text, \
        "对外端口要抽成 HOST_PORT，且容器内端口固定 8080"
    assert "PORT: 8080" in text, "容器内端口写死 8080，别让 .env 的 PORT 改到它"
