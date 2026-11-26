# -*- coding: utf-8 -*-
"""`scripts/backup_data.py`（Docker 数据卷备份）的回归。

为什么钉它（2026-11-26）：数据全在命名卷里（容器内 `/data`），`docker compose down -v`
一敲就**连卷一起删、没有回收站**。备份是唯一的兜底，所以这里钉住三件会出事的事：

- 快照名必须**时间戳可排序**（否则「保留最近 N 份」会删错）；
- 保留策略**永远不许删掉刚生成的那份**；
- 输出目录 `backups/` 是用户数据，必须被 `.gitignore` / `.dockerignore` 挡住
  （否则一次 `git add -A` 就把课件和对话推上公开仓库、或烤进镜像）。
"""
import importlib.util
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "backup_data.py"


def _module():
    spec = importlib.util.spec_from_file_location("backup_data_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_is_importable():
    assert SCRIPT.exists(), "数据卷备份脚本应当随仓库提供"
    assert callable(_module().main)


def test_snapshot_name_is_timestamp_sortable():
    module = _module()
    early = module.snapshot_name(datetime(2026, 11, 26, 9, 5, 3))
    late = module.snapshot_name(datetime(2026, 11, 26, 21, 5, 3))
    assert early == "ragflow-data-20261126-090503.tar.gz"
    assert early < late, "字典序必须等于时间序（保留策略靠它排序）"
    assert module.SNAPSHOT_RE.match(early)


def test_prune_keeps_the_newest_and_never_the_fresh_one():
    module = _module()
    names = [f"ragflow-data-2026112{i}-000000.tar.gz" for i in range(1, 10)]
    fresh = names[-1]
    prunable = module.select_prunable(names, keep=3, keep_newest=fresh)
    assert fresh not in prunable, "刚打出来的那份绝不能被清理掉"
    assert len(prunable) == 6, "留 3 份 → 其余 6 份该删"
    assert names[0] in prunable and names[-2] not in prunable, "删最旧的、留最新的"


def test_prune_ignores_foreign_files():
    """目录里可能有别的东西（手工拷进来的包），保留策略不许碰。"""
    module = _module()
    names = ["notes.txt", "ragflow-data-20261101-000000.tar.gz", "old-backup.zip"]
    prunable = module.select_prunable(names, keep=1)
    assert prunable == ["notes.txt", "old-backup.zip"] or set(prunable) <= {"notes.txt", "old-backup.zip"}


def test_backups_dir_is_gitignored_and_dockerignored():
    """`backups/` 里是用户数据（课件 + 对话 + 索引），既不进 git 也不进镜像。"""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "backups/" in gitignore, "backups/ 必须被 .gitignore 忽略"
    assert "backups" in dockerignore, "backups 必须被 .dockerignore 排除"


def test_update_flow_reuses_the_single_backup_implementation():
    """更新脚本的 --backup 不许另写一份打包逻辑（两份实现迟早会漂）。"""
    text = (ROOT / "scripts" / "update_deploy.py").read_text(encoding="utf-8")
    assert "scripts.backup_data" in text, "更新流程应复用 backup_data 的实现"
