#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""备份 Docker 数据卷（会话状态 + 上传原件 + 向量库）。

为什么需要它（2026-11-26）
--------------------------
数据全在命名卷里（`ai_rag_project_ragflow-data` → 容器内 `/data`），而它在 Windows 上
**不是一个能直接复制的文件夹**（躺在 Docker Desktop 的 WSL2 虚拟磁盘里）。也就是说：

- `docker compose down` 安全（卷保留）；
- `docker compose down -v` **连卷一起删，没有任何回收站**；
- 删了容器镜像重建没问题，删了卷 = 课件、对话、索引全没。

所以备份只能是「进容器把 `/data` 打包出来」。

做法与两个刻意的选择
--------------------
1. **先停容器再打包**：向量库是 SQLite（`chroma.sqlite3`），运行中边写边 `tar` 可能拿到
   写了一半的页。文件小（实测 ~140 MB），停几秒无所谓；`--no-stop` 可以跳过（自己承担风险）。
2. **保留最近 N 份**（默认 7）：不清理的话一天跑一次就是无限增长。

用法
----
    python scripts/backup_data.py                  # 停容器 → 打包 → 启容器 → 只留最近 7 份
    python scripts/backup_data.py --keep 30        # 留 30 份
    python scripts/backup_data.py --no-stop        # 不停容器（有并发写时不建议）
    python scripts/backup_data.py --out D:/bak     # 换个输出目录

恢复（**会覆盖当前 /data**，先确认文件名）：
    docker compose stop
    docker run --rm -v ai_rag_project_ragflow-data:/data \\
        -v "D:/Python/ai_rag_project/backups:/backup" alpine \\
        sh -c "rm -rf /data/*; tar xzf /backup/<文件名>.tar.gz -C /data"
    docker compose start
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOLUME_HINT = "ragflow-data"
SERVICE = "ragflow"
SNAPSHOT_RE = re.compile(r"^ragflow-data-(\d{8}-\d{6})\.tar\.gz$")


def snapshot_name(now=None):
    """快照文件名：`ragflow-data-YYYYmmdd-HHMMSS.tar.gz`（时间戳前缀，字典序 = 时间序）。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"ragflow-data-{stamp}.tar.gz"


def select_prunable(names, keep, keep_newest=None):
    """按 `keep` 选出该删的旧快照（保留最新的 keep 份，并且**永不删** `keep_newest`）。"""
    known = sorted(name for name in names if SNAPSHOT_RE.match(name))
    if keep_newest in known:
        rest = [name for name in known if name != keep_newest]
        survivors = known[-keep:] if keep > 0 else []
        prunable = [name for name in rest if name not in survivors]
        return prunable
    if keep <= 0:
        return list(known)
    return known[:-keep]


def _ensure_utf8_stdout():
    """Windows 控制台默认 GBK，打「✓」这类字符会 UnicodeEncodeError 直接把脚本打断。

    这里把输出流切成 UTF-8（errors=replace）：现代 Windows Terminal 显示正常，
    老控制台最差也只是个别字符变问号，**不会因为打印而失败**。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _run(cmd, check=True):
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
                          capture_output=True)
    if check and proc.returncode != 0:
        detail = ((proc.stderr or proc.stdout or "").strip().splitlines() or [""])[-3:]
        raise RuntimeError(f"命令失败（{' '.join(cmd)}）：{' / '.join(detail)}")
    return proc


def find_volume():
    """定位 compose 用的数据卷（名字里带 `ragflow-data`，可能是 `<项目名>_ragflow-data`）。"""
    proc = _run(["docker", "volume", "ls", "--filter", f"name={VOLUME_HINT}",
                 "--format", "{{.Name}}"])
    names = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not names:
        raise RuntimeError(f"找不到数据卷（名字含 {VOLUME_HINT}）——服务还没起过？")
    # 优先精确后缀，避免误伤别的项目里名字相近的卷
    exact = [name for name in names if name.endswith(VOLUME_HINT)]
    return (exact or names)[0]


def backup(out_dir=None, keep=7, stop_container=True, start_after=True, service=SERVICE):
    """打一份快照，返回 `Path`。`start_after=False` 留给调用方稍后自己启动（更新流程用）。"""
    _ensure_utf8_stdout()
    volume = find_volume()
    target_dir = Path(out_dir) if out_dir else (ROOT / "backups")
    target_dir.mkdir(parents=True, exist_ok=True)
    name = snapshot_name()
    stopped = False
    try:
        if stop_container:
            print(f"==> 停容器 {service}（拿到一致的 SQLite 快照）", flush=True)
            _run(["docker", "compose", "stop", service], check=False)
            stopped = True
        print(f"==> 打包数据卷 {volume} -> {name}", flush=True)
        # 容器内路径故意用 :ro：备份过程绝不能改到原数据
        _run(["docker", "run", "--rm", "-v", f"{volume}:/data:ro",
              "-v", f"{target_dir}:/backup", "alpine",
              "tar", "czf", f"/backup/{name}", "-C", "/data", "."])
    finally:
        if stopped and start_after:
            print(f"==> 启容器 {service}", flush=True)
            _run(["docker", "compose", "start", service], check=False)

    path = target_dir / name
    for old in select_prunable([p.name for p in target_dir.glob("ragflow-data-*.tar.gz")],
                              keep, keep_newest=name):
        (target_dir / old).unlink(missing_ok=True)
        print(f"    清理旧快照 {old}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="备份 Docker 数据卷")
    parser.add_argument("--out", default=None, help="输出目录（默认 ./backups）")
    parser.add_argument("--keep", type=int, default=7, help="保留最近几份（默认 7）")
    parser.add_argument("--no-stop", action="store_true", help="不停容器（有写入时不建议）")
    args = parser.parse_args(argv)
    try:
        path = backup(out_dir=args.out, keep=args.keep, stop_container=not args.no_stop)
    except RuntimeError as exc:
        print(f"[FAIL] 备份失败：{exc}", file=sys.stderr)
        return 1
    size = path.stat().st_size / 1024 / 1024
    print(f"[OK] 备份完成：{path}（{size:.1f} MB）", flush=True)
    print("  恢复见本脚本开头的说明（会覆盖当前 /data，务必先确认文件名）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
