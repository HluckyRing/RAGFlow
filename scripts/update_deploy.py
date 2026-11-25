#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""更新并重启 Docker 部署：`git pull` → 重建镜像 → **等真健康**才算成功。

为什么需要它（2026-11-25）
--------------------------
`docker compose up -d` **不会**因为源码变了就重建镜像 —— compose 的 `build:` 只在镜像不存在时
才构建。于是「`git pull` 完就以为更新好了」是个静默陷阱：容器里跑的还是旧代码（本项目就踩过：
8080 上那个实例比最新提交早了一个多小时）。

正确姿势只有一句：`docker compose up -d --build`。这个脚本把前后该做的事串起来：

1. **前置检查**：Docker 守护进程在跑、`docker-compose.yml` 在、`.env` 里有 `AUTH_TOKEN`
   —— 容器必然对外监听（`HOST=0.0.0.0`），没有令牌服务会**拒绝启动**，与其等 120 s 超时，
   不如在第一步就说清；
2. **`git pull --ff-only`**：工作区脏就**中止**（不替用户决定 stash/提交，本地改动不能静默丢）；
3. **可选备份**：`--backup` 把数据卷打成 `backups/ragflow-data-<时间戳>.tar.gz`；
4. **`docker compose up -d --build`**：层缓存热时约 20 s（改 `src` 只重跑 COPY 层）；
5. **等健康检查**：轮询 `docker inspect` 的 `State.Health.Status`，**容器退出就立即报错并打日志**
   （最常见原因是 `.env` 缺 `AUTH_TOKEN`）；最后容器内 `curl /api/session` 再验一次。

用法
----
    python scripts/update_deploy.py              # 拉代码 + 重建 + 等健康
    python scripts/update_deploy.py --backup     # 先备份数据卷
    python scripts/update_deploy.py --no-pull    # 只重建当前本地代码（自己改完代码时）
    python scripts/update_deploy.py --dry-run    # 只做前置检查，不动任何东西

退出码：0 成功 / 1 失败（原因打在最后一行）/ 2 前置检查不过。
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = "docker-compose.yml"
ENV_FILE = ".env"
DEFAULT_SERVICE = "ragflow"
# `.env` 里必须有非空的 AUTH_TOKEN（值可以带引号）。注释行不算。
_AUTH_TOKEN_RE = re.compile(r"(?m)^\s*AUTH_TOKEN\s*=\s*[\"']?\S")


def has_auth_token(env_text):
    """`.env` 文本里有没有非空的 AUTH_TOKEN —— 容器没有它会拒绝启动。"""
    for line in str(env_text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _AUTH_TOKEN_RE.match(line):
            return True
    return False


def health_state(inspect_json):
    """从 `docker inspect` 的输出里取 `(运行状态, 健康状态)`。

    健康状态可能是 `starting` / `healthy` / `unhealthy`，镜像没定义 HEALTHCHECK 时是 `none`。
    """
    try:
        data = inspect_json[0] if isinstance(inspect_json, list) else inspect_json
    except (IndexError, TypeError):
        return ("", "none")
    state = data.get("State") or {}
    health = (state.get("Health") or {}).get("Status") or "none"
    return (state.get("Status") or "", health)


def _run(cmd, check=True, capture=True):
    """跑一条命令；失败时把 stderr 一起带出来（不吞掉原因）。"""
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
                          capture_output=capture)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"命令失败（{' '.join(cmd)}）：{' / '.join(detail)}")
    return proc


def say(msg):
    print(f"==> {msg}", flush=True)


def fail(msg):
    print(f"✗ {msg}", file=sys.stderr, flush=True)
    return 1


def preflight():
    """返回 `None` 表示可以继续，否则返回错误文案。"""
    if not (ROOT / COMPOSE_FILE).exists():
        return f"没找到 {COMPOSE_FILE}：脚本要在仓库里跑"
    if not (ROOT / ENV_FILE).exists():
        return ("没有 .env：先 cp .env.example .env 填 API_KEY，"
                "并用 python -c \"import secrets;print(secrets.token_hex(16))\" 生成 AUTH_TOKEN")
    if not has_auth_token((ROOT / ENV_FILE).read_text(encoding="utf-8", errors="replace")):
        return ("`.env` 里没有 AUTH_TOKEN —— 容器必然对外监听（HOST=0.0.0.0），"
                "没有它服务会**拒绝启动**。生成一串写进 .env：\n"
                "    python -c \"import secrets; print(secrets.token_hex(16))\"")
    try:
        _run(["docker", "version", "--format", "{{.ServerVersion}}"])
    except (RuntimeError, FileNotFoundError):
        return "Docker 守护进程没在跑（或没装）：先启动 Docker Desktop"
    return None


def backup_volume():
    """把数据卷打成 tar.gz（会话 + 上传原件 + 向量库）。没找到卷就跳过。"""
    proc = _run(["docker", "volume", "ls", "--filter", "name=ragflow-data",
                 "--format", "{{.Name}}"])
    names = [n.strip() for n in (proc.stdout or "").splitlines() if n.strip()]
    if not names:
        print("  没找到数据卷（首次部署？），跳过备份", flush=True)
        return
    volume = names[0]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "backups"
    out_dir.mkdir(exist_ok=True)
    name = f"ragflow-data-{stamp}.tar.gz"
    say(f"备份数据卷 {volume} → backups/{name}")
    _run(["docker", "run", "--rm", "-v", f"{volume}:/data:ro",
          "-v", f"{out_dir}:/backup", "alpine",
          "sh", "-c", f"tar czf /backup/{name} -C /data ."])


def wait_healthy(container, timeout):
    """轮询到 healthy（或镜像没定义 HEALTHCHECK 时按 running 算），返回 `(ok, 原因)`。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        try:
            proc = _run(["docker", "inspect", container], check=False)
            import json
            status, health = health_state(json.loads(proc.stdout or "[]"))
        except (RuntimeError, ValueError):
            continue
        if status == "exited":
            return (False, "容器退出了 —— 最常见原因：.env 缺 AUTH_TOKEN（对外监听必须有）或 API_KEY 写错")
        if health in ("healthy", "none") and status == "running":
            return (True, "")
    return (False, f"等健康检查超时（{timeout}s）—— 看 docker compose logs")


def main(argv=None):
    parser = argparse.ArgumentParser(description="更新并重启 Docker 部署")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="compose 服务名（默认 ragflow）")
    parser.add_argument("--health-timeout", type=int, default=120, help="等健康检查的秒数")
    parser.add_argument("--backup", action="store_true", help="重建前备份数据卷")
    parser.add_argument("--no-pull", action="store_true", help="不执行 git pull，只重建当前代码")
    parser.add_argument("--dry-run", action="store_true", help="只做前置检查")
    args = parser.parse_args(argv)

    problem = preflight()
    if args.dry_run:
        print("前置检查通过" if not problem else f"前置检查不通过：{problem}")
        return 0 if not problem else 2
    if problem:
        return 2 if fail(problem) else 2

    if not args.no_pull:
        dirty = (_run(["git", "status", "--porcelain"]).stdout or "").strip()
        if dirty:
            print("工作区有未提交改动（不替你 stash/提交，免得本地改动静默丢）：", flush=True)
            for line in dirty.splitlines()[:10]:
                print(f"  {line}", flush=True)
            return fail("先提交或 git stash，再跑更新（或用 --no-pull 只重建当前代码）")
        say("git pull --ff-only")
        proc = _run(["git", "pull", "--ff-only"], check=False)
        if proc.returncode != 0:
            return fail("git pull 失败（分支不一致？）—— 手动处理后重跑")

    commit = (_run(["git", "rev-parse", "--short", "HEAD"]).stdout or "").strip()
    say(f"目标版本：{commit}")

    if args.backup:
        try:
            backup_volume()
        except RuntimeError as exc:
            return fail(f"备份失败，已中止（不带 --backup 可跳过）：{exc}")

    say("docker compose up -d --build（层缓存热时约 20 s）")
    proc = _run(["docker", "compose", "up", "-d", "--build"], check=False)
    if proc.returncode != 0:
        _run(["docker", "compose", "logs", "--tail", "40"], check=False)
        return fail("构建/启动失败，日志见上")

    proc = _run(["docker", "compose", "ps", "-q", args.service])
    container = (proc.stdout or "").strip().splitlines()
    if not container:
        return fail("compose 没起出容器")
    container = container[0]

    say(f"等健康检查（最多 {args.health_timeout}s）")
    ok, reason = wait_healthy(container, args.health_timeout)
    if not ok:
        _run(["docker", "compose", "logs", "--tail", "40"], check=False)
        return fail(reason)

    smoke = _run(["docker", "exec", container, "sh", "-c",
                  "curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/session"],
                 check=False)
    code = (smoke.stdout or "").strip()
    print(f"✓ 已更新到 {commit}（容器 {container[:12]} 健康，/api/session → {code}）", flush=True)
    print("  数据在卷里，未受影响；回滚：git checkout <旧提交> 后重跑本脚本 --no-pull", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
