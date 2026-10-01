# -*- coding: utf-8 -*-
"""会话状态持久化 —— 每个 session_id 一个独立的 JSON 文件。

设计要点：
- 文件名由 sha1(session_id) 派生，客户端传什么 sid 都无法逃出 STATE_DIR（防路径穿越）
- 写入用「唯一临时名 → fsync → os.replace」，保证原子性，失败会清理临时文件
- 每会话一把 RLock，串行化同一会话的「改内存 + 落盘」
- 进程内缓存，避免每个请求都重读磁盘、重建 Chroma collection
- 旧版全局 kb_state.json 通过「改名」做一次性原子认领

已知限制：缓存和锁都在进程内，因此只支持单 worker 部署（默认就是单 worker）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_rag.state")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.getenv("STATE_DIR") or (PROJECT_ROOT / "state"))
LEGACY_STATE_FILE = Path(os.getenv("LEGACY_STATE_FILE") or (PROJECT_ROOT / "kb_state.json"))
SCHEMA_VERSION = 2

# 文件存在但不可用（损坏已备份 / 归属校验不符）。不要用它去覆盖原文件。
UNUSABLE = object()

_cache: dict[str, dict] = {}
_locks: dict[str, threading.RLock] = {}
_registry_lock = threading.Lock()


# ── 基础工具 ──

def file_for(session_id: str) -> Path:
    """由 session_id 派生状态文件路径。sha1 保证结果永远是 STATE_DIR 下的一个文件名。"""
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:32]
    return STATE_DIR / f"{digest}.json"


def lock_for(session_id: str) -> threading.RLock:
    """取该会话的锁。用 RLock，允许路由内嵌套调用 _save_state。"""
    with _registry_lock:
        lock = _locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            _locks[session_id] = lock
        return lock


def ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def new_state(session_id: str) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "session_id": session_id,
        "active_conv": None,
        "conversations": {},
    }


# ── 读写 ──

def read_file(session_id: str) -> Optional[dict]:
    """读磁盘。

    返回 dict 表示读到有效数据；None 表示「没有这个文件」；
    UNUSABLE 表示「文件在但不可用」，调用方不要覆盖它。
    """
    path = file_for(session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        broken = path.with_name(f"{path.stem}.corrupt-{int(time.time())}.json")
        try:
            os.replace(path, broken)
            logger.error("状态文件损坏，已备份为 %s：%s", broken.name, e)
        except Exception as e2:
            logger.error("状态文件损坏且备份失败（%s）：%s", path.name, e2)
        return None

    if not isinstance(data, dict):
        logger.error("状态文件格式异常（不是 JSON 对象）：%s", path.name)
        return UNUSABLE
    stored = data.get("session_id")
    if stored and stored != session_id:
        logger.error("状态文件归属校验失败（期望 %s，文件内是 %s）：%s", session_id, stored, path.name)
        return UNUSABLE
    data.setdefault("active_conv", None)
    data.setdefault("conversations", {})
    return data


def write_file(session_id: str, data: dict) -> bool:
    """原子写入。返回是否成功（失败只记日志，不抛异常，避免打断请求流程）。"""
    try:
        ensure_dir()
        path = file_for(session_id)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        payload = dict(data)
        payload["version"] = SCHEMA_VERSION
        payload["session_id"] = session_id
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            raise
        return True
    except Exception as e:
        logger.error("保存会话状态失败（%s）：%s", session_id, e)
        return False


# ── 旧版全局状态的一次性迁移 ──

def legacy_available() -> bool:
    return LEGACY_STATE_FILE.exists()


def claim_legacy(session_id: str) -> Optional[dict]:
    """把旧版全局 kb_state.json 认领给 session_id。

    用 os.replace 改名做原子的一次性判定：改名成功 = 本次认领生效，
    其他会话再也认领不到。原文件保留为 kb_state.json.migrated，便于回滚。
    """
    if not LEGACY_STATE_FILE.exists():
        return None
    try:
        raw = json.loads(LEGACY_STATE_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("旧状态文件读取失败，跳过迁移：%s", e)
        return None
    convs = raw.get("conversations") if isinstance(raw, dict) else None
    if not isinstance(convs, dict):
        logger.warning("旧状态文件格式不符，跳过迁移")
        return None

    migrated = LEGACY_STATE_FILE.with_suffix(f".json.migrated")
    try:
        os.replace(LEGACY_STATE_FILE, migrated)
    except Exception as e:
        logger.warning("旧状态文件认领失败：%s", e)
        return None

    data = new_state(session_id)
    data["active_conv"] = raw.get("active_conv")
    data["conversations"] = convs
    write_file(session_id, data)
    logger.info("已把旧状态（%d 个对话）迁移到会话 %s，原文件保留为 %s",
                len(convs), session_id, migrated.name)
    return data


# ── 对外主入口 ──

def load(session_id: str, allow_legacy: bool = True) -> dict:
    """取会话状态：优先命中内存缓存，否则读磁盘；都没有就新建。

    allow_legacy=False 用于服务端自己新发的 sid（/api/session）——
    这类 sid 不该去认领旧数据，同时立刻落一个空文件，把它标记成「已存在的会话」。
    """
    with lock_for(session_id):
        cached = _cache.get(session_id)
        if cached is not None:
            return cached

        data = read_file(session_id)
        if data is UNUSABLE:
            logger.error("会话 %s 的状态文件不可用，本次使用空状态（不覆盖原文件）", session_id)
            data = new_state(session_id)
        elif data is None:
            if allow_legacy and legacy_available():
                data = claim_legacy(session_id)
            if data is None:
                data = new_state(session_id)
                write_file(session_id, data)
        _cache[session_id] = data
        return data


def save(session_id: str, payload: dict) -> bool:
    with lock_for(session_id):
        return write_file(session_id, payload)


def drop_cache(session_id: str) -> None:
    """仅用于测试。"""
    with lock_for(session_id):
        _cache.pop(session_id, None)
