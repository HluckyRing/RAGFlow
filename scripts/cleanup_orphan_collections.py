#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""孤儿 Chroma collection 清理（P2）。

背景：每个对话一个 collection，删对话时 server 会级联删除（`drop_collection`）。
但如果状态文件丢过 / 换过 sid / 手工删过 state 文件，就会留下没人引用的 collection，
在 chroma_db 里只增不减。

判定：
- 「被引用」= 出现在任一状态文件（`state/*.json`、`kb_state.json`、
  `kb_state.json.migrated`）的 `conversations[*].collection_name`
- 「孤儿」= Chroma 里存在、但没有任何状态引用的 collection
- 只有名字以 `kb_conv_` 开头的孤儿才会被删；其它未引用集合默认只报告
  （要一起删得显式加 `--include-foreign`）

用法：
    python scripts/cleanup_orphan_collections.py              # 干跑（默认），只打印计划
    python scripts/cleanup_orphan_collections.py --apply      # 真正执行删除

退出码：0 正常/无事可做/干跑；1 有删除失败；2 被安全闸中止。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import state
from src.config import VECTOR_DB_PATH, logger

PROJECT_COLLECTION_PREFIX = "kb_conv_"


def _legacy_candidates(legacy_file):
    """旧版全局状态：原文件与迁移后的 `.json.migrated` 都要看。"""
    legacy_file = Path(legacy_file)
    return [legacy_file, legacy_file.with_suffix(".json.migrated")]


def collect_referenced(state_paths):
    """扫所有状态文件，收集被引用的 collection 名。

    单个文件坏掉只记警告并跳过 —— 清理脚本绝不能因为一个坏文件就误判别人的数据为孤儿。
    """
    names = set()
    for path in state_paths:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning("读取状态文件失败，跳过 %s: %s", path, e)
            continue
        try:
            data = json.loads(raw)
        except ValueError as e:
            logger.warning("状态文件不是合法 JSON，跳过 %s: %s", path, e)
            continue
        convs = data.get("conversations") if isinstance(data, dict) else None
        if not isinstance(convs, dict):
            continue
        for conv in convs.values():
            if not isinstance(conv, dict):
                continue
            name = conv.get("collection_name") or conv.get("_cname")
            if name:
                names.add(str(name))
    return names


def find_orphans(all_names, referenced, include_foreign=False):
    """返回 (可删的孤儿, 保留报告的未引用「外来」集合)，两个列表都按名字排序。"""
    orphans, foreign = [], []
    for name in sorted(set(all_names)):
        if name in referenced:
            continue
        if name.startswith(PROJECT_COLLECTION_PREFIX):
            orphans.append(name)
        else:
            foreign.append(name)
    if include_foreign:
        orphans.extend(foreign)
        foreign = []
    return orphans, foreign


def _stop(client):
    """尽力关掉 Chroma 的 sqlite 句柄：Windows 上不放手会挡住目录清理。"""
    try:
        client._system.stop()
    except Exception:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="清理没有状态引用的孤儿 Chroma collection（默认干跑，不删任何东西）")
    parser.add_argument("--apply", action="store_true", help="真的删除；不加则只打印计划")
    parser.add_argument("--include-foreign", action="store_true",
                        help=f"连不以 {PROJECT_COLLECTION_PREFIX} 开头的未引用集合也删")
    parser.add_argument("--force", action="store_true",
                        help="一个被引用的集合都没扫到时也照删（危险，仅在你清楚状态文件在哪时使用）")
    parser.add_argument("--db-path", default=None, help=f"向量库路径，默认 {VECTOR_DB_PATH}")
    parser.add_argument("--state-dir", default=None, help="状态目录，默认 src.state.STATE_DIR")
    parser.add_argument("--legacy-file", default=None,
                        help="旧版全局状态文件，默认 src.state.LEGACY_STATE_FILE")
    args = parser.parse_args(argv)

    db_path = Path(args.db_path) if args.db_path else Path(VECTOR_DB_PATH)
    state_dir = Path(args.state_dir) if args.state_dir else Path(state.STATE_DIR)
    legacy_file = Path(args.legacy_file) if args.legacy_file else Path(state.LEGACY_STATE_FILE)

    if not db_path.exists():
        print(f"向量库路径不存在，无需清理：{db_path}")
        return 0

    state_files = sorted(state_dir.glob("*.json")) if state_dir.exists() else []
    state_files += _legacy_candidates(legacy_file)
    existing_state_files = [p for p in state_files if p.exists()]
    referenced = collect_referenced(state_files)

    import chromadb

    client = chromadb.PersistentClient(path=str(db_path))
    try:
        all_names = sorted(c.name for c in client.list_collections())
        orphans, foreign = find_orphans(all_names, referenced, args.include_foreign)

        print(f"向量库：{db_path}")
        print(f"状态文件：{len(existing_state_files)} 个")
        for path in existing_state_files:
            print(f"  {path}")
        print(f"集合总数 {len(all_names)}，被状态引用 {len(referenced)}，"
              f"可清理孤儿 {len(orphans)}，保留的未引用集合 {len(foreign)}")
        for name in foreign:
            print(f"  [保留] {name}（不以 {PROJECT_COLLECTION_PREFIX} 开头；要一起删请加 --include-foreign）")

        if not orphans:
            print("没有需要清理的孤儿集合。")
            return 0

        for name in orphans:
            print(f"  [待删] {name}")

        if not referenced and not args.force:
            print("!! 一个被引用的集合都没扫到，这通常说明 --state-dir 指错了。")
            print("   为避免把库删空，已中止；确认路径无误可加 --force。")
            return 2

        if not args.apply:
            print(f"干跑模式：以上 {len(orphans)} 个集合会被删除。加 --apply 真正执行。")
            return 0

        deleted, failed = 0, []
        for name in orphans:
            try:
                client.delete_collection(name)
                deleted += 1
                print(f"  已删除 {name}")
            except Exception as e:
                failed.append((name, str(e)[:120]))
        print(f"完成：删除 {deleted} 个，失败 {len(failed)} 个。")
        for name, err in failed:
            print(f"  删除失败 {name}: {err}")
        return 1 if failed else 0
    finally:
        # 只在以 CLI 方式直接运行时才关 Chroma 的 system：它按路径共享，
        # 在进程内停机会连累同进程的其它客户端（正常退出时 OS 会释放句柄）。
        if __name__ == "__main__":
            _stop(client)


if __name__ == "__main__":
    raise SystemExit(main())
