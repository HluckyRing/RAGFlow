#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按当前 CHUNK_SIZE 重建所有对话的向量索引（P1 配套）。

为什么需要它
------------
CHUNK_SIZE 从 600 调到 480 之后，**已经上传过的对话**，它的 collection 里还是按
600 字切好的旧块 —— 不重建的话，切片改动对老对话等于没生效（重新上传任一文件会
触发整库重建，但逐个对话手工重传不现实）。

做法
----
读每个状态文件里 `conversations[*].files[].file_text`，重新切分并写回该对话的
collection，**复用 `src.server._rebuild_collection` 的「先加新、再删旧」语义**
（导入同一份实现，不复制一遍，避免两处逻辑跑偏）。索引是从状态派生的，所以重建
不需要改状态文件。

安全约定（与 `cleanup_orphan_collections.py` 一致）
--------------------------------------------------
- 默认**干跑**，只打印计划；`--apply` 才真的重建
- 一个状态文件都没扫到就中止（通常是 `--state-dir` 指错），`--force` 才越过
- 坏状态文件跳过并告警，不因为一个坏文件去动别人的数据
- 同一个 conv_id 只重建一次：`state/` 会话文件先扫、优先保留，legacy 只补 `state/`
  里没有的对话（两份副本的 collection 名相同，都重建会让后跑的覆盖前一份）

想先在副本上验证，用环境变量把向量库指到临时目录即可（config 已支持覆盖）：

    VECTOR_DB_PATH=/tmp/chroma-test STATE_DIR=/tmp/state-test python scripts/reindex_all.py --apply

退出码：0 正常/无事可做/干跑；1 有对话重建失败；2 被安全闸中止。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import state  # noqa: E402
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, logger  # noqa: E402


def _legacy_candidates(legacy_file):
    """旧版全局状态：原文件与迁移后的 `.json.migrated` 都要看。"""
    legacy_file = Path(legacy_file)
    return [legacy_file, legacy_file.with_suffix(".json.migrated")]


def collect_conversations(state_dir, legacy_file):
    """扫所有状态文件，返回 `(要重建的对话, 被跳过的重复对话)`。

    两个列表的元素都是 `(来源文件, session_id, conv_id, conv)`。坏文件只告警跳过
    —— 重建脚本绝不能因为一个坏文件就漏掉（或误动）别的对话。

    **同一个 conv_id 只保留首次遇到的那一份**，后到的重复项进第二个列表、只告警。
    为什么必须去重：集合名由 conv_id 派生（`server._conv_cname` / 状态里存下的
    `kb_conv_<conv_id>_<hash>`），同一个 conv_id 的两份副本指向**同一个 collection**，
    两份都重建就是「后跑的覆盖前一份」—— 静默毁数据。

    扫描顺序刻意是 `state/` 会话文件优先、legacy（`kb_state.json` / `.json.migrated`）
    追加在后：legacy 是**迁移前的旧快照**，内容可能更旧。真实数据里就有两个对话同时
    出现在 `.migrated` 与 `state/` 会话文件里（内容不同、collection_name 相同），
    所以 state/ 必须先扫、优先保留；legacy 只补 state/ 里没有的对话。
    """
    state_dir = Path(state_dir)
    paths = sorted(state_dir.glob("*.json")) if state_dir.exists() else []
    paths += [p for p in _legacy_candidates(legacy_file) if p.exists()]

    found = []
    duplicated = []
    seen = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("状态文件读取失败，跳过 %s：%s", path, e)
            continue
        if not isinstance(data, dict):
            logger.warning("状态文件不是 JSON 对象，跳过 %s", path)
            continue
        convs = data.get("conversations")
        if not isinstance(convs, dict):
            continue
        session_id = data.get("session_id") or path.stem
        for conv_id, conv in convs.items():
            if not isinstance(conv, dict):
                continue
            first = seen.get(conv_id)
            if first is not None:
                duplicated.append((path, session_id, conv_id, conv))
                logger.warning(
                    "跳过重复对话 %s（%s）：同一个 conv_id 已由 %s 负责；集合名由 conv_id "
                    "派生，两份都重建会让后跑的覆盖前一份。.json.migrated 是迁移前的旧"
                    "快照，确认不再需要可以归档该文件", conv_id, path, first)
                continue
            seen[conv_id] = path
            found.append((path, session_id, conv_id, conv))
    return found, duplicated


def _chunk_count(collection):
    """尽力取当前块数；取不到返回 None（只影响打印，不影响重建）。"""
    try:
        return collection.count()
    except Exception:
        return None


def open_collection(name):
    """打开（必要时创建）该对话的 collection。

    单独抽成一个函数是为了让测试能注入假实现 —— 真实实现会加载 embedding 模型。
    """
    import src.retrieval as retrieval

    collection, ok = retrieval.init_vector_store(name)
    if not ok or collection is None:
        raise RuntimeError(f"向量库不可用：{name}")
    return collection


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=f"按当前 CHUNK_SIZE={CHUNK_SIZE} 重建所有对话的向量索引（默认干跑）")
    parser.add_argument("--apply", action="store_true", help="真的重建；不加则只打印计划")
    parser.add_argument("--state-dir", default=None,
                        help=f"状态目录，默认 {state.STATE_DIR}")
    parser.add_argument("--force", action="store_true",
                        help="一个状态文件都没扫到时也照跑（仅在你清楚状态文件在哪时使用）")
    args = parser.parse_args(argv)

    state_dir = Path(args.state_dir) if args.state_dir else Path(state.STATE_DIR)
    legacy_file = Path(state.LEGACY_STATE_FILE)

    # 延迟导入：只有真要重建时才需要 FastAPI / Chroma 那一整套
    import src.server as server

    conversations, duplicated = collect_conversations(state_dir, legacy_file)
    if not conversations and not args.force:
        print(f"没有扫到任何状态文件：{state_dir}")
        print("   这通常说明 --state-dir 指错了；确认无误可加 --force。")
        return 2

    targets = [(path, sid, cid, conv) for path, sid, cid, conv in conversations
               if conv.get("files")]
    no_files = len(conversations) - len(targets)

    print(f"状态目录：{state_dir}")
    print(f"目标切片：CHUNK_SIZE={CHUNK_SIZE}（overlap={CHUNK_OVERLAP}）")
    print(f"对话 {len(conversations)} 个，其中有文件的 {len(targets)} 个，"
          f"无文件跳过 {no_files} 个")
    if duplicated:
        print(f"重复对话 {len(duplicated)} 个"
              f"（同一 conv_id 已由更早扫到的状态文件负责，跳过不重建）：")
        for dup_path, _dup_sid, dup_conv_id, _dup_conv in duplicated:
            print(f"  [跳过] {dup_conv_id}（{dup_path}）")

    if not targets:
        print("没有需要重建的对话。")
        return 0

    if not args.apply:
        for _path, _sid, conv_id, conv in targets:
            print(f"  [待重建] {conv_id}（{len(conv.get('files') or [])} 个文件）")
        print(f"干跑模式：以上 {len(targets)} 个对话会被重建。加 --apply 真正执行。")
        return 0

    failed = []
    for _path, _sid, conv_id, conv in targets:
        cname = conv.get("collection_name") or server._conv_cname(conv_id)
        try:
            collection = open_collection(cname)
            before = _chunk_count(collection)
            server._rebuild_collection({**conv, "collection": collection})
            after = _chunk_count(collection)
            print(f"  已重建 {conv_id}：{cname} 块数 {before} -> {after}")
        except Exception as e:
            failed.append((conv_id, str(e)[:160]))
            print(f"  重建失败 {conv_id}：{e}")

    print(f"完成：成功 {len(targets) - len(failed)} 个，失败 {len(failed)} 个。")
    for conv_id, err in failed:
        print(f"  失败 {conv_id}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
