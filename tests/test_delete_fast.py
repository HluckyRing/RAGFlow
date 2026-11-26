# -*- coding: utf-8 -*-
"""删除文件 / 索引重建的**耗时口径**（2026-11-26）。

背景（真实反馈 + 实测）：删一个不小心传错的文件要等很久，界面上还没反应。根因两处：

1. 删除走的是 `_rebuild_collection` —— 把**剩下所有文件**重新切片 + 重新 embedding。
   实测 3 个小文件（~20 块）删一个要 4.4 s；真实语料 811 块时约 3 分钟。删除根本不需要
   embedding，应该只删该文件的 chunk（`delete(where={"file_name": ...})`）。
2. `_rebuild_collection` 是**逐块** `collection.add`，811 块 = 811 次 embedding 调用。

这里钉住四件事：定向删除真的被用上、老索引仍然正确（退回重建而不是静默删 0 条）、
块数不符也退回重建、重建改成**批量**写。
"""
import math

import pytest

import src.server as server
from src.pdf_ingestion import split_files


class WhereCollection:
    """支持 `where=` 过滤与 `count()` 的假 collection（chromadb 的真实能力子集）。"""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})       # id -> (doc, meta)
        self.add_calls = 0
        self.delete_where_calls = 0

    def add(self, documents, ids, metadatas=None):
        self.add_calls += 1
        metas = metadatas or [{} for _ in ids]
        for cid, doc, meta in zip(ids, documents, metas):
            self.rows[cid] = (doc, meta)

    def get(self, where=None, **_kw):
        if where:
            return {"ids": [cid for cid, (_d, m) in self.rows.items()
                            if all((m or {}).get(k) == v for k, v in where.items())]}
        return {"ids": list(self.rows)}

    def delete(self, ids=None, where=None):
        if where:
            self.delete_where_calls += 1
            targets = [cid for cid, (_d, m) in self.rows.items()
                       if all((m or {}).get(k) == v for k, v in where.items())]
            for cid in targets:
                self.rows.pop(cid, None)
        elif ids:
            for cid in ids:
                self.rows.pop(cid, None)

    def count(self):
        return len(self.rows)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """隔离状态目录 + 用假 collection（同 tests/test_upload.py 的做法）。"""
    import src.state as state
    from fastapi.testclient import TestClient

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (WhereCollection(), True))
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()


def _conv(files):
    return {"files": files, "collection": WhereCollection()}


def _text(tag, size=1200):
    """造一段够切出若干块的正文字。"""
    return f"{tag}\n\n" + ("这是一段用来测试切片的正文。" * (size // 14))


def test_delete_only_drops_that_files_chunks_without_rebuilding():
    """删一个文件不许再整份重建：定向 delete(where=file_name)，一次 add 都不该有。"""
    files = [{"file_name": "甲.txt", "file_text": _text("甲")},
             {"file_name": "乙.txt", "file_text": _text("乙")}]
    conv = _conv(files)
    server._rebuild_collection(conv)
    adds_before = conv["collection"].add_calls
    kept_ids = [cid for cid, (_d, m) in conv["collection"].rows.items()
                if (m or {}).get("file_name") == "乙.txt"]
    assert kept_ids, "重建后应当有乙的 chunk"

    conv["files"] = [f for f in conv["files"] if f["file_name"] != "甲.txt"]   # _remove_file 的次序
    ok = server._delete_file_chunks(conv, "甲.txt")

    assert ok is True, "定向删除应当成功"
    assert conv["collection"].delete_where_calls == 1
    assert conv["collection"].add_calls == adds_before, "定向删除不该再触发 embedding"
    assert not [cid for cid, (_d, m) in conv["collection"].rows.items()
                if (m or {}).get("file_name") == "甲.txt"], "甲的 chunk 必须清干净"
    assert all(cid in conv["collection"].rows for cid in kept_ids), "乙的 chunk 一条都不许动"
    assert conv["collection"].count() == len(kept_ids)


def test_old_index_without_file_name_metadata_falls_back_to_rebuild():
    """P2 之前的老索引没有 file_name 元数据：`delete(where=...)` 会**静默删 0 条**。

    这种情况必须退回整份重建 —— 否则检索还会召回已删文件的内容，比慢更糟。
    """
    stale_rows = {f"old_{i}": ("旧内容", {"source": "legacy"}) for i in range(4)}
    conv = {"files": [{"file_name": "乙.txt", "file_text": _text("乙")}],
            "collection": WhereCollection(stale_rows)}

    ok = server._delete_file_chunks(conv, "甲.txt")

    assert ok is False, "块数对不上就必须报「没删干净」，让调用方重建"
    server._rebuild_collection(conv)
    expected = server._expected_chunk_count(conv)
    assert conv["collection"].count() == expected, "重建后索引块数必须与当前文件一致"


def test_chunk_count_mismatch_falls_back_to_rebuild():
    """过滤条件删掉了、但索引里还残留别的脏块（历史遗留）→ 也要退回重建。"""
    conv = {"files": [{"file_name": "乙.txt", "file_text": _text("乙")}],
            "collection": WhereCollection({"ghost_1": ("幽灵", {"file_name": "甲.txt"})})}
    conv["collection"].rows["ghost_2"] = ("另一个幽灵", {"file_name": "丙.txt"})

    ok = server._delete_file_chunks(conv, "甲.txt")

    assert ok is False


def test_deleting_last_file_empties_the_index():
    """删光文件必须把索引清空（否则会残留可召回的孤儿 chunk）。"""
    conv = _conv([{"file_name": "唯一.txt", "file_text": _text("唯一")}])
    server._rebuild_collection(conv)
    assert conv["collection"].count() > 0

    conv["files"] = []
    assert server._delete_file_chunks(conv, "唯一.txt") is True
    assert conv["collection"].count() == 0


def test_rebuild_writes_in_batches_not_one_chunk_at_a_time():
    """811 块的语料逐块 add = 811 次 embedding 调用（分钟级）；改成分批后是十几次。"""
    big = "填充句子。" * 20000          # 约 10 万字 → 两百多块，必定跨批
    conv = _conv([{"file_name": "大文件.txt", "file_text": big}])

    server._rebuild_collection(conv)

    total = server._expected_chunk_count(conv)
    assert total > server.EMBED_BATCH, f"构造要足够大才能跨批，实际 {total} 块"
    assert conv["collection"].add_calls == math.ceil(total / server.EMBED_BATCH), \
        "add 调用次数应当是 ceil(块数 / EMBED_BATCH)"
    assert conv["collection"].count() == total, "分批写不许丢块"


def test_remove_file_end_to_end_is_targeted(client, monkeypatch):
    """走完整的 DELETE 路由：返回的 files 正确，且没触发重建（add 次数不变）。"""
    monkeypatch.setattr(server, "load_file", lambda f: _text("内容"))
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (WhereCollection(), True))
    sid = client.get("/api/session").json()["session_id"]
    up = client.post("/api/upload",
                     data={"session_id": sid, "conv_id": ""},
                     files=[("files", ("甲.txt", b"a", "text/plain")),
                            ("files", ("乙.txt", b"b", "text/plain"))])
    conv_id = up.json()["conv_id"]
    conv = server.sessions[sid]["conversations"][conv_id]
    adds_before = conv["collection"].add_calls

    r = client.delete(f"/api/conversations/{conv_id}/files/甲.txt", params={"session_id": sid})

    assert r.status_code == 200, r.text
    assert [f["file_name"] for f in r.json()["files"]] == ["乙.txt"]
    assert conv["collection"].add_calls == adds_before, "删除不该再重建索引"
    assert conv["collection"].count() == server._expected_chunk_count(conv)
