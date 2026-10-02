# -*- coding: utf-8 -*-
"""上传端点回归：一次请求多文件、删除单个文件（P1-17）。

多文件必须**一次请求只重建一次索引**：_rebuild_collection 是「清空 + 全量重写」，
如果前端按文件数一个个传，N 个文件会触发 N 次全量重嵌入（O(N²) 的活）。
"""
import pytest
from fastapi.testclient import TestClient

import src.server as server
import src.state as state


class FakeCollection:
    """记录写入/清空的假 collection，用来断言索引被正确重建。"""

    def __init__(self):
        self.added = []
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        return {"ids": list(range(len(self.added)))}

    def delete(self, ids):
        self.added = []

    def add(self, documents, ids):
        self.added.extend(documents)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (FakeCollection(), True))
    # 用文件名当解析结果，避免真的去读 PDF/DOCX
    monkeypatch.setattr(server, "load_file", lambda f: getattr(f, "filename", "") or "")
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def _post(client, sid, items, field="files", conv_id=None):
    files = [(field, (name, content, "text/plain")) for name, content in items]
    data = {"session_id": sid}
    if conv_id:
        data["conv_id"] = conv_id
    return client.post("/api/upload", files=files, data=data)


# ── 多文件上传 ──

def test_upload_multiple_files_in_one_request(client):
    sid = client.get("/api/session").json()["session_id"]

    r = _post(client, sid, [("a.txt", b"aaa"), ("b.txt", b"bbb")])

    assert r.status_code == 200, r.text
    d = r.json()
    assert d["file_names"] == ["a.txt", "b.txt"]
    assert d["file_count"] == 2

    conv = server.sessions[sid]["conversations"][d["conv_id"]]
    assert [f["file_name"] for f in conv["files"]] == ["a.txt", "b.txt"]
    assert conv["full_text"] == "a.txt\n\nb.txt"

    coll = conv["collection"]
    assert any("a.txt" in c for c in coll.added) and any("b.txt" in c for c in coll.added)
    assert coll.get_calls == 1, f"一次批量上传只该重建一次索引，实际 {coll.get_calls} 次"


def test_upload_accepts_legacy_single_file_field(client):
    """旧的 file（单数）字段仍然要能用，避免破坏既有调用方。"""
    sid = client.get("/api/session").json()["session_id"]

    r = _post(client, sid, [("x.txt", b"x")], field="file")

    assert r.status_code == 200, r.text
    assert r.json()["file_names"] == ["x.txt"]


def test_upload_without_any_file_returns_400(client):
    sid = client.get("/api/session").json()["session_id"]

    r = client.post("/api/upload", data={"session_id": sid})

    assert r.status_code == 400
    assert "文件" in r.json()["error"]


def test_upload_total_size_limit_covers_whole_batch(client, monkeypatch):
    """上限按「本次请求的总字节数」算，不能每个文件单独合规就放行。"""
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 1)
    sid = client.get("/api/session").json()["session_id"]

    r = _post(client, sid, [("a.txt", b"x" * 8), ("b.txt", b"y" * 8)])

    assert r.status_code == 413
    assert "过大" in r.json()["error"]


def test_upload_rejects_empty_content_in_batch(client, monkeypatch):
    """批次里有空文件就整体拒绝，不能只落一半。"""
    monkeypatch.setattr(server, "load_file",
                        lambda f: "" if f.filename == "empty.txt" else "内容")
    sid = client.get("/api/session").json()["session_id"]

    r = _post(client, sid, [("a.txt", b"aaa"), ("empty.txt", b"   ")])

    assert r.status_code == 400
    assert "empty.txt" in r.json()["error"]
    assert server.sessions[sid]["conversations"] == {}, "失败时不该落下半个对话"


# ── 删除单个文件 ──

def _upload_two(client, sid):
    d = _post(client, sid, [("a.txt", b"aaa"), ("b.txt", b"bbb")]).json()
    return d["conv_id"]


def test_delete_single_file_rebuilds_index(client):
    sid = client.get("/api/session").json()["session_id"]
    cid = _upload_two(client, sid)

    r = client.delete(f"/api/conversations/{cid}/files/b.txt", params={"session_id": sid})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file_count"] == 1
    assert [f["file_name"] for f in body["files"]] == ["a.txt"]

    conv = server.sessions[sid]["conversations"][cid]
    assert [f["file_name"] for f in conv["files"]] == ["a.txt"]
    assert not any("b.txt" in c for c in conv["collection"].added), "被删文件的索引必须清掉"
    assert any("a.txt" in c for c in conv["collection"].added)


def test_delete_last_file_clears_full_text_and_index(client):
    """删掉最后一个文件后，派生的 full_text 必须清空，索引也不能留旧内容。"""
    sid = client.get("/api/session").json()["session_id"]
    d = _post(client, sid, [("only.txt", b"only")]).json()
    cid = d["conv_id"]

    r = client.delete(f"/api/conversations/{cid}/files/only.txt", params={"session_id": sid})

    assert r.status_code == 200, r.text
    conv = server.sessions[sid]["conversations"][cid]
    assert conv["files"] == []
    assert conv["full_text"] == "", "最后一个文件删掉后不能留着旧全文"
    assert conv["collection"].added == [], "索引里不该留旧内容"


def test_delete_missing_file_returns_404(client):
    sid = client.get("/api/session").json()["session_id"]
    cid = _upload_two(client, sid)

    r = client.delete(f"/api/conversations/{cid}/files/nope.txt", params={"session_id": sid})

    assert r.status_code == 404
    conv = server.sessions[sid]["conversations"][cid]
    assert len(conv["files"]) == 2, "没删成功就不能动原有文件"


def test_delete_file_in_missing_conversation_returns_404(client):
    sid = client.get("/api/session").json()["session_id"]

    r = client.delete("/api/conversations/no-such-conv/files/a.txt", params={"session_id": sid})

    assert r.status_code == 404


def test_delete_file_with_chinese_and_space_in_name(client):
    """文件名含中文/空格时，前端会做 percent 编码，服务端要能正确还原。"""
    from urllib.parse import quote

    sid = client.get("/api/session").json()["session_id"]
    name = "财报 2024.txt"
    cid = _post(client, sid, [(name, b"x")]).json()["conv_id"]

    r = client.delete(f"/api/conversations/{cid}/files/{quote(name, safe='')}",
                      params={"session_id": sid})

    assert r.status_code == 200, r.text
    assert server.sessions[sid]["conversations"][cid]["files"] == []
