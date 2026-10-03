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
    """记录写入/清空的假 collection，用来断言索引被正确重建。

    按 id 记账，而不是「一 delete 就清空」：_rebuild_collection 改成
    「先加新 chunk、成功后再删旧 id」之后，delete 只该删掉传给它的那些 id。
    这个替身对改造前后两种顺序都成立。
    """

    def __init__(self):
        self.docs = {}
        self.get_calls = 0

    @property
    def added(self):
        return list(self.docs.values())

    def get(self):
        self.get_calls += 1
        return {"ids": list(self.docs.keys())}

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    def add(self, documents, ids):
        for i, doc in zip(ids, documents):
            self.docs[i] = doc


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


# ── 脏字符（孤立代理字符）不得进入 embedding 与落盘 ──
# 2026-10-03 线上 500 的现场：pypdf 解析字体映射损坏的 PDF 会产出孤立代理字符，
# 它既能让 sentence-transformers 的 tokenizer 抛
# TypeError: TextEncodeInput must be Union[...]，
# 也能让 state 的 json.dump(ensure_ascii=False) 抛 UnicodeEncodeError。

def test_rebuild_collection_strips_surrogates_before_embedding():
    """不需要真模型：断言进 collection.add 的文本里没有代理字符就够了。"""
    conv = {"files": [{"file_name": "坏.pdf", "file_text": "正常\ud800文本"}],
            "collection": FakeCollection()}

    server._rebuild_collection(conv)

    added = conv["collection"].added
    assert added, "应当照常重建索引"
    assert all("\ud800" not in d for d in added), f"脏字符必须剥掉再进 embedding: {added!r}"


def test_upload_strips_surrogates_before_persisting(client, monkeypatch):
    """落盘那一步也会炸：ensure_ascii=False 遇到代理字符直接 UnicodeEncodeError，
    所以必须在写进 files[] 之前就洗干净。"""
    monkeypatch.setattr(server, "load_file", lambda f: "正常\ud800文本")

    sid = client.get("/api/session").json()["session_id"]
    r = _post(client, sid, [("坏.pdf", b"x")])

    assert r.status_code == 200, r.text
    conv = server.sessions[sid]["conversations"][r.json()["conv_id"]]
    assert conv["files"][0]["file_text"] == "正常文本"
    assert conv["full_text"] == "正常文本"


# ── 重建失败不得毁掉在用的索引 ──

def test_rebuild_keeps_working_index_when_add_fails():
    """add 中途抛错时，原来那份可用的索引必须原样保留。

    2026-10-03 的 500 就顺手造成了这个后果：_rebuild_collection 是「先 delete
    全部旧 id，再逐个 add」，add 一抛错，索引已经被删光 —— 一个大文件解析出的
    脏字符，就能把整个对话的检索毁掉（状态文件里明明还有正文，但索引空了）。
    """
    coll = FakeCollection()
    conv = {"files": [{"file_name": "a.txt", "file_text": "旧内容"}], "collection": coll}
    server._rebuild_collection(conv)
    before = list(coll.added)
    assert before, "先建立一份可用索引"

    def exploding_add(documents, ids):
        raise RuntimeError("embedding 挂了")

    coll.add = exploding_add
    conv["files"] = [{"file_name": "b.txt", "file_text": "新内容"}]

    with pytest.raises(RuntimeError):
        server._rebuild_collection(conv)

    assert list(coll.added) == before, f"旧索引必须原样保留，实际变成 {coll.added!r}"
