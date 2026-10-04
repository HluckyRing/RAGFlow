# -*- coding: utf-8 -*-
"""文件预览接口：读取单个文件的抽取正文。

预览读的是 loaders 抽取出的纯文本（服务端不保存上传原件，所以预览的就是进索引的
内容）。接口必须只读：不得落盘、不得重建索引，且要守住会话隔离。
"""
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

import src.server as server
import src.state as state


class FakeCollection:
    """与 tests/test_upload.py 同款替身：按 id 记账，便于断言「预览没碰索引」。"""

    def __init__(self):
        self.docs = {}
        self.get_calls = 0

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
    # 默认用文件名当解析结果，避免真的去读 PDF/DOCX
    monkeypatch.setattr(server, "load_file", lambda f: getattr(f, "filename", "") or "")
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def _make_loader(monkeypatch, mapping):
    monkeypatch.setattr(server, "load_file", lambda f: mapping[f.filename])


def _upload(client, sid, items, conv_id=None):
    files = [("files", (name, text.encode("utf-8"), "text/plain")) for name, text in items]
    data = {"session_id": sid}
    if conv_id:
        data["conv_id"] = conv_id
    return client.post("/api/upload", files=files, data=data)


def _upload_one(client, monkeypatch, name, text):
    _make_loader(monkeypatch, {name: text})
    sid = client.get("/api/session").json()["session_id"]
    d = _upload(client, sid, [(name, text)]).json()
    return sid, d["conv_id"]


def _get(client, sid, cid, name):
    return client.get(f"/api/conversations/{cid}/files/{quote(name, safe='')}",
                      params={"session_id": sid})


def test_preview_returns_extracted_text_and_char_count(client, monkeypatch):
    text = "第一段\n\n第二段"
    sid, cid = _upload_one(client, monkeypatch, "报告.txt", text)

    r = _get(client, sid, cid, "报告.txt")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["file_name"] == "报告.txt"
    assert body["file_text"] == text
    assert body["char_count"] == len(text)


def test_preview_missing_file_returns_404(client, monkeypatch):
    sid, cid = _upload_one(client, monkeypatch, "a.txt", "aaa")

    r = _get(client, sid, cid, "nope.txt")

    assert r.status_code == 404
    assert "文件" in r.json()["error"]


def test_preview_missing_conversation_returns_404(client, monkeypatch):
    sid, _ = _upload_one(client, monkeypatch, "a.txt", "aaa")

    r = _get(client, sid, "no-such-conv", "a.txt")

    assert r.status_code == 404
    assert "对话" in r.json()["error"]


def test_preview_is_isolated_by_session(client, monkeypatch):
    """别人 session 的对话拿不到 —— 预览也不能破会话隔离。"""
    text = "会话 A 的机密"
    _make_loader(monkeypatch, {"secret.txt": text})
    sid_a = client.get("/api/session").json()["session_id"]
    cid = _upload(client, sid_a, [("secret.txt", text)]).json()["conv_id"]

    sid_b = client.get("/api/session").json()["session_id"]
    r = _get(client, sid_b, cid, "secret.txt")

    assert r.status_code == 404, "跨 session 不应读到别人对话的文件"


def test_preview_handles_chinese_and_space_in_name(client, monkeypatch):
    name = "财报 2024.txt"
    sid, cid = _upload_one(client, monkeypatch, name, "内容")

    r = _get(client, sid, cid, name)

    assert r.status_code == 200, r.text
    assert r.json()["file_text"] == "内容"


def test_preview_is_read_only_does_not_rebuild_index(client, monkeypatch):
    """预览是只读：不能触发 collection 重建，也不能改动文件列表与派生全文。"""
    sid, cid = _upload_one(client, monkeypatch, "a.txt", "aaa")
    conv = server.sessions[sid]["conversations"][cid]
    coll = conv["collection"]
    before_calls = coll.get_calls
    before_docs = list(coll.docs.items())

    r = _get(client, sid, cid, "a.txt")

    assert r.status_code == 200, r.text
    assert coll.get_calls == before_calls, "预览不该读取/重建索引"
    assert list(coll.docs.items()) == before_docs, "预览不该改索引内容"
    assert [f["file_name"] for f in conv["files"]] == ["a.txt"]
    assert conv["full_text"] == "aaa"


def test_preview_returns_latest_content_after_reupload(client, monkeypatch):
    """同名文件重传后，预览必须读到新正文，不能停在旧内容。"""
    sid = client.get("/api/session").json()["session_id"]
    _make_loader(monkeypatch, {"a.txt": "旧内容"})
    cid = _upload(client, sid, [("a.txt", "旧内容")]).json()["conv_id"]
    _make_loader(monkeypatch, {"a.txt": "新内容"})
    _upload(client, sid, [("a.txt", "新内容")], conv_id=cid)

    r = _get(client, sid, cid, "a.txt")

    assert r.status_code == 200, r.text
    assert r.json()["file_text"] == "新内容"
