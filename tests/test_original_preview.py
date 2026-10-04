# -*- coding: utf-8 -*-
"""原文件预览：上传时持久化原件、/raw 取原件、删除级联与旧文件兼容。

背景：此前服务端只存抽取出的 file_text，原始字节解析完就丢。要做「原文件预览」
必须先把原件落盘，并让上传 / 单文件删除 / 对话删除三条路径都照顾到它。
"""
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

import src.server as server
import src.state as state


class FakeCollection:
    def __init__(self):
        self.docs = {}

    def get(self):
        return {"ids": list(self.docs.keys())}

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)

    def add(self, documents, ids):
        for i, d in zip(ids, documents):
            self.docs[i] = d

    def count(self):
        return len(self.docs)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"))
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (FakeCollection(), True))
    monkeypatch.setattr(server, "load_file", lambda f: "text:" + (f.filename or ""))
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def _sid(client):
    return client.get("/api/session").json()["session_id"]


def _upload(client, sid, items, conv_id=None):
    files = [("files", (name, data, "application/octet-stream")) for name, data in items]
    data = {"session_id": sid}
    if conv_id:
        data["conv_id"] = conv_id
    return client.post("/api/upload", files=files, data=data)


def _conv(client, sid, name="a.txt", data=b"hello original"):
    d = _upload(client, sid, [(name, data)]).json()
    return d["conv_id"]


def _raw(client, sid, cid, name):
    return client.get(f"/api/conversations/{cid}/files/{quote(name, safe='')}/raw",
                      params={"session_id": sid})


def _preview(client, sid, cid, name):
    return client.get(f"/api/conversations/{cid}/files/{quote(name, safe='')}",
                      params={"session_id": sid})


def _stored_path(stored_name):
    matches = list(Path(server.UPLOAD_DIR).rglob(stored_name))
    assert len(matches) == 1, f"应恰好有一个原件文件，实际 {matches}"
    return matches[0]


def _all_stored():
    root = Path(server.UPLOAD_DIR)
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


# ── 上传持久化 ──

def test_upload_persists_original_bytes_and_stored_name(client):
    sid = _sid(client)
    raw = b"\x00\x01hello original\xff"
    cid = _conv(client, sid, "a.txt", raw)

    f = server.sessions[sid]["conversations"][cid]["files"][0]
    assert f.get("stored_name"), "files[] 必须记录原件引用 stored_name"
    assert _stored_path(f["stored_name"]).read_bytes() == raw


def test_upload_parse_failure_leaves_no_original(client, monkeypatch):
    def boom(_f):
        raise ValueError("解析失败")

    monkeypatch.setattr(server, "load_file", boom)
    sid = _sid(client)

    r = _upload(client, sid, [("b.txt", b"data")])

    assert r.status_code == 400
    assert _all_stored() == [], "解析失败不能留下孤儿原件"


def test_reupload_same_name_replaces_original(client):
    sid = _sid(client)
    cid = _conv(client, sid, "a.txt", b"first")
    old = _stored_path(server.sessions[sid]["conversations"][cid]["files"][0]["stored_name"])

    _upload(client, sid, [("a.txt", b"second")], conv_id=cid)

    f = server.sessions[sid]["conversations"][cid]["files"][0]
    assert _stored_path(f["stored_name"]).read_bytes() == b"second"
    assert not old.exists(), "重传同名文件后旧原件必须清掉，不然越积越多"
    assert len(_all_stored()) == 1


# ── /raw 接口 ──

def test_raw_serves_original_with_safe_headers(client):
    sid = _sid(client)
    raw = b"original bytes"
    cid = _conv(client, sid, "note.txt", raw)

    r = _raw(client, sid, cid, "note.txt")

    assert r.status_code == 200, r.text
    assert r.content == raw
    assert "text/plain" in r.headers["content-type"]
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert "inline" in r.headers.get("content-disposition", "")


def test_raw_pdf_inline_office_attachment(client):
    sid = _sid(client)
    cid = _upload(client, sid, [("d.pdf", b"%PDF-1.4 fake"),
                                ("w.docx", b"PK\x03\x04 fake")]).json()["conv_id"]

    pdf = _raw(client, sid, cid, "d.pdf")
    assert pdf.status_code == 200
    assert "application/pdf" in pdf.headers["content-type"]
    assert "inline" in pdf.headers.get("content-disposition", "")

    docx = _raw(client, sid, cid, "w.docx")
    assert docx.status_code == 200
    assert "wordprocessingml" in docx.headers["content-type"]
    assert "attachment" in docx.headers.get("content-disposition", "")


def test_raw_missing_file_returns_404(client):
    sid = _sid(client)
    cid = _conv(client, sid)

    r = _raw(client, sid, cid, "nope.txt")

    assert r.status_code == 404
    assert "文件" in r.json()["error"]


def test_raw_without_original_returns_404_with_chinese_hint(client):
    """升级前的旧数据没有 stored_name：要明确说原件没保存，而不是笼统 404。"""
    sid = _sid(client)
    cid = _conv(client, sid)
    server.sessions[sid]["conversations"][cid]["files"][0].pop("stored_name")

    r = _raw(client, sid, cid, "a.txt")

    assert r.status_code == 404
    assert "原件" in r.json()["error"]


def test_raw_is_isolated_by_session(client):
    sid_a = _sid(client)
    cid = _conv(client, sid_a)
    sid_b = _sid(client)

    r = _raw(client, sid_b, cid, "a.txt")

    assert r.status_code == 404


def test_original_survives_session_reload(client):
    """stored_name 必须落盘：清掉内存缓存（模拟重启）后原文件预览仍要可用。"""
    sid = _sid(client)
    cid = _conv(client, sid, "a.txt", b"persisted original")

    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()

    r = _raw(client, sid, cid, "a.txt")

    assert r.status_code == 200, r.text
    assert r.content == b"persisted original"


def test_preview_json_reports_has_original_and_ext(client):
    sid = _sid(client)
    cid = _conv(client, sid, "note.md", b"# hi")

    d = _preview(client, sid, cid, "note.md").json()
    assert d["has_original"] is True
    assert d["ext"] == ".md"

    server.sessions[sid]["conversations"][cid]["files"][0].pop("stored_name")
    d2 = _preview(client, sid, cid, "note.md").json()
    assert d2["has_original"] is False


# ── 删除级联 ──

def test_delete_file_removes_original(client):
    sid = _sid(client)
    cid = _conv(client, sid)
    stored = _stored_path(server.sessions[sid]["conversations"][cid]["files"][0]["stored_name"])

    r = client.delete(f"/api/conversations/{cid}/files/a.txt", params={"session_id": sid})

    assert r.status_code == 200, r.text
    assert not stored.exists()
    assert _all_stored() == []


def test_delete_conversation_removes_originals(client):
    sid = _sid(client)
    cid = _conv(client, sid)
    stored = _stored_path(server.sessions[sid]["conversations"][cid]["files"][0]["stored_name"])
    assert stored.exists()

    r = client.delete(f"/api/conversations/{cid}", params={"session_id": sid})

    assert r.status_code == 200, r.text
    assert not stored.exists()


# ── vendor 静态资源 ──

def test_vendor_route_whitelists_assets(client):
    r = client.get("/vendor/jszip.min.js")
    assert r.status_code == 200
    assert len(r.content) > 1000
    assert "javascript" in r.headers["content-type"]

    pptx = client.get("/vendor/aiden0z-pptx-renderer.browser.es.js")
    assert pptx.status_code == 200, pptx.text
    assert len(pptx.content) > 1000
    assert "javascript" in pptx.headers["content-type"]


def test_vendor_route_rejects_unknown_and_traversal(client):
    assert client.get("/vendor/not-a-lib.js").status_code == 404
    assert client.get("/vendor/..%2F..%2Fsrc%2Fserver.py").status_code == 404
