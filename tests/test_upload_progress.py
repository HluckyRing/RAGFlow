# -*- coding: utf-8 -*-
"""上传进度条（2026-11-26）。

背景：上传一个文件要「解析 → 切片 → 逐批 embedding」，实测 26 万字 / 250 块要 13.6 s，
而前端只有一个静态的「仍在解析中…」—— 用户分不清是卡死还是在干活。

做法：客户端生成 `upload_id` 随上传请求发过去，后端在内存进度表里记阶段
（`parse` 第几个文件 / `index` 已嵌入几块），前端轮询 `/api/upload/progress`。

这里钉住：阶段真的上报了、进度端点有归属校验、任务结束后条目被删、老客户端（不带
upload_id）行为不变。
"""
import pytest
from fastapi.testclient import TestClient

import src.server as server
import src.state as state


class FakeCollection:
    def __init__(self):
        self.docs = {}

    def get(self, where=None, **_kw):
        return {"ids": list(self.docs)}

    def delete(self, ids=None, where=None):
        for cid in (ids or []):
            self.docs.pop(cid, None)

    def add(self, documents, ids, metadatas=None):
        for cid in ids:
            self.docs[cid] = True

    def count(self):
        return len(self.docs)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    server._UPLOAD_PROGRESS.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (FakeCollection(), True))
    yield TestClient(server.app)
    server.sessions.clear()
    server._UPLOAD_PROGRESS.clear()
    state._cache.clear()


def _post(client, sid, files, upload_id=None):
    data = {"session_id": sid, "conv_id": ""}
    if upload_id is not None:
        data["upload_id"] = upload_id
    return client.post("/api/upload", data=data,
                       files=[("files", (name, body, "text/plain")) for name, body in files])


def test_upload_reports_parse_then_index_progress(client, monkeypatch):
    """阶段必须按「解析第几个文件 → 嵌入第几块」上报，且 index 的 total 是真实块数。"""
    seen = []
    original = server._progress_update

    def spy(upload_id, **fields):
        if upload_id:
            seen.append(dict(fields))
        return original(upload_id, **fields)

    monkeypatch.setattr(server, "_progress_update", spy)
    monkeypatch.setattr(server, "load_file", lambda f: "正文内容。" * 400)

    sid = client.get("/api/session").json()["session_id"]
    r = _post(client, sid, [("甲.txt", b"a"), ("乙.txt", b"b")], upload_id="task-1")
    assert r.status_code == 200, r.text

    stages = [s.get("stage") for s in seen]
    assert "parse" in stages and "index" in stages, f"要上报解析与嵌入两个阶段，实际 {stages}"
    index = [s for s in seen if s.get("stage") == "index" and s.get("total")]
    assert index, "embedding 阶段必须带 total（前端靠它算百分比）"
    assert index[-1]["done"] == index[-1]["total"], "最后一条 index 必须是 done == total"
    parses = [s for s in seen if s.get("stage") == "parse"]
    assert parses[0]["done"] == 0 and parses[0]["total"] == 2, "解析阶段按文件计数"
    assert parses[-1]["done"] == 2


def test_progress_entry_is_removed_when_upload_finishes(client, monkeypatch):
    """上传结束就该删条目：否则失败的请求会在内存里留垃圾。"""
    monkeypatch.setattr(server, "load_file", lambda f: "内容")
    sid = client.get("/api/session").json()["session_id"]
    assert _post(client, sid, [("甲.txt", b"a")], upload_id="task-2").status_code == 200
    assert server._progress_snapshot("task-2") is None


def test_progress_endpoint_reports_a_live_task(client):
    """进度端点返回阶段与耗时；`done`/`total` 一律整数（前端直接用）。"""
    server._progress_update("task-3", stage="index", done=64, total=250, file_name="")
    sid = client.get("/api/session").json()["session_id"]

    r = client.get("/api/upload/progress", params={"session_id": sid, "upload_id": "task-3"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stage"] == "index" and body["done"] == 64 and body["total"] == 250
    assert isinstance(body["elapsed"], float)


def test_progress_endpoint_hides_other_browsers_tasks(client):
    """别人的上传任务不该被你查到（和其它接口同一套归属校验）。"""
    server._progress_update("task-4", stage="index", done=1, total=2)
    sid = client.get("/api/session").json()["session_id"]
    other = TestClient(server.app)          # 另一个「浏览器」：没有归属 cookie
    r = other.get("/api/upload/progress", params={"session_id": sid, "upload_id": "task-4"})
    assert r.status_code == 403, r.text


def test_unknown_progress_id_is_404(client):
    """任务不存在/已结束 → 404，前端据此安静地停轮询。"""
    sid = client.get("/api/session").json()["session_id"]
    r = client.get("/api/upload/progress", params={"session_id": sid, "upload_id": "nope"})
    assert r.status_code == 404


def test_upload_without_upload_id_still_works(client, monkeypatch):
    """老客户端不带 upload_id：行为逐位不变，也不往进度表里写东西。"""
    monkeypatch.setattr(server, "load_file", lambda f: "内容")
    sid = client.get("/api/session").json()["session_id"]

    assert _post(client, sid, [("甲.txt", b"a")]).status_code == 200
    assert server._UPLOAD_PROGRESS == {}


def test_stale_progress_entries_are_recycled():
    """5 分钟没人更新的条目要顺手清掉（失败的上传不会自己删）。"""
    server._progress_update("old-task", stage="parse", done=0, total=1)
    with server._UPLOAD_PROGRESS_LOCK:
        server._UPLOAD_PROGRESS["old-task"]["updated"] -= server.UPLOAD_PROGRESS_TTL + 1

    server._progress_update("new-task", stage="parse", done=0, total=1)

    assert server._progress_snapshot("old-task") is None
    assert server._progress_snapshot("new-task") is not None
