# -*- coding: utf-8 -*-
"""并发回归：慢请求不得阻塞事件循环。

改造前 generate() 是 async 生成器，且文件解析 / embedding 推理 / LLM 请求 /
写盘都直接跑在事件循环上，所以一个慢请求会把同进程的所有请求一起卡住。
这里把慢函数 mock 成阻塞 sleep，再用一个轻量请求去探测事件循环是否还活着。
"""
import asyncio
import inspect
import time

import httpx
import pytest

import src.server as server
import src.state as state

PROBE_SLEEP = 0.15       # 探测用的睡眠时长
PROBE_BUDGET = 0.20      # 允许的唤醒漂移上限；循环被卡住时会明显超过它


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (None, False))
    yield
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")


def _conversation_with_file(ac_sid, ac_conv_id):
    """给对话塞一个文件，跳过真实上传（避免碰 Chroma）。"""
    conv = server.sessions[ac_sid]["conversations"][ac_conv_id]
    conv["files"] = [{"file_name": "a.txt", "file_text": "x"}]
    conv["full_text"] = "x"
    conv["collection"] = None
    conv["use_vector"] = False
    return conv


async def _probe_while_busy(ac, busy_task, label):
    """探测事件循环是否还活着。

    关键是量 sleep 的【唤醒漂移】而不是探测请求自身的耗时：事件循环被卡住时，
    探测请求只是被推迟开始，它的耗时依然是毫秒级。只有当睡眠无法按时唤醒
    才说明循环被占住了。
    """
    t0 = time.perf_counter()
    await asyncio.sleep(PROBE_SLEEP)
    drift = (time.perf_counter() - t0) - PROBE_SLEEP

    r = await ac.get("/api/session")
    assert r.status_code == 200
    assert drift < PROBE_BUDGET, (
        f"{label}阻塞了事件循环：sleep({PROBE_SLEEP}) 实际漂移 {drift:.2f}s"
    )
    await busy_task


def test_slow_upload_does_not_block_event_loop(app_env, monkeypatch):
    def slow_load_file(_file):
        time.sleep(1.2)                       # 模拟解析大 PDF
        return "文档内容"

    monkeypatch.setattr(server, "load_file", slow_load_file)

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            task = asyncio.create_task(ac.post(
                "/api/upload",
                files={"file": ("a.txt", b"x", "text/plain")},
                data={"session_id": sid},
            ))
            await _probe_while_busy(ac, task, "慢上传")
            assert task.result().status_code == 200

    asyncio.run(scenario())


def test_slow_retrieval_does_not_block_event_loop(app_env, monkeypatch):
    def slow_retrieve(*args, **kwargs):
        time.sleep(1.2)                       # 模拟 HyDE 的 LLM 调用 + 向量检索
        return "背景内容", ["背景内容"]

    monkeypatch.setattr(server, "retrieve_and_build_context", slow_retrieve)
    monkeypatch.setattr(server, "stream_answer", lambda q, c, h: iter(["回答"]))

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            conv_id = (await ac.post("/api/conversations",
                                     json={"session_id": sid, "name": "T"})).json()["id"]
            _conversation_with_file(sid, conv_id)

            task = asyncio.create_task(ac.post("/api/chat", json={
                "question": "问题", "session_id": sid, "conv_id": conv_id}))
            await _probe_while_busy(ac, task, "慢检索")
            assert task.result().status_code == 200

    asyncio.run(scenario())


def test_slow_sse_stream_does_not_block_event_loop(app_env, monkeypatch):
    """这条专门盯着 generate() 必须是同步生成器。

    写成 async 生成器时它会走 AsyncIterable 分支、在事件循环上迭代，
    stream_answer 里等 LLM 出 token 的 sleep 就会把整个服务卡住。
    """
    monkeypatch.setattr(server, "retrieve_and_build_context",
                        lambda *a, **k: ("背景内容", ["背景内容"]))

    def slow_stream(question, context, history):
        for i in range(3):
            time.sleep(0.5)                   # 模拟等待 LLM 逐 token 返回
            yield f"片段{i}"

    monkeypatch.setattr(server, "stream_answer", slow_stream)

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            conv_id = (await ac.post("/api/conversations",
                                     json={"session_id": sid, "name": "T"})).json()["id"]
            _conversation_with_file(sid, conv_id)

            task = asyncio.create_task(ac.post("/api/chat", json={
                "question": "问题", "session_id": sid, "conv_id": conv_id}))
            await _probe_while_busy(ac, task, "SSE 流式输出")

            assert '"type": "done"' in task.result().text
            msgs = server.sessions[sid]["conversations"][conv_id]["messages"]
            assert [m["role"] for m in msgs] == ["user", "assistant"]
            assert msgs[1]["content"] == "片段0片段1片段2"

    asyncio.run(scenario())


def test_upload_to_missing_conversation_returns_404(app_env, monkeypatch):
    monkeypatch.setattr(server, "load_file", lambda _f: "内容")

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            r = await ac.post("/api/upload",
                              files={"file": ("a.txt", b"x", "text/plain")},
                              data={"session_id": sid, "conv_id": "不存在的对话"})
            assert r.status_code == 404

    asyncio.run(scenario())


def test_chat_hands_a_sync_generator_to_streaming_response(app_env, monkeypatch):
    """结构性守卫：交给 StreamingResponse 的必须是【同步】生成器。

    Starlette 只在 content 不是 AsyncIterable 时才用 iterate_in_threadpool 包装；
    一旦写成 async 生成器，它就会在事件循环上被迭代，慢 token 会卡住全服务。
    """
    monkeypatch.setattr(server, "retrieve_and_build_context",
                        lambda *a, **k: ("背景内容", ["背景内容"]))
    monkeypatch.setattr(server, "stream_answer", lambda q, c, h: iter(["回答"]))

    captured = {}
    real_streaming_response = server.StreamingResponse

    def spy(content, **kwargs):
        captured["content"] = content
        return real_streaming_response(content, **kwargs)

    monkeypatch.setattr(server, "StreamingResponse", spy)

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            conv_id = (await ac.post("/api/conversations",
                                     json={"session_id": sid, "name": "T"})).json()["id"]
            _conversation_with_file(sid, conv_id)
            await ac.post("/api/chat", json={
                "question": "问题", "session_id": sid, "conv_id": conv_id})

    asyncio.run(scenario())

    content = captured["content"]
    assert inspect.isgenerator(content), f"应当是同步生成器，实际是 {type(content).__name__}"
    assert not inspect.isasyncgen(content), "async 生成器会被 Starlette 放到事件循环上迭代"


def test_upload_builds_collection_for_api_created_conversation(app_env, monkeypatch):
    """先 POST /api/conversations 建对话、再上传的路径，必须补建向量库。

    P1-13（建对话不再提前建库）改造时曾漏掉这里：_commit_upload 只在
    「顺便新建对话」的分支里建库，于是这条路径上 collection 永远是 None，
    向量检索被静默降级成关键词检索 —— 单元测试因为 mock 了 init_vector_store
    而没发现，是真实 uvicorn 冒烟测出来的。
    """
    created = []

    class FakeCollection:
        def __init__(self):
            self.added = []

        def get(self):
            return {"ids": []}

        def add(self, documents, ids):
            self.added.extend(documents)

        def delete(self, ids):
            pass

    def fake_init_vector_store(name=None):
        coll = FakeCollection()
        created.append((name, coll))
        return coll, True

    monkeypatch.setattr(server, "init_vector_store", fake_init_vector_store)
    monkeypatch.setattr(server, "load_file", lambda _f: "文档内容")

    async def scenario():
        async with _client() as ac:
            sid = (await ac.get("/api/session")).json()["session_id"]
            conv_id = (await ac.post("/api/conversations",
                                     json={"session_id": sid, "name": "T"})).json()["id"]

            # 建对话阶段：刻意不建库，也不该产生空集合
            assert server.sessions[sid]["conversations"][conv_id]["collection"] is None
            assert created == [], "建对话时不该创建 collection"

            r = await ac.post("/api/upload",
                              files={"file": ("a.txt", b"x", "text/plain")},
                              data={"session_id": sid, "conv_id": conv_id})
            assert r.status_code == 200

            conv = server.sessions[sid]["conversations"][conv_id]
            assert conv["collection"] is not None, "上传后必须补建向量库"
            assert conv["use_vector"] is True
            assert created and created[0][0] == conv["_cname"]
            assert conv["collection"].added, "上传的文本应当被写入索引"

    asyncio.run(scenario())

