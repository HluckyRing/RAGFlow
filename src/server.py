# -*- coding: utf-8 -*-
import json
import uuid
import os
import time
import threading
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

import src.state as state
from src.config import logger
from src.loaders import load_file
from src.pdf_ingestion import split_text
from src.retrieval import init_vector_store, sanitize_collection_name, drop_collection
from src.llm import resolve_query, retrieve_and_build_context, stream_answer

# session_id -> 状态 dict。与 state 模块的缓存共用同一个对象，改动会被 _save_state 落盘。
sessions = {}
_sessions_lock = threading.Lock()

app = FastAPI(title="RAGFlow")


def _conv_cname(conv_id):
    return sanitize_collection_name("conv_" + conv_id)


def _serializable(sess):
    """剥掉 collection / use_vector / _cname 这些不可序列化的运行时字段。"""
    convs_out = {}
    for cid, c in sess.get("conversations", {}).items():
        convs_out[cid] = {
            "name": c.get("name", "对话"),
            "created_at": c.get("created_at", 0),
            "files": [{"file_name": f.get("file_name"), "file_text": f.get("file_text", "")}
                      for f in c.get("files", [])],
            "messages": c.get("messages", []),
            "full_text": c.get("full_text", ""),
            "collection_name": c.get("_cname") or c.get("collection_name") or _conv_cname(cid),
        }
    return {"active_conv": sess.get("active_conv"), "conversations": convs_out}


def _save_state(session_id):
    """把内存里的会话状态落盘。真正的写盘在 state.save() 的会话锁内完成。"""
    with _sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        return
    state.save(session_id, _serializable(sess))


def _hydrate_collections(session_id, data):
    """首次加载后为每个对话恢复 Chroma 句柄和运行时字段（只做一次）。"""
    for cid, cd in data.get("conversations", {}).items():
        cname = cd.get("collection_name") or _conv_cname(cid)
        collection, use_vector = init_vector_store(cname)
        cd["collection"] = collection
        cd["use_vector"] = use_vector
        cd["_cname"] = cname
        if not cd.get("full_text") and cd.get("files"):
            cd["full_text"] = "\n\n".join(f.get("file_text", "") for f in cd["files"])


def _ensure_session(session_id, client_supplied=True):
    """取会话状态。

    只有首次访问才从磁盘加载并重建 collection，避免每个请求都全量重载。
    client_supplied=False 表示 sid 是服务端刚生成的，不参与旧状态认领。
    """
    if not session_id:
        session_id = uuid.uuid4().hex[:16]
        client_supplied = False
    with _sessions_lock:
        sess = sessions.get(session_id)
    if sess is not None:
        return session_id, sess

    data = state.load(session_id, allow_legacy=client_supplied)
    _hydrate_collections(session_id, data)
    with _sessions_lock:
        sess = sessions.setdefault(session_id, data)
    return session_id, sess


def _rebuild_collection(conv):
    """Rebuild ChromaDB collection from all files in a conversation."""
    collection = conv.get("collection")
    if not collection:
        return
    try:
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
    except Exception as e:
        logger.debug("清理 collection 数据失败: %s", e)
    full_text = conv.get("full_text", "")
    if full_text:
        chunks = split_text(full_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/session")
async def create_session():
    sid, _ = _ensure_session("")
    return {"session_id": sid}


@app.get("/api/state")
async def get_state(session_id: str):
    sid, sess = _ensure_session(session_id)
    convs = []
    for cid, c in sess.get("conversations", {}).items():
        convs.append({
            "id": cid, "name": c["name"],
            "created_at": c.get("created_at", 0),
            "file_count": len(c.get("files", [])),
            "message_count": len(c.get("messages", [])),
        })
    return {
        "conversations": sorted(convs, key=lambda x: x["created_at"], reverse=True),
        "active_conv": sess.get("active_conv"),
    }


@app.post("/api/conversations")
async def create_conversation(request: Request):
    data = await request.json()
    sid, sess = _ensure_session(data.get("session_id", ""))
    cid = uuid.uuid4().hex[:12]
    name = data.get("name") or time.strftime("%m-%d %H:%M")
    cname = _conv_cname(cid)
    collection, use_vector = init_vector_store(cname)
    conv = {
        "id": cid, "name": name, "created_at": time.time(),
        "files": [], "full_text": "",
        "collection": collection, "use_vector": use_vector,
        "messages": [], "_cname": cname,
    }
    with state.lock_for(sid):
        sess["conversations"][cid] = conv
        sess["active_conv"] = cid
        _save_state(sid)
    return {"id": cid, "name": name, "session_id": sid}


@app.post("/api/conversations/{conv_id}/switch")
async def switch_conversation(conv_id: str, request: Request):
    data = await request.json()
    sid, sess = _ensure_session(data.get("session_id", ""))
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        sess["active_conv"] = conv_id
        _save_state(sid)
    return {"active_conv": conv_id}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: Request):
    data = await request.json()
    sid, sess = _ensure_session(data.get("session_id", ""))
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        sess["conversations"][conv_id]["name"] = data.get("name", "对话")
        _save_state(sid)
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, session_id: str):
    sid, sess = _ensure_session(session_id)
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        conv = sess["conversations"].pop(conv_id)
        if sess.get("active_conv") == conv_id:
            remaining = list(sess["conversations"].keys())
            sess["active_conv"] = remaining[0] if remaining else None
        _save_state(sid)
    # 集合也要删掉，否则 chroma_db 只增不减
    drop_collection(conv.get("_cname") or conv.get("collection_name"))
    return {"ok": True}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, session_id: str):
    sid, sess = _ensure_session(session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    return {
        "id": conv_id, "name": conv["name"],
        "created_at": conv.get("created_at", 0),
        "files": [{"file_name": f["file_name"]} for f in conv.get("files", [])],
        "messages": conv.get("messages", []),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(""), conv_id: str = Form("")):
    sid, sess = _ensure_session(session_id)

    try:
        file_text = load_file(file)          # 解析放在锁外，别让慢 I/O 占着会话锁
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not file_text.strip():
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    file_name = file.filename

    with state.lock_for(sid):
        if not conv_id:
            conv_id = uuid.uuid4().hex[:12]
            cname = _conv_cname(conv_id)
            collection, use_vector = init_vector_store(cname)
            sess["conversations"][conv_id] = {
                "id": conv_id, "name": time.strftime("%m-%d %H:%M"), "created_at": time.time(),
                "files": [], "full_text": "",
                "collection": collection, "use_vector": use_vector,
                "messages": [], "_cname": cname,
            }
            sess["active_conv"] = conv_id

        conv = sess["conversations"].get(conv_id)
        if not conv:
            return JSONResponse({"error": "对话不存在"}, status_code=404)

        existing = next((f for f in conv.get("files", []) if f["file_name"] == file_name), None)
        if existing:
            existing["file_text"] = file_text
        else:
            conv.setdefault("files", []).append({"file_name": file_name, "file_text": file_text})

        conv["full_text"] = "\n\n".join(f["file_text"] for f in conv["files"])
        _rebuild_collection(conv)
        _save_state(sid)

        result = {
            "conv_id": conv_id, "file_name": file_name, "file_count": len(conv["files"]),
            "conv_name": conv["name"],
            "files": [{"file_name": f["file_name"]} for f in conv["files"]],
            "messages": conv.get("messages", []),
        }
    return result


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")
    conv_id = data.get("conv_id", "")
    if not question or not session_id:
        return JSONResponse({"error": "缺少参数"}, status_code=400)

    sid, sess = _ensure_session(session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=400)
    if not conv.get("files"):
        return JSONResponse({"error": "请先上传文件"}, status_code=400)

    # 取快照：此刻【不含】当前提问，所以 resolve_query 里的上一轮就是 [-1]
    history = list(conv.get("messages", []))
    search_query = resolve_query(question, history)
    file_count = len(conv.get("files", []))
    dynamic_top_k = max(10, file_count * 3)
    context, _ = retrieve_and_build_context(
        search_query, conv["full_text"], conv["collection"], conv["use_vector"],
        top_k=dynamic_top_k
    )
    if not context:
        return JSONResponse({"error": "未找到相关内容"}, status_code=400)

    with state.lock_for(sid):
        # 检索期间对话可能被删掉，这里按 id 重新取一次
        target = sessions.get(sid, {}).get("conversations", {}).get(conv_id)
        if target is None:
            return JSONResponse({"error": "对话不存在"}, status_code=400)
        target.setdefault("messages", []).append({"role": "user", "content": question})
        _save_state(sid)

    async def generate():
        full_answer = ""
        for token in stream_answer(question, context, history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        # 流式结束后按 id 重新取回对话再落盘：不持有过期引用，也不怕中途被删
        with state.lock_for(sid):
            target = sessions.get(sid, {}).get("conversations", {}).get(conv_id)
            if target is not None:
                target.setdefault("messages", []).append({"role": "assistant", "content": full_answer})
                _save_state(sid)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, session_id: str):
    sid, sess = _ensure_session(session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    return {"messages": conv.get("messages", []) if conv else []}


if __name__ == "__main__":
    import uvicorn
    logger.info("RAGFlow 启动: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
