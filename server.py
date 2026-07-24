import json
import uuid
import os
import time
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from src.config import logger
from src.loaders import load_file
from src.pdf_ingestion import split_text
from src.retrieval import init_vector_store, sanitize_collection_name
from src.llm import resolve_query, retrieve_and_build_context, stream_answer

STATE_FILE = "kb_state.json"
sessions = {}

app = FastAPI(title="RAGFlow")


def _cname(name):
    return sanitize_collection_name(name)


def _save_state(session_id):
    sess = sessions.get(session_id)
    if not sess:
        return
    data = {"active_conv": sess.get("active_conv")}
    convo = {}
    for cid, c in sess.get("conversations", {}).items():
        convo[cid] = {
            "name": c["name"],
            "kb_name": c.get("kb_name"),
            "messages": c["messages"],
            "created_at": c["created_at"],
        }
    data["conversations"] = convo
    kbs = {}
    for name, kb in sess.get("kbs", {}).items():
        kbs[name] = {
            "collection_name": kb.get("_cname", _cname(name)),
            "files": kb["files"],
            "full_text": kb["full_text"],
            "use_vector": kb["use_vector"],
        }
    data["kbs"] = kbs
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("保存状态失败: %s", e)


def _load_state(session_id):
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        sess = sessions[session_id]
        sess["kbs"] = {}
        for name, kb_data in data.get("kbs", {}).items():
            cname = kb_data["collection_name"]
            collection, use_vector = init_vector_store(cname)
            sess["kbs"][name] = {
                "files": kb_data.get("files", []),
                "full_text": kb_data.get("full_text", ""),
                "collection": collection,
                "use_vector": use_vector,
                "_cname": cname,
            }
        sess["conversations"] = {}
        for cid, cdata in data.get("conversations", {}).items():
            sess["conversations"][cid] = {
                "id": cid,
                "name": cdata.get("name", "对话"),
                "kb_name": cdata.get("kb_name"),
                "messages": cdata.get("messages", []),
                "created_at": cdata.get("created_at", time.time()),
            }
        sess["active_conv"] = data.get("active_conv")
        logger.info("已恢复 %d 个知识库, %d 个对话", len(sess["kbs"]), len(sess["conversations"]))
    except Exception as e:
        logger.warning("恢复状态失败: %s", e)


def _add_text_to_kb(kb, new_text):
    kb["full_text"] += "\n\n" + new_text
    collection = kb.get("collection")
    if collection:
        new_chunks = split_text(new_text)
        offset = collection.count()
        for i, chunk in enumerate(new_chunks):
            collection.add(documents=[chunk], ids=[f"c{offset + i}"])


def _register_kb(session_id, kb_name, file_text, filename):
    sess = sessions[session_id]
    if kb_name in sess.get("kbs", {}):
        kb = sess["kbs"][kb_name]
        if filename and filename not in kb["files"]:
            kb["files"].append(filename)
            _add_text_to_kb(kb, file_text)
        _save_state(session_id)
        return kb, False
    cname = _cname(kb_name)
    collection, use_vector = init_vector_store(cname)
    kb = {
        "files": [filename] if filename else [],
        "full_text": file_text,
        "collection": collection,
        "use_vector": use_vector,
        "_cname": cname,
    }
    if collection and file_text and collection.count() == 0:
        chunks = split_text(file_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])
    sess.setdefault("kbs", {})[kb_name] = kb
    _save_state(session_id)
    return kb, True


def _ensure_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = uuid.uuid4().hex[:16]
        sessions[session_id] = {"kbs": {}, "conversations": {}, "active_conv": None}
        _load_state(session_id)
    sess = sessions[session_id]
    sess.setdefault("kbs", {})
    sess.setdefault("conversations", {})
    return session_id, sess


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/session")
async def create_session():
    sid, _ = _ensure_session("")
    return {"session_id": sid}


@app.get("/api/state")
async def get_state(session_id: str):
    _, sess = _ensure_session(session_id)
    convs = []
    for cid, c in sess.get("conversations", {}).items():
        convs.append({
            "id": cid,
            "name": c["name"],
            "kb_name": c.get("kb_name"),
            "message_count": len(c.get("messages", [])),
            "created_at": c.get("created_at", 0),
        })
    kbs = [{"name": n, "file_count": len(k.get("files", [])), "chars": len(k.get("full_text", ""))}
           for n, k in sess.get("kbs", {}).items()]
    return {
        "conversations": sorted(convs, key=lambda x: x["created_at"], reverse=True),
        "active_conv": sess.get("active_conv"),
        "kbs": kbs,
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = "", kb_name: str = ""):
    sid, sess = _ensure_session(session_id)
    try:
        full_text = load_file(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not full_text.strip():
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    name = kb_name.strip() if kb_name.strip() else file.filename
    kb, is_new = _register_kb(sid, name, full_text, file.filename)
    _save_state(sid)
    return {"session_id": sid, "kb_name": name, "is_new": is_new, "chars": len(full_text)}


@app.delete("/api/kbs/{kb_name}")
async def delete_kb(kb_name: str, session_id: str):
    _, sess = _ensure_session(session_id)
    if kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    del sess["kbs"][kb_name]
    for c in sess.get("conversations", {}).values():
        if c.get("kb_name") == kb_name:
            c["kb_name"] = None
    _save_state(session_id)
    return {"ok": True}


@app.post("/api/conversations")
async def create_conversation(request: Request):
    data = await request.json()
    sid, sess = _ensure_session(data.get("session_id", ""))
    cid = uuid.uuid4().hex[:12]
    kb_name = data.get("kb_name", "")
    if kb_name and kb_name not in sess.get("kbs", {}):
        kb_name = ""
    conv = {
        "id": cid,
        "name": data.get("name", "新对话"),
        "kb_name": kb_name,
        "messages": [],
        "created_at": time.time(),
    }
    sess["conversations"][cid] = conv
    if not sess.get("active_conv"):
        sess["active_conv"] = cid
    _save_state(sid)
    return {"id": cid, "name": conv["name"], "kb_name": kb_name, "session_id": sid}


@app.post("/api/conversations/{conv_id}/switch")
async def switch_conversation(conv_id: str, request: Request):
    data = await request.json()
    _, sess = _ensure_session(data.get("session_id", ""))
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    sess["active_conv"] = conv_id
    _save_state(data["session_id"])
    return {"active_conv": conv_id}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, session_id: str):
    _, sess = _ensure_session(session_id)
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    del sess["conversations"][conv_id]
    if sess.get("active_conv") == conv_id:
        remaining = list(sess["conversations"].keys())
        sess["active_conv"] = remaining[0] if remaining else None
    _save_state(session_id)
    return {"ok": True}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: Request):
    data = await request.json()
    _, sess = _ensure_session(data.get("session_id", ""))
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    sess["conversations"][conv_id]["name"] = data.get("name", "对话")
    _save_state(data["session_id"])
    return {"ok": True}


@app.post("/api/conversations/{conv_id}/bind-kb")
async def bind_kb(conv_id: str, request: Request):
    data = await request.json()
    _, sess = _ensure_session(data.get("session_id", ""))
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    kb_name = data.get("kb_name", "")
    if kb_name and kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    sess["conversations"][conv_id]["kb_name"] = kb_name
    _save_state(data["session_id"])
    return {"ok": True}


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")
    conv_id = data.get("conv_id", "")
    if not question or not session_id:
        return JSONResponse({"error": "缺少参数"}, status_code=400)

    _, sess = _ensure_session(session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=400)

    kb = sess.get("kbs", {}).get(conv.get("kb_name")) if conv.get("kb_name") else None
    if not kb:
        return JSONResponse({"error": "请先上传文件"}, status_code=400)

    history = conv.get("messages", [])
    search_query = resolve_query(question, history)
    context, sources = retrieve_and_build_context(
        search_query, kb["full_text"], kb["collection"], kb["use_vector"]
    )
    if not context:
        return JSONResponse({"error": "未找到相关内容"}, status_code=400)

    conv["messages"].append({"role": "user", "content": question})
    if conv.get("kb_name") and not conv["name"].startswith("📄"):
        pass
    _save_state(session_id)

    async def generate():
        full_answer = ""
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s[:300] for s in sources]})}\n\n"
        for token in stream_answer(question, context, history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        conv["messages"].append({"role": "assistant", "content": full_answer})
        _save_state(session_id)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, session_id: str):
    _, sess = _ensure_session(session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    return {"messages": conv.get("messages", []) if conv else []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
