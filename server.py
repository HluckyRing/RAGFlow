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


def _conv_cname(conv_id):
    return sanitize_collection_name("conv_" + conv_id)


def _save_state(session_id):
    sess = sessions.get(session_id)
    if not sess:
        return
    data = {"active_conv": sess.get("active_conv")}
    convs_out = {}
    for cid, c in sess.get("conversations", {}).items():
        convs_out[cid] = {
            "name": c["name"],
            "created_at": c["created_at"],
            "files": [{"file_name": f["file_name"], "file_text": f["file_text"]} for f in c.get("files", [])],
            "messages": c.get("messages", []),
            "full_text": c.get("full_text", ""),
            "collection_name": c.get("_cname", _conv_cname(cid)),
        }
    data["conversations"] = convs_out
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("save state failed: %s", e)


def _load_state(session_id):
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        sess = sessions[session_id]
        convs = {}
        for cid, cd in data.get("conversations", {}).items():
            cname = cd.get("collection_name", _conv_cname(cid))
            collection, use_vector = init_vector_store(cname)
            full_text = cd.get("full_text", "")
            files = cd.get("files", [])
            if not full_text and files:
                full_text = "\n\n".join(f["file_text"] for f in files)
            convs[cid] = {
                "id": cid, "name": cd.get("name", "对话"),
                "created_at": cd.get("created_at", time.time()),
                "files": files,
                "full_text": full_text,
                "collection": collection,
                "use_vector": use_vector,
                "messages": cd.get("messages", []),
                "_cname": cname,
            }
        sess["conversations"] = convs
        sess["active_conv"] = data.get("active_conv")
        logger.info("restored %d conversations", len(convs))
    except Exception as e:
        logger.warning("load state failed: %s", e)


def _ensure_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = uuid.uuid4().hex[:16]
        sessions[session_id] = {"conversations": {}, "active_conv": None}
        _load_state(session_id)
    sess = sessions[session_id]
    sess.setdefault("conversations", {})
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
    except Exception:
        pass
    full_text = conv.get("full_text", "")
    if full_text:
        chunks = split_text(full_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])


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
    now = time.strftime("%m-%d %H:%M") if data.get("name") else time.strftime("%m-%d %H:%M")
    name = data.get("name") or now
    cname = _conv_cname(cid)
    collection, use_vector = init_vector_store(cname)
    conv = {
        "id": cid, "name": name, "created_at": time.time(),
        "files": [], "full_text": "",
        "collection": collection, "use_vector": use_vector,
        "messages": [], "_cname": cname,
    }
    sess["conversations"][cid] = conv
    sess["active_conv"] = cid
    _save_state(sid)
    return {"id": cid, "name": name, "session_id": sid}


@app.post("/api/conversations/{conv_id}/switch")
async def switch_conversation(conv_id: str, request: Request):
    data = await request.json()
    _, sess = _ensure_session(data.get("session_id", ""))
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    sess["active_conv"] = conv_id
    _save_state(data["session_id"])
    return {"active_conv": conv_id}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: Request):
    data = await request.json()
    _, sess = _ensure_session(data.get("session_id", ""))
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    sess["conversations"][conv_id]["name"] = data.get("name", "对话")
    _save_state(data["session_id"])
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, session_id: str):
    _, sess = _ensure_session(session_id)
    if conv_id not in sess.get("conversations", {}):
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    conv = sess["conversations"].pop(conv_id)
    try:
        collection = conv.get("collection")
        if collection:
            ids = collection.get().get("ids", [])
            if ids:
                collection.delete(ids=ids)
    except Exception:
        pass
    if sess.get("active_conv") == conv_id:
        remaining = list(sess["conversations"].keys())
        sess["active_conv"] = remaining[0] if remaining else None
    _save_state(session_id)
    return {"ok": True}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, session_id: str):
    _, sess = _ensure_session(session_id)
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
async def upload_file(file: UploadFile = File(...), session_id: str = "", conv_id: str = ""):
    sid, sess = _ensure_session(session_id)
    if not conv_id:
        return JSONResponse({"error": "请先创建或选择对话"}, status_code=400)

    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=404)

    try:
        file_text = load_file(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not file_text.strip():
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    file_name = file.filename
    existing = next((f for f in conv.get("files", []) if f["file_name"] == file_name), None)
    if existing:
        existing["file_text"] = file_text
    else:
        conv.setdefault("files", []).append({"file_name": file_name, "file_text": file_text})

    texts = [f["file_text"] for f in conv["files"]]
    conv["full_text"] = "\n\n".join(texts)
    _rebuild_collection(conv)
    _save_state(sid)

    return {"file_name": file_name, "file_count": len(conv["files"])}


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
    if not conv.get("files"):
        return JSONResponse({"error": "请先上传文件"}, status_code=400)

    history = conv.get("messages", [])
    search_query = resolve_query(question, history)
    context, sources = retrieve_and_build_context(
        search_query, conv["full_text"], conv["collection"], conv["use_vector"]
    )
    if not context:
        return JSONResponse({"error": "未找到相关内容"}, status_code=400)

    conv["messages"].append({"role": "user", "content": question})
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
