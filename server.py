import json
import uuid
import os
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


def _cname(kb_name):
    return sanitize_collection_name(kb_name)


def _save_state(session_id):
    sess = sessions.get(session_id)
    if not sess:
        return
    data = {"kbs": {}, "active_kb": sess.get("active_kb")}
    for name, kb in sess.get("kbs", {}).items():
        data["kbs"][name] = {
            "collection_name": kb.get("_cname", _cname(name)),
            "files": kb["files"],
            "full_text": kb["full_text"],
            "use_vector": kb["use_vector"],
            "messages": kb["messages"],
        }
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
        sess = {"kbs": {}, "active_kb": data.get("active_kb")}
        for name, kb_data in data.get("kbs", {}).items():
            cname = kb_data["collection_name"]
            collection, use_vector = init_vector_store(cname)
            sess["kbs"][name] = {
                "files": kb_data.get("files", []),
                "full_text": kb_data.get("full_text", ""),
                "collection": collection,
                "use_vector": use_vector,
                "messages": kb_data.get("messages", []),
                "_cname": cname,
            }
        if sess["kbs"]:
            sessions[session_id] = sess
            logger.info("已恢复 %d 个知识库", len(sess["kbs"]))
    except Exception as e:
        logger.warning("恢复状态失败: %s", e)


def _add_text_to_kb(kb, new_text):
    """Append text to KB, re-index if needed. Returns success."""
    kb["full_text"] += "\n\n" + new_text
    collection = kb.get("collection")
    if collection:
        new_chunks = split_text(new_text)
        offset = collection.count()
        for i, chunk in enumerate(new_chunks):
            collection.add(documents=[chunk], ids=[f"c{offset + i}"])
        logger.info("已追加 %d 个文本块", len(new_chunks))


def _register_kb(session_id, kb_name, file_text, filename):
    """Create or get KB, optionally add file text. Returns kb dict and is_new."""
    sess = sessions[session_id]
    if kb_name in sess["kbs"]:
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
        "messages": [],
        "_cname": cname,
    }
    if collection and file_text and collection.count() == 0:
        chunks = split_text(file_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])
    sess["kbs"][kb_name] = kb
    _save_state(session_id)
    return kb, True


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/session")
async def create_session():
    sid = uuid.uuid4().hex[:16]
    sessions[sid] = {"kbs": {}, "active_kb": None}
    _load_state(sid)
    return {"session_id": sid}


@app.get("/api/kbs")
async def list_kbs(session_id: str):
    sess = sessions.get(session_id, {})
    result = []
    for name, kb in sess.get("kbs", {}).items():
        result.append({
            "name": name,
            "active": name == sess.get("active_kb"),
            "file_count": len(kb.get("files", [])),
            "files": kb.get("files", []),
            "chars": len(kb.get("full_text", "")),
        })
    return {"kbs": result}


@app.post("/api/kbs")
async def create_kb(file: UploadFile = File(...), session_id: str = "", kb_name: str = ""):
    if not session_id:
        session_id = uuid.uuid4().hex[:16]
        sessions[session_id] = {"kbs": {}, "active_kb": None}
        _load_state(session_id)
    sess = sessions[session_id]

    try:
        full_text = load_file(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not full_text.strip():
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    name = kb_name.strip() if kb_name.strip() else file.filename
    kb, is_new = _register_kb(session_id, name, full_text, file.filename)
    sess["active_kb"] = name
    _save_state(session_id)
    return {"name": name, "is_new": is_new, "session_id": session_id, "chars": len(full_text)}


@app.post("/api/kb-create")
async def create_empty_kb(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    kb_name = data.get("name", "").strip()
    if not kb_name:
        return JSONResponse({"error": "名称不能为空"}, status_code=400)

    sess = sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "会话不存在"}, status_code=400)
    if kb_name in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库已存在"}, status_code=400)

    _register_kb(session_id, kb_name, "", None)
    sess["active_kb"] = kb_name
    _save_state(session_id)
    return {"name": kb_name, "is_new": True}


@app.put("/api/kbs/{kb_name}")
async def rename_kb(kb_name: str, request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return JSONResponse({"error": "新名称不能为空"}, status_code=400)

    sess = sessions.get(session_id)
    if not sess or kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    if new_name in sess["kbs"]:
        return JSONResponse({"error": "名称已存在"}, status_code=400)

    kb = sess["kbs"].pop(kb_name)
    sess["kbs"][new_name] = kb
    if sess["active_kb"] == kb_name:
        sess["active_kb"] = new_name
    _save_state(session_id)
    return {"name": new_name}


@app.delete("/api/kbs/{kb_name}")
async def delete_kb(kb_name: str, session_id: str):
    sess = sessions.get(session_id)
    if not sess or kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    del sess["kbs"][kb_name]
    if sess["active_kb"] == kb_name:
        sess["active_kb"] = next(iter(sess["kbs"]), None) if sess["kbs"] else None
    _save_state(session_id)
    return {"ok": True}


@app.post("/api/kb-switch")
async def switch_kb(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    kb_name = data.get("kb_name", "")
    sess = sessions.get(session_id)
    if not sess or kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    sess["active_kb"] = kb_name
    _save_state(session_id)
    return {"active_kb": kb_name}


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")
    if not question or not session_id:
        return JSONResponse({"error": "缺少参数"}, status_code=400)

    sess = sessions.get(session_id)
    if not sess:
        return JSONResponse({"error": "会话不存在"}, status_code=400)

    active = sess.get("active_kb")
    kb = sess["kbs"].get(active) if active else None
    if not kb:
        return JSONResponse({"error": "请先上传文件"}, status_code=400)

    history = kb.get("messages", [])
    search_query = resolve_query(question, history)
    context, sources = retrieve_and_build_context(
        search_query, kb["full_text"], kb["collection"], kb["use_vector"]
    )
    if not context:
        return JSONResponse({"error": "未找到相关内容"}, status_code=400)

    kb["messages"].append({"role": "user", "content": question})
    _save_state(session_id)

    async def generate():
        full_answer = ""
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s[:300] for s in sources]})}\n\n"
        for token in stream_answer(question, context, history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        kb["messages"].append({"role": "assistant", "content": full_answer})
        _save_state(session_id)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/history")
async def get_history(session_id: str):
    sess = sessions.get(session_id)
    if not sess:
        return {"messages": []}
    active = sess.get("active_kb")
    kb = sess["kbs"].get(active) if active else None
    return {"messages": kb.get("messages", []) if kb else []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
