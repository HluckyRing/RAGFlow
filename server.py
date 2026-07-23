import json
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from contextlib import asynccontextmanager

from src.config import logger
from src.loaders import load_file
from src.pdf_ingestion import split_text
from src.retrieval import init_vector_store, sanitize_collection_name
from src.llm import resolve_query, retrieve_and_build_context, stream_answer

sessions = {}

def _get_session():
    import threading
    tid = threading.current_thread().ident
    return tid


app = FastAPI(title="RAGFlow")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/api/session")
async def create_session():
    sid = uuid.uuid4().hex[:16]
    sessions[sid] = {"kbs": {}, "active_kb": None}
    return {"session_id": sid}


@app.get("/api/kbs")
async def list_kbs(session_id: str):
    sess = sessions.get(session_id, {})
    kbs = [{"name": n, "active": n == sess.get("active_kb")} for n in sess.get("kbs", {}).keys()]
    return {"kbs": kbs}


@app.post("/api/kbs")
async def create_kb(file: UploadFile = File(...), session_id: str = ""):
    if not session_id:
        session_id = uuid.uuid4().hex[:16]
        sessions[session_id] = {"kbs": {}, "active_kb": None}
    sess = sessions[session_id]

    try:
        full_text = load_file(file)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not full_text.strip():
        return JSONResponse({"error": "文件内容为空"}, status_code=400)

    kb_name = file.filename
    cname = sanitize_collection_name(kb_name)
    collection, use_vector = init_vector_store(cname)
    if collection and collection.count() == 0:
        chunks = split_text(full_text)
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"c{i}"])

    sess["kbs"][kb_name] = {
        "full_text": full_text,
        "collection": collection,
        "use_vector": use_vector,
        "messages": [],
    }
    sess["active_kb"] = kb_name

    return {"name": kb_name, "session_id": session_id, "chars": len(full_text)}


@app.delete("/api/kbs/{kb_name}")
async def delete_kb(kb_name: str, session_id: str):
    sess = sessions.get(session_id)
    if not sess or kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    del sess["kbs"][kb_name]
    if sess["active_kb"] == kb_name:
        sess["active_kb"] = next(iter(sess["kbs"]), None) if sess["kbs"] else None
    return {"ok": True}


@app.post("/api/kbs/switch")
async def switch_kb(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    kb_name = data.get("kb_name", "")
    sess = sessions.get(session_id)
    if not sess or kb_name not in sess.get("kbs", {}):
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    sess["active_kb"] = kb_name
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

    async def generate():
        full_answer = ""
        yield f"data: {json.dumps({'type': 'sources', 'sources': [s[:300] for s in sources]})}\n\n"
        for token in stream_answer(question, context, history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        kb["messages"].append({"role": "assistant", "content": full_answer})
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
