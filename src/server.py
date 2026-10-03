# -*- coding: utf-8 -*-
import json
import uuid
import os
import time
import threading
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

import src.state as state
from src.config import logger, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, HOST, PORT
from src.loaders import load_file
from src.pdf_ingestion import split_text
from src.text_utils import strip_surrogates
from src.retrieval import init_vector_store, sanitize_collection_name, drop_collection
from src.llm import resolve_query, retrieve_and_build_context, stream_answer, LLMStreamError

# session_id -> 状态 dict。与 state 模块的缓存共用同一个对象，改动会被 _save_state 落盘。
sessions = {}
_sessions_lock = threading.Lock()

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

app = FastAPI(title="RAGFlow")


def _conv_cname(conv_id):
    return sanitize_collection_name("conv_" + conv_id)


def _conv_full_text(conv):
    """对话全文 = 各文件文本拼接。这是派生值，不落盘（见 P1-9）。

    这里**不做空值兜底**：文件被删光时结果必须是空串，否则索引里会残留已删掉的
    正文。历史状态文件里「有 full_text 却没有 files」的兼容放在 _hydrate_collections。
    """
    return "\n\n".join(f.get("file_text", "") for f in (conv.get("files") or []))


def _sanitize_messages(messages):
    """落盘前洗一遍消息正文（客户端可以用 \\ud800 这类 JSON 转义直接塞进来）。"""
    out = []
    for m in messages:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            m = {**m, "content": strip_surrogates(m["content"])}
        out.append(m)
    return out


def _serializable(sess):
    """剥掉 collection / use_vector / _cname 这些不可序列化的运行时字段。

    顺带把正文洗干净：state 用 json.dump(ensure_ascii=False) 落盘，孤立代理字符会让
    整个保存抛 UnicodeEncodeError，而且该异常被 state 内部吞掉 —— 表现为「接口成功、
    状态没存」。这是落盘前的最后一道防线。
    """
    convs_out = {}
    for cid, c in sess.get("conversations", {}).items():
        convs_out[cid] = {
            "name": c.get("name", "对话"),
            "created_at": c.get("created_at", 0),
            "files": [{"file_name": f.get("file_name"),
                       "file_text": strip_surrogates(f.get("file_text", ""))}
                      for f in c.get("files", [])],
            "messages": _sanitize_messages(c.get("messages", [])),
            # 不落盘 full_text：它是 files 里各 file_text 的拼接，加载时由
            # _hydrate_collections 重新派生（见 P1-9）。
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
    """首次加载后为每个对话恢复 Chroma 句柄和运行时字段（只做一次）。

    会触发 embedding 模型加载，属于阻塞操作，调用方必须放进线程池。
    """
    for cid, cd in data.get("conversations", {}).items():
        cname = cd.get("collection_name") or _conv_cname(cid)
        collection, use_vector = init_vector_store(cname)
        cd["collection"] = collection
        cd["use_vector"] = use_vector
        cd["_cname"] = cname
        # full_text 不落盘，每次加载都从 files 重新拼，顺带修正历史文件里可能过期的值；
        # 只有「有 full_text 却没有 files」的旧格式才保留原值，免得把正文弄丢
        cd["full_text"] = _conv_full_text(cd) if cd.get("files") else cd.get("full_text", "")


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


def _append_message(sid, conv_id, message, truncate_to=None):
    """追加一条消息并落盘。同步实现，调用方统一用线程池执行。

    truncate_to 不为 None 时先把 messages 截到该长度：截断与追加在同一把会话锁内
    完成，重新生成/编辑重发靠它覆盖旧轮次，而不是往历史里再叠一份。

    没带 time 的消息在这里补服务器时间戳 —— 时间戳只活在前端内存里的话，
    刷新一次就没了。
    """
    with state.lock_for(sid):
        # 按 id 重新取一次：期间对话可能已被删除
        target = sessions.get(sid, {}).get("conversations", {}).get(conv_id)
        if target is None:
            return False
        msgs = target.setdefault("messages", [])
        if truncate_to is not None:
            del msgs[truncate_to:]
        message.setdefault("time", time.time())
        msgs.append(message)
        _save_state(sid)
        return True


def _commit_upload(sid, conv_id, uploads):
    """把一批上传内容并入会话、重建向量库并落盘。整块同步执行，交给线程池跑。

    uploads 是 [(file_name, file_text), ...]。**一次批量只重建一次索引** ——
    _rebuild_collection 是「清空 + 全量重写」，逐文件提交会让 N 个文件退化成
    O(N²) 次重复嵌入。
    embedding 推理和写盘都是阻塞操作，放在事件循环上会卡住所有请求。
    返回 None 表示对话不存在。
    """
    sess = sessions.get(sid)
    if sess is None:
        return None
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
            return None

        # 对话可能是通过 POST /api/conversations 建的（那时刻意没建库），
        # 所以这里要按需补建，不能只在「顺便新建对话」的分支里建。
        if conv.get("collection") is None:
            cname = conv.get("_cname") or _conv_cname(conv_id)
            conv["collection"], conv["use_vector"] = init_vector_store(cname)
            conv["_cname"] = cname

        names = []
        for file_name, file_text in uploads:
            names.append(file_name)
            # 上传文本进入会话数据的唯一入口，脏字符必须在这里就洗掉：
            # 只清洗 embedding 输入是不够的 —— state 用 json.dump(ensure_ascii=False)
            # 落盘，代理字符会让写盘那一步直接 UnicodeEncodeError（2026-10-03 的 500）。
            file_text = strip_surrogates(file_text)
            existing = next((f for f in conv.get("files", []) if f["file_name"] == file_name), None)
            if existing:
                existing["file_text"] = file_text
            else:
                conv.setdefault("files", []).append({"file_name": file_name, "file_text": file_text})

        conv["full_text"] = _conv_full_text(conv)
        _rebuild_collection(conv)
        _save_state(sid)

        return {
            "conv_id": conv_id, "file_names": names, "file_count": len(conv["files"]),
            "conv_name": conv["name"],
            "files": [{"file_name": f["file_name"]} for f in conv["files"]],
            "messages": conv.get("messages", []),
        }


def _remove_file(sid, conv_id, file_name):
    """从对话里移除一个文件、重建索引并落盘。

    返回 None 表示对话不存在，False 表示该对话里没有这个文件。
    """
    sess = sessions.get(sid)
    if sess is None:
        return None
    with state.lock_for(sid):
        conv = sess.get("conversations", {}).get(conv_id)
        if not conv:
            return None
        before = conv.get("files", [])
        after = [f for f in before if f.get("file_name") != file_name]
        if len(after) == len(before):
            return False
        conv["files"] = after
        conv["full_text"] = _conv_full_text(conv)
        _rebuild_collection(conv)
        _save_state(sid)
        return {
            "file_count": len(after),
            "files": [{"file_name": f["file_name"]} for f in after],
            "messages": conv.get("messages", []),
        }


def _rebuild_collection(conv):
    """Rebuild ChromaDB collection from all files in a conversation.

    顺序很讲究：**先把新 chunk 全部写成功，再删旧 id**。反过来的话，add 中途抛错
    （embedding 收到脏字符、模型挂了、磁盘满……）会把一份原本可用的索引清空 ——
    2026-10-03 那次 500 就真发生了：状态文件里正文还在，索引却空了。
    新 id 带一代随机前缀避免和旧 id 撞车；旧 id 留到新内容写成功之后才清理。
    """
    collection = conv.get("collection")
    if not collection:
        return
    try:
        old_ids = list(collection.get()["ids"])
    except Exception as e:
        logger.debug("读取 collection 现有 id 失败: %s", e)
        old_ids = []

    # 第二道防线：历史状态或别处来的文本也可能带脏字符，这里是进 embedding 的最后一关
    full_text = strip_surrogates(_conv_full_text(conv))
    new_ids = []
    if full_text:
        generation = uuid.uuid4().hex[:8]
        chunks = split_text(full_text)
        for i, chunk in enumerate(chunks):
            cid = f"{generation}_c{i}"
            collection.add(documents=[chunk], ids=[cid])
            new_ids.append(cid)

    # 全空时 stale 就是全部旧 id —— 也就是「删光文件必须把索引也清空」那条路径
    new_set = set(new_ids)
    stale = [i for i in old_ids if i not in new_set]
    if stale:
        try:
            collection.delete(ids=stale)
        except Exception as e:
            # 删不掉只是新旧 chunk 并存（检索可能召回两份），总比把索引弄丢好
            logger.warning("删除旧索引失败，新旧 chunk 会同时存在: %s", e)
    logger.info("索引已重建: 新增 %d 个 chunk，清理 %d 个旧 chunk", len(new_ids), len(stale))


# ── Routes ──
#
# 约定：所有可能阻塞的调用（文件解析、embedding 推理、LLM 网络请求、写盘）
# 一律通过 run_in_threadpool 执行，保证事件循环不被卡住。

@app.get("/", response_class=HTMLResponse)
async def index():
    def _read():
        with open(_INDEX_PATH, encoding="utf-8") as f:
            return f.read()

    return await run_in_threadpool(_read)


@app.get("/api/session")
async def create_session():
    sid, _ = await run_in_threadpool(_ensure_session, "")
    return {"session_id": sid}


@app.get("/api/state")
async def get_state(session_id: str):
    sid, sess = await run_in_threadpool(_ensure_session, session_id)
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
    sid, sess = await run_in_threadpool(_ensure_session, data.get("session_id", ""))
    cid = uuid.uuid4().hex[:12]
    name = data.get("name") or time.strftime("%m-%d %H:%M")
    cname = _conv_cname(cid)
    # 这里刻意不建 Chroma collection：既避免一次可能几秒的模型加载阻塞请求，
    # 也不会因为「建了对话却没传文件」而留下空集合。真正的建库推迟到首次上传。
    conv = {
        "id": cid, "name": name, "created_at": time.time(),
        "files": [], "full_text": "",
        "collection": None, "use_vector": False,
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
    sid, sess = await run_in_threadpool(_ensure_session, data.get("session_id", ""))
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        sess["active_conv"] = conv_id
        _save_state(sid)
    return {"active_conv": conv_id}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, request: Request):
    data = await request.json()
    sid, sess = await run_in_threadpool(_ensure_session, data.get("session_id", ""))
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        sess["conversations"][conv_id]["name"] = data.get("name", "对话")
        _save_state(sid)
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, session_id: str):
    sid, sess = await run_in_threadpool(_ensure_session, session_id)
    with state.lock_for(sid):
        if conv_id not in sess.get("conversations", {}):
            return JSONResponse({"error": "对话不存在"}, status_code=404)
        conv = sess["conversations"].pop(conv_id)
        if sess.get("active_conv") == conv_id:
            remaining = list(sess["conversations"].keys())
            sess["active_conv"] = remaining[0] if remaining else None
        _save_state(sid)
    # 集合也要删掉，否则 chroma_db 只增不减
    await run_in_threadpool(drop_collection, conv.get("_cname") or conv.get("collection_name"))
    return {"ok": True}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, session_id: str):
    sid, sess = await run_in_threadpool(_ensure_session, session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    return {
        "id": conv_id, "name": conv["name"],
        "created_at": conv.get("created_at", 0),
        "files": [{"file_name": f["file_name"]} for f in conv.get("files", [])],
        "messages": conv.get("messages", []),
    }


@app.delete("/api/conversations/{conv_id}/files/{file_name}")
async def delete_file(conv_id: str, file_name: str, session_id: str):
    """删除对话里的单个文件，并重建该对话的索引。"""
    sid, _ = await run_in_threadpool(_ensure_session, session_id)
    result = await run_in_threadpool(_remove_file, sid, conv_id, file_name)
    if result is None:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    if result is False:
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    return result


def _upload_size(file):
    """取上传文件的字节数。

    Starlette 会给 UploadFile 带 size；拿不到时（老版本 / 非 multipart 来源）
    退回 seek 量一次长度，保证上限判断不会因为缺字段而形同虚设。
    """
    size = getattr(file, "size", None)
    if size is not None:
        return size
    stream = getattr(file, "file", None)
    if stream is None or not hasattr(stream, "seek"):
        return None
    pos = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(pos)
    return size


@app.post("/api/upload")
async def upload_file(files: list[UploadFile] | None = File(None),
                      file: UploadFile | None = File(None),
                      session_id: str = Form(""), conv_id: str = Form("")):
    # files（复数，可一次多个）是首选；file（单数）保留兼容既有调用方
    uploads = ([file] if file is not None else []) + list(files or [])
    if not uploads:
        return JSONResponse({"error": "没有收到文件"}, status_code=400)

    # 上限按整批总字节数算，并挡在入口：解析大 PDF/DOCX 是全流程最贵的一步。
    # 只累计拿得到的大小 —— 不能因为某个文件量不出大小就整批放行。
    total = sum(s for s in (_upload_size(f) for f in uploads) if s is not None)
    if total > MAX_UPLOAD_BYTES:
        logger.warning("拒绝超限上传: %d 个文件共 %.1f MB (> %d MB)",
                       len(uploads), total / 1024 / 1024, MAX_UPLOAD_MB)
        return JSONResponse(
            {"error": f"本次上传过大：共 {total} 字节（{total / 1024 / 1024:.1f} MB），"
                      f"超过上限 {MAX_UPLOAD_MB} MB"},
            status_code=413,
        )

    sid, _ = await run_in_threadpool(_ensure_session, session_id)

    parsed = []
    for uf in uploads:
        name = uf.filename or "未命名文件"
        try:
            # 解析 PDF/DOCX/XLSX 是纯阻塞 I/O + CPU，必须移出事件循环
            text = await run_in_threadpool(load_file, uf)
        except Exception as e:
            return JSONResponse({"error": f"{name}: {e}"}, status_code=400)
        if not text.strip():
            return JSONResponse({"error": f"文件内容为空：{name}"}, status_code=400)
        parsed.append((name, text))

    result = await run_in_threadpool(_commit_upload, sid, conv_id, parsed)
    if result is None:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    return result


def _sse_event(payload):
    return f"data: {json.dumps(payload)}\n\n"


def _chat_stream(sid, conv_id, question, context, history):
    """同步生成器：把 stream_answer 的 token 转成 SSE，并在结束时落盘。

    Starlette 会用 iterate_in_threadpool 迭代它，因此 stream_answer 里等 LLM
    出 token 的阻塞不会占住事件循环。（写成 async 生成器反而会走 AsyncIterable
    分支、在事件循环上迭代，把整个服务卡住 —— 这正是改造前的问题。）

    前端点「停止生成」时这个生成器会被 close()、抛 GeneratorExit：已经吐给前端
    的半截回答必须落盘（否则刷新后凭空消失），并打上 stopped 标记，免得下次加载
    把一段半截回答当成完整回答。错误文案仍然只推给前端、绝不进历史。
    """
    full_answer = ""
    stopped = False
    try:
        try:
            for token in stream_answer(question, context, history):
                full_answer += token
                yield _sse_event({"type": "token", "content": token})
        except LLMStreamError as e:
            logger.warning("流式回答中断，错误文案不写入历史: %s", e)
            # 错误照样推给前端（前端不用改），但绝不写进 messages
            yield _sse_event({"type": "token", "content": "\n\n" + str(e)})
    except GeneratorExit:
        stopped = True
        raise
    finally:
        # 已生成的部分照常落盘，不丢用户已经看到的内容；空回答不落一条空的 assistant
        if full_answer.strip():
            entry = {"role": "assistant", "content": full_answer}
            if stopped:
                entry["stopped"] = True
            _append_message(sid, conv_id, entry)
    yield _sse_event({"type": "done"})


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    session_id = data.get("session_id", "")
    conv_id = data.get("conv_id", "")
    truncate_to = data.get("truncate_to")
    if not question or not session_id:
        return JSONResponse({"error": "缺少参数"}, status_code=400)
    if truncate_to is not None and (isinstance(truncate_to, bool) or not isinstance(truncate_to, int)):
        return JSONResponse({"error": "truncate_to 必须是整数"}, status_code=400)

    sid, sess = await run_in_threadpool(_ensure_session, session_id)
    conv = sess.get("conversations", {}).get(conv_id)
    if not conv:
        return JSONResponse({"error": "对话不存在"}, status_code=400)
    if not conv.get("files"):
        return JSONResponse({"error": "请先上传文件"}, status_code=400)

    stored = list(conv.get("messages", []))
    if truncate_to is not None:
        if truncate_to < 0 or truncate_to > len(stored):
            return JSONResponse(
                {"error": f"truncate_to 超出消息范围（0-{len(stored)}）"}, status_code=400)
        stored = stored[:truncate_to]

    # 取快照：此刻【不含】当前提问，所以 resolve_query 里的上一轮就是 [-1]。
    # 重新生成/编辑重发走的就是这条路径：被截掉的那一轮既不在 history 里，
    # 落盘时也不会再叠一份。
    history = list(stored)
    search_query = resolve_query(question, history)
    file_count = len(conv.get("files", []))
    dynamic_top_k = max(10, file_count * 3)
    # HyDE 的 LLM 调用 + Chroma 查询 + embedding 推理全是阻塞的
    context, _ = await run_in_threadpool(
        retrieve_and_build_context, search_query, _conv_full_text(conv),
        conv["collection"], conv["use_vector"], dynamic_top_k,
    )
    if not context:
        return JSONResponse({"error": "未找到相关内容"}, status_code=400)

    # 检索成功才把提问写进历史（带 truncate_to 时同时覆盖旧轮次），
    # 所以「没检索到内容」的提问依然不会入库。
    if not await run_in_threadpool(
            _append_message, sid, conv_id, {"role": "user", "content": question}, truncate_to):
        return JSONResponse({"error": "对话不存在"}, status_code=400)

    return StreamingResponse(_chat_stream(sid, conv_id, question, context, history),
                             media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    logger.info("RAGFlow 启动: http://%s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
