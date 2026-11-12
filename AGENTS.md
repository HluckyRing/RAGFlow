# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

RAGFlow — 基于 RAG 架构的本地知识库问答系统，支持 PDF/Word/Excel/PPT/CSV/TXT/Markdown 多格式文件上传，通过 HyDE（假设性文档检索）提升召回率，FastAPI + SSE 流式输出，前端为零依赖原生 HTML/CSS/JS。

## 启动与开发

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（从项目根执行）
python server.py                # 根目录快捷入口（推荐）
python -m src.server            # 等价写法
# 注意：python src/server.py 不可用 —— 以脚本方式运行时 sys.path[0] 是 src/，
# import src.xxx 会因找不到项目根而报 ModuleNotFoundError
# 服务运行在 http://localhost:8080
```

测试：`pip install -r requirements-dev.txt` 后执行 `pytest tests -v`（覆盖状态隔离、并发保存、旧数据迁移、路径穿越防护）。无 lint/format 配置。 前端那几个纯函数另有**真渲染**测试：`tests/test_frontend_render.py` 用 Node 内置 `vm`跑 `index.html` 里的真实函数源码 + 一个极小 DOM 替身（不引 npm 依赖、不装 headless 浏览器），本机没有 `node` 时整组自动 skip。
仍建议手动回归：启动服务 → 浏览器打开 `http://localhost:8080` → 上传文件 → 提问。

## 架构

**数据流**: 上传文件 → `loaders.py` 提取文本 → `pdf_ingestion.py` 按段落切片 → ChromaDB 向量索引 → 用户提问 → `retrieval.py` HyDE 检索（LLM 生成假设答案再查向量库）→ `llm.py` 流式生成答案（SSE）

**其他目录**:

- `legacy/` — 历史版本归档（早期控制台版 RAG 实现）
- `data/` — 示例文档（中文考试题库 PDF）
- `chroma_db/` — ChromaDB 持久化存储（gitignore）

**模块职责**:

| 文件 | 职责 |
| --- | --- |
| `src/config.py` | 环境变量加载、OpenAI 客户端初始化、日志配置、全局常量（CHUNK_SIZE, TOP_K 等） |
| `src/loaders.py` | 多格式文件加载器，通过 `LOADERS` 字典按扩展名分发，`load_file()` 为统一入口；`_decode_bytes()` 做多编码回退（utf-8-sig/utf-8/gbk/gb2312/big5/latin-1） |
| `src/pdf_ingestion.py` | `split_text()` — 按段落切分文本，支持 chunk_size/overlap 可配 |
| `src/text_utils.py` | 中文文本轻量处理：`strip_question_words()` 剥离疑问词，`content_phrases()` 抽实义片段（指代消解用），`extract_terms()` 生成 2/3-gram（关键词检索用） |
| `src/retrieval.py` | ChromaDB 向量存储管理 + `hyde_retrieve()` 核心检索（双路融合 → 距离闸门 → 失败降级关键词匹配） |
| `src/llm.py` | `resolve_query()` 指代消解（短问题 + 代词时拼接上文实体）+ `stream_answer()` SSE 流式生成 |
| `src/prompts.py` | HyDE 和 QA 两套 system prompt 模板 |
| `src/server.py` | FastAPI 应用：会话/对话 CRUD、文件上传、SSE 聊天接口 |
| `src/state.py` | 会话状态持久化：每个 session_id 一个 JSON 文件（`state/`），进程内缓存 + 每会话锁 + 原子写 + 旧数据迁移 |

**关键设计决策**:
- **HyDE 检索**: 先让 LLM 生成"假设性答案"，用这个答案文本去做向量匹配，比直接用问题检索召回率更高
- **工程降级链**: 向量检索失败 → 自动切换关键词匹配（中文 2/3-gram + 英文词打分，`sqrt(长度)` 归一化）。一条都没命中时返回空，由上层提示「未找到相关内容」，不拿无关段落喂模型。降级时**不再**调用 HyDE
- **HyDE 按需调用**: 只有 `use_vector` 为真时才生成假设性答案；关键词检索用原始问题（HyDE 生成的长段落不适合做词频匹配）
- **状态持久化**: 会话和对话数据按 session_id 隔离，存于 `state/<sha1(sid)>.json`。文件名由 sid 哈希派生，客户端传什么都无法写出 STATE_DIR（防路径穿越）。使用 atomic write（写唯一临时文件 → fsync → rename）
- **会话隔离**: 状态只在会话首次访问时从磁盘加载一次，之后以内存为准；同一会话的「改内存 + 落盘」由每会话 RLock 串行化。缓存和锁都在进程内，因此**只支持单 worker 部署**
- **旧数据迁移**: 首次遇到「客户端自带、服务端未见过的 sid」时，把旧版 `kb_state.json` 原子改名认领给它（保留为 `kb_state.json.migrated` 便于回滚）；`/api/session` 新发的 sid 不参与认领
- **多轮对话**: `resolve_query()` 在问题 < 15 字且含指代词（它/它们/这个/该/上述…，刻意不含裸「这」「那」；含「其他」「其中」等也不算）时，从上一轮用户消息提取实义片段拼到当前问题前。注意 `history` 是**不含当前提问**的快照，所以上一轮就是 `user_msgs[-1]`
- **ChromaDB**: 使用 PersistentClient 持久化到项目根下的 `chroma_db/`（`VECTOR_DB_PATH`，相对路径锚定项目根），每个对话独立 collection（命名 `kb_{name}_{hash}`）
- **不阻塞事件循环**: 所有可能阻塞的调用（文件解析、embedding 推理、LLM 网络请求、写盘）一律经 `run_in_threadpool` 执行。SSE 的 `_chat_stream()` 必须是**同步**生成器 —— Starlette 只对非 AsyncIterable 用 `iterate_in_threadpool` 迭代，写成 `async def` 反而会在事件循环上迭代、把全服务卡死
- **按需建向量库**: 建对话时不创建 Chroma collection，推迟到首次上传时在 `_commit_upload` 里补建（不论对话是 API 建的还是上传时顺带建的），避免「建了对话没传文件」留下空集合
- **Embedding 模型**: BAAI/bge-small-zh-v1.5，通过 `HF_ENDPOINT` 环境变量支持 HuggingFace 镜像
- **相关性阈值**: 向量结果按 l2 距离过滤（collection 未指定 `hnsw:space`，走 Chroma 默认的 l2；BGE 是归一化向量，距离 = 2 − 2cos）。默认 `MAX_DISTANCE=0.55` —— 实测本项目语料上相关查询 max≈0.47、无关查询 min≈0.62，取分离带中点。超阈值的块不进 context；全部超阈值则走关键词兜底。**换语料或换 embedding 模型必须重新标定**
- **流式失败不入历史**: `stream_answer` 失败时抛 `LLMStreamError`，而不是把错误文案当回答 yield 出去（否则会被写进 `messages`，下一轮又被当上下文喂回模型）。server 捕获后仍把错误推给前端，但只落盘**已生成的部分回答**，错误文案绝不入库
- **路径锚定项目根**: `config.PROJECT_ROOT` 是唯一的项目根定义，`resolve_path()` 把相对路径（含默认值）锚定到它。`VECTOR_DB_PATH` / `STATE_DIR` / `LEGACY_STATE_FILE` 都走它 —— 从任何目录启动都不会各自生出一份向量库或状态目录
- **状态文件不存派生字段**: 磁盘上只存 `files[].file_text`；`full_text` 由 `server._conv_full_text()` 在运行时拼接、加载时重新派生。旧文件里带 `full_text` 也能无损读，但新写入不再把正文存两遍
- **缺 API_KEY 只降级不崩**: `config.build_client()` 缺 key 时记一条中文 ERROR 并返回 `None`（不再让 openai SDK 抛英文异常把 import 阶段带崩）。此时向量/关键词检索照常，HyDE 跳过，问答抛 `LLMStreamError` 给前端中文提示
- **上传大小上限**: `MAX_UPLOAD_MB`（默认 20）在 `/api/upload` 入口按 `_upload_size()` 判断，**按整批总字节数**算，超限直接 413、不进入解析 —— 解析大 PDF/DOCX 是全流程最贵的一步
- **多文件上传与单文件删除**: 上传用复数 `files` 字段一次提交整批（兼容旧的单数 `file`），`_commit_upload(sid, conv_id, [(name, text)])` 一次只重建一次索引 —— `_rebuild_collection` 是全量重写，逐文件提交会退化成 O(N²) 次重复嵌入。`DELETE /api/conversations/{conv_id}/files/{file_name}` 删除单个文件并重建索引；**删光文件时派生 `full_text` 必须为空**，否则索引里会残留已删正文
- **文件预览只读抽取正文**: `GET /api/conversations/{conv_id}/files/{file_name}` 返回 `{file_name, file_text, char_count}`，读的是 `loaders` 抽取的纯文本（原件另存，见下一条）。只读：不落盘、不重建索引；按 `session_id` 会话隔离，对话/文件不存在分别 404 中文错误。前端「查看文件」每行加「预览」按钮，右侧抽屉展示正文（一律 `textContent`，绝不 `innerHTML`）；超过 `PREVIEW_MAX_CHARS=1000000` 字符截断并提示「仅显示前 N / 共 M 字符」
- **原文件预览要持久化原件**: 上传时把原字节写到 `UPLOAD_DIR`（默认项目根 `uploads/`；目录为 `sha1(session_id)/conv_id/<随机名><扩展名>`，`files[]` 记 `stored_name` 引用），`GET /api/conversations/{conv_id}/files/{file_name}/raw` 取原件：PDF/文本 inline 且带 `nosniff`，Office attachment；单文件删除/对话删除都要级联清理原件，重传同名文件替换旧原件。前端抽屉默认「原文件」、可切「抽取文本」：Markdown 用 toMd 渲染成 HTML（先转义 HTML）、DOCX 用 docx-preview（options 必须第 4 个参数、第 3 个 styleContainer 传 null）、XLSX 用 SheetJS 自己建表、PPTX 用 Apache-2.0 的 @aiden0z/pptx-renderer 浏览器 ESM bundle 动态 import；库都在 `src/templates/vendor/`、由白名单路由 `/vendor/{name}` 提供（**不是 CDN**，JSZip→docx-preview→SheetJS 顺序不可反），PPTX 渲染失败回退「下载原件」；升级前无原件的旧文件回退抽取文本并标注「原件未保存」
- **PPT 只支持 OOXML**: `.pptx`/`.pptm` 走 `python-pptx`（同名软降级标志 `HAS_PPTX`）；旧版二进制 `.ppt` 明确拒绝并提示「另存为 .pptx」。**不要把 `.ppt` 塞进 `LOADERS`** —— `load_file()` 报错里的「支持: …」是按 `LOADERS.keys()` 生成的，塞进去等于把不支持的格式列成支持的。抽取内容 = 文本框/占位符段落 + 表格（`" | "` 拼接）+ **组合形状递归下钻** + 演讲者备注（`[备注]` 前缀），页间用 `=== 第 N 页 ===` 分隔；**整页无文字时不输出页头**，纯图片 PPT 因此照旧落回「文件内容为空」的 400
- **孤立代理字符必须在入库前剥离**: pypdf 解析字体映射损坏的 PDF 会产出孤立代理字符（U+D800–U+DFFF），它能让 embedding 抛 `TypeError: TextEncodeInput must be Union[...]`，也能让 state 的 `json.dump(ensure_ascii=False)` 抛 `UnicodeEncodeError`（那一步被 state 的 try 包住，表现为**静默保存失败**，比 500 更隐蔽）。清洗共三道：`_commit_upload` 写进 `files[].file_text` 之前（保住内存与响应）、`_rebuild_collection` 进 embedding 之前（防御历史数据）、`_serializable` 落盘之前（连客户端塞进 `message.content` 的 JSON 转义也一起洗）；实现是 `text_utils.strip_surrogates()`，剥掉几个记一条 WARNING，不静默丢字符。已实测 **lxml 会直接拒绝代理字符**（XMLSyntaxError），所以 docx/xlsx/pptx 不是来源，txt/csv/md 的解码回退链也产生不了 —— 实际来源就是 pypdf
- **重建索引必须先加后删**: `_rebuild_collection` 先把新 chunk 全部写成功，再删旧 id（新 id 带随机代际前缀，避免和旧 id 撞车）。反过来「先 delete 再 add」时，add 一旦抛错（脏字符、模型挂了、磁盘满）就会把一份**原本可用的索引清空** —— 2026-10-03 的 500 就这样毁过一次索引：状态文件里正文还在，索引却空了。删旧 id 失败只记 WARNING：新旧并存顶多多召回一份重复，好过把索引弄丢
- **引用必须可核验，来源不许编造**: 块按「文件 + 页/工作表」切分（`pdf_ingestion.split_files`），来源 `{file_name, chunk_index, section}` 写进 Chroma `metadatas`，`retrieve_and_build_context` 给每块加 `【来源：…】`，`prompts.py` 要求模型只引用这个来源、不要自己数「根据背景第X段」。页码来自 loader 留下的结构标记（pdf/pptx 的 `=== 第 N 页 ===`、xlsx 的 `=== 工作表名 ===`），而且**每一节独立切片**——块永不跨页，否则贴任何页码都会说错（2026-07-24 那版 sources 引用就是因为不准确被删掉的）。没有元数据的老索引标「来源未标注」，绝不编造
- **前端显示引用的唯一数据来源是后端推的 `sources` 事件**: `/api/chat` 在第一个 token 之前推一条 `{"type":"sources","sources":[{file_name, section}]}`（内容由 `llm.source_list()` 去重、丢掉没有来源的），并写进 assistant 消息的 `sources` 字段（`_sanitize_messages` 用 `{**m}`，所以自动落盘），刷新后由 `renderChat()` 重新渲染。**前端不得自己推断来源**，没有来源就什么都不显示；一个 token 都没出来（纯报错）时不推 sources，免得给错误消息挂引用。文件名是用户可控的动态数据，只能用 `createElement` + `textContent` 写进 chip（静态守卫在 `tests/test_frontend.py`）
- **前端动态数据不进内联处理器**: 对话名/文件名一律经 `textContent` / `dataset` 落地，事件用 `addEventListener` 绑定。内联事件处理器是 HTML 属性、会被当代码解析，动态数据里一个单引号就能闭合字符串执行任意 JS（P1-15 的老写法）；`esc()` 也补了单引号转义作为纵深防御
- **前端主题与响应式**: 配色收敛为 CSS 变量，深色值挂在 `html[data-theme="dark"]`。`<head>` 里的内联脚本在 CSS 生效前定主题（避免首屏闪白）：没手动选过就跟随 `prefers-color-scheme`，手动切换后写入 `localStorage.ragflow_theme` 并长期优先。≤820px 时侧边栏改为固定抽屉 + 遮罩，`syncSidebar()` 在断点变化时同步开合
- **重新生成/编辑重发靠 truncate_to 覆盖旧轮次**: `/api/chat` 多了可选整数 `truncate_to`（`0..len(messages)`），语义是「先把 `messages` 截到该长度，再追加本次提问」；不传时行为与改造前完全一致，越界/非整数/布尔一律 400 中文提示。截断与追加在 `_append_message` 的**同一把会话锁内**完成，不会被并发写截到一半。重新生成不需要新接口：截到这条提问处 + 用同样的文字重发即可，服务端因此不会留下重复轮次
- **停止生成要落半截回答**: SSE 生成器 `_chat_stream()` 用 `try/finally` 落盘已生成的部分，客户端中断（生成器被 `close()` → `GeneratorExit`）时给消息标 `stopped=true`，前端显示「⏹ 已停止生成」；错误文案仍然绝不进 `messages`。抽成模块级函数是为了能直接测「客户端中途断开」这条路径
- **消息时间戳由服务端落盘**: `_append_message()` 给缺 `time` 的消息补服务器时间。时间戳只活在前端内存里的话，刷新就没了
- **默认只监听回环**: `HOST`（默认 `127.0.0.1`）/ `PORT`（默认 8080）统一定义在 `config.py`，两个可用入口（`python server.py`、`python -m src.server`）都复用它。**`python src/server.py` 不能当脚本跑**（`sys.path[0]` 会变成 `src/`，`import src.xxx` 直接 ModuleNotFoundError，只能作为模块被导入）。这个服务没有鉴权，要对外提供必须显式设 `HOST=0.0.0.0`，不要在代码里写死对外地址
- **归档脚本不关 TLS 校验**: `legacy/` 里曾用 `ssl._create_default_https_context = ssl._create_unverified_context` 全局关掉证书校验，已移除；`tests/test_security.py` 有静态守卫防止被写回来
- **孤儿 collection 清理**: `scripts/cleanup_orphan_collections.py` —— 默认干跑，要 `--apply` 才真删；只删名字以 `kb_conv_` 开头且不被任何状态文件引用的集合；如果一个被引用的集合都没扫到（通常是路径指错）就中止，只有 `--force` 能越过。状态来源含 `state/*.json` 与旧版 `kb_state.json(.migrated)`
- **重建索引按 conv_id 去重**: `scripts/reindex_all.py` —— 默认干跑，要 `--apply` 才真重建；一个状态文件都没扫到就中止，只有 `--force` 能越过。**同一个 conv_id 只重建一次**：`state/*.json` 会话文件先扫、优先保留，旧版 `kb_state.json(.migrated)` 只补 `state/` 里没有的对话 —— 集合名由 conv_id 派生，两份都重建就是让后跑的迁移前旧快照**覆盖线上索引**（真实数据里 `.migrated` 与 `state/` 就有两个同 conv_id、内容不同的对话）。被跳过的重复项打印 `[跳过]` 并记 WARNING
- **评测语料的结构标记不存二进制**: `scripts/eval_retrieval.py` 的语料目录里，带页标记的那份是文本 spec `tests/fixtures/eval_corpus/eval_pptx.json`（评测现场用 python-pptx 生成 pptx → 真实 `loaders.load_pptx` 抽文本），仓库不放 `.pptx`。标注用可选字段 `expect_sections`（如 `["第 3 页"]`）驱动 **section 归属正确率**；**标了页码而命中块没有 section 必须判 False**（判成「不知道」会让这条指标静默失效），并且要用保险丝钉住「标注的页码真的存在于该文件的切片里」。另外每条样本都记 `path`（`vector`/`keyword`）：**关键词兜底样本的分数是 2-gram 词面打分、不是向量距离**，看指标前先按它拆开（2026-10-06 实测 41 条里 7 条走兜底，且含全部负样本）
- **真实语料只记数字不留文件**: 真实语料基线用 `scripts/eval_real_corpus.py` —— 只读 `state/*.json` 的 `files[].file_text`、`chromadb.EphemeralClient` 临时库、不落文件、不打印正文，数字记进 `docs/retrieval-baseline-2026-10-06.md`。真实语料没有 gold，**不要**给它套 recall/MRR/section 那套指标（只会得到看起来像「检索坏了」的假数字）；「无关提问」只能用合成负样本当代理并标注清楚。默认 `--no-hyde`（免费可重复），`--hyde` 才花额度、且那才是线上口径。两个脚本都支持 `--fusion` / `--fusion-gate`，`eval_retrieval.py` 还有 `--hyde-cache <json>`：**对比不同开关时必须带上它**，否则两轮比的是大模型随机性（同一配置连跑两次 MRR 0.972 / 0.958）
- **推理模型的小 max_tokens 会把内容吃光**: `deepseek-flash` 是推理模型，`_generate_hyde` 写死的 `max_tokens=200` 会被 reasoning 全吃掉（实测 `finish_reason=length`、`reasoning_tokens=200`、`content` 空串），空内容又被 `or question` 悄悄兜成原问题 —— **HyDE 自模型切到 `deepseek-flash` 起一直是关着的**（2026-10-06 才发现），此前标着「HyDE 口径」的数字全是 no-hyde。修法：这次调用显式 `extra_body={"thinking": {"type": "disabled"}}`（HyDE 不需要推理链、cap 保持 200），**空内容必须记 WARNING**；`eval_real_corpus.py --hyde` 跑前有自检，生成文本与原问题相同就直接中止
- **可答性判断（默认关，改成让生成自己拒答）**: `src/answerability.py` 在检索出 context 之后、生成回答之前让模型判一次「这段材料能不能回答这个问题」，输出 `能：<原文依据>` 或 `不能`（`temperature=0`、`max_tokens=32`、关思考），判「不能」走「未找到相关内容」400 拒答路径。**但默认 `ANSWERABILITY_CHECK=0`**：2026-11-06 实测它每条提问固定 **+0.78 s**，而缩短判定输入并不省时间（top1 1.01 s / top2 0.92 s / 全长 0.78 s，延迟是 API 往返不是 prefill），用距离门槛跳过判定又会漏掉域内负样本（省 31/44 漏 3/8）；同时**生成那一次调用自己就会拒答** —— `QA_SYSTEM_PROMPT_TEMPLATE` 第 1 条本来就要求「背景没有答案就说背景中没有提到相关内容」，`scripts/measure_inline_abstention.py` 在 59 条上实测：域内负样本 **8/8**、远域 **2/2** 正确拒答，正样本 **0/36** 误拒，与独立判定分毫不差却**零额外调用**（生成中位 1.30 s，带判定是 ≈2.1 s）。所以默认走 inline 拒答；要「硬 400 + 完全不生成」的严格口径才设 `ANSWERABILITY_CHECK=1`。两条纪律保留：① 判定要**逼模型给出原文依据**（只答「能/不能」会按字面判，误杀措辞不同但同义的正样本 2/36 → 0/36）；② 判定故障一律 `None` + WARNING、按老行为作答（fail-open）。**拒答要能被看见**：inline 口径下模型按 `prompts.REFUSAL_MARKER`（`【无答案】`）开头作答，服务端在流式**开头缓冲判定**（标记可能被拆在两个 token 里：`【无答`+`案】`，`llm.refusal_state` 三态：True/False/None-还不知道），判成拒答就推 `{"type":"refused"}` 事件、剥掉标记、落盘 `refused: true`，前端只认事件/字段、绝不猜文案，渲染成「⚠️ 知识库里没有能回答这个问题的内容」标签。**但「查过什么」不丢**：`refused` 事件与落盘都带 `searched_files`（按**文件名**去重、取自真正进了 context 的块），前端渲染成「（已检索 N 个文件）」—— 只给计数、不给文件名，避免把某个文件名跟「答案出处」绑在一起**标记只表示「整条答不了」**：2026-11-08 实测发现模型会把「有一半答不了」的复合问题也整条打上标记（8 条部分可答样本里 5 条如此，正文其实答对了可答部分），于是提示词改成三档（完全没涉及→写标记；只答得了一部分→正常作答+另起一段说明缺口、**不许写标记**；能完整回答→正常回答），并划清「材料只在讲相关话题 ≠ 回答了用户问的那件事」。改后：过度拒答 5/8→**0/8**，答了可答部分+点明缺口 3/8→**8/8**，同时域内负样本 8/8、远域 2/2、正样本 0/36 一分不掉。部分可答样本在 fixture 里是 `kind: "partial"`，字段 `answer_terms`（可答事实，必须在语料里）与 `absent_terms`（不可答主题，必须不在语料里），两条都用测试盯住换了提示词必须用 `scripts/measure_inline_abstention.py` 复测「模型听不听话」（实测 10/10 条拒答都带标记、36 条正样本 0 误标）**拒答不挂来源**：sources 事件推迟到判定之后才推（token 本来就在缓冲，正常回答仍然「sources 先于第一个 token」到达），判成拒答则既不推也不落盘 —— 材料答不了这个问题，把来源贴给用户只会让人以为那就是答案出处。
- **复合问题的三种形状（2026-11-08 第二批）**: 一个问句里的几个子问题按**顺序**列进 `parts`（每个子问题只站一边：`answer_terms` 或 `absent_terms`），`shape` 四选一 —— `two_parts` / `three_parts` / `gap_in_middle`（不可答那问在**中间**）/ `followup`（带 `history` 的多轮补问）；`expect_refused: true` 表示整条本来就该拒答（那时 `answer_terms` 可以为空）。`load_queries` 只校验**结构**（`shape` 四选一、每个 part 恰好一侧、平铺字段必须与 `parts` 拍平逐字相同、`history` 的 role 成对、`expect_refused` 必须是布尔、`gap_in_middle` 的第 2 问必须是缺口），**与语料一致**（`answer_terms` 逐字在语料、`absent_terms` 逐字不在语料、子问题按问句先后顺序、至少一条 `followup` 的 `resolve_query()` 返回值 ≠ 原问句、至少 4 条样本有 ≥2 个可答项否则「漏答可答项」那个桶永远是 0 = 从没测过）由 `tests/test_eval_retrieval.py` 钉住。`classify_partial` 从五桶扩到**九桶**：新增 `dropped_part_with_gap` / `dropped_part_silent`（点明了缺口却**吞掉某个本可答的子问题** —— 整条消息的桶看不见它，要 `part_coverage()` 逐项判）与 `refused_without_marker` / `answered_without_refusal`（该拒的样本：说了「没有」却没写标记 / 直接硬答）；「缺口在中间」另有 `post_gap_coverage()`（**缺口后面那一问还活着吗**），`summarize_partial()` 按 shape 拆开算，**该答的与该拒的绝不混进同一个分母**。多轮必须先过 `resolve_query`（`plan_search_queries`：检索用消解后的问句、生成用原问句 + `history`，与 `server.chat` 同一条路）。**两条量尺纪律**：① 词面命中前抹掉**装饰性**字符（Markdown 强调、各种引号、括号 —— 模型答对了却写 `保留 **30 天**` 会把锚点切成两半），但**不做同义改写**（「可以折算」≠「可折算」，太松比太脆更危险）；② `answer_terms` 只用**最短事实核**（≤12 字、语料逐字、正确回答绕不过去），`absent_terms` 给 2–3 个**同义主题词**（模型会写「参加人员」而不是问句里的「请哪些人参加」）。**判定/标注口径的改动可以离线重算**（分桶只依赖「回答文本 + 标注」，前提是 `--json` 存了 answer 原文），只有改提示词或检索才必须真跑。数字与复盘见 `docs/retrieval-baseline-2026-10-06.md` 的「复合问题的三种形状」一节
- **复合问题的更多形状与两种新量尺（2026-11-08 第三轮）**: partial 现在 24 条、7 种 `shape` —— 新增 `four_parts`（四问、正好 1 处缺口）/ `multi_gap`（四问、**≥2 处**缺口）/ `followup_partial`（多轮 **+** 部分可答：上一轮答了一半并点明缺口，本轮接着追问）；`followup` 里还有一条 **4 轮历史（8 条消息）**，专压 `stream_answer` 只取最近 5 条那根截断线。两种新量尺：① `gap_coverage()` 按子问题**逐处**算「缺口全点明」—— 扁平 `absent_terms` 是 `any()` 语义，多缺口样本只点明第一处也会判 `answered_with_gap`（这不是 bug，是结构决定的），所以多缺口/四问的漏答率**只能按 `parts` 逐问算**；② **该拒的样本（`expect_refused`）不计入缺口分母** —— 正确的整条拒答本来就不复述主题词，拿它算「缺口没点明」会冤枉正确行为。缺口主题词两条纪律（都有测试钉住）：**每处缺口至少一个 ≤3 字的裸主题核**（`期限`/`天数`/`境外`/`燃料`/`加密`/`校验`/`参加`/`异地` —— 带修饰的长词会被「的 / 最长 / 可请」这类插入词切断，`病假天数` 匹配不上「病假的最长可请期限」）；**`病假工资` 一族**没有合法裸核（«工资» 在语料里，不能当缺口主题词），必须同时给 `病假期间工资` 与 `病假期间的工资` 两种写法（实测模型两种都会写）。真实语料上的复合问题走 `scripts/eval_compound_retrieval.py`（`state/*.json` 只读、gold 现场派生、命中=同片或同文件同节、单问基线没命中的子问**不进**合并覆盖率分母）：**覆盖@5 与覆盖@10 完全相同（+0.0pp）**，未覆盖子问的目标片名次是**双峰**的（第 1 附近或 20 名开外，6–10 名的一条都没有），所以「调 `TOP_K`」不是杠杆，方向是**每问分路检索 / query 分解**；该脚本量不到「装下之后答没答全」，那是 `measure_inline_abstention.py` 那条路。完整数字与边界见 `docs/retrieval-baseline-2026-10-06.md` 的「复合问题的更多形状」一节
- **每问分路检索的对照实验（2026-11-08 第四轮）**: `scripts/eval_compound_retrieval.py --route` 在**同一份 context 预算**下比三臂 —— `merged`（整句一次检索，现状）/ `interleave`（每问各查一次后**轮流分名额**：先每问第 1 名、再每问第 2 名……）/ `rrf`（每问各查一次后按名次 RRF，与线上 `retrieval._rrf_fuse` 同口径）。两条纪律（都有测试钉住）：**上下文预算必须一样**（三臂的 context 都不超过 budget 块 —— 分路臂要是能塞 3 倍块，赢的是 token 预算不是检索质量）、**分母必须一样**（都是「单问基线命中的子问」）；`merged` 臂由**新代码路径**重算并与原有「覆盖@k」逐位比对，不一致直接中止 —— 新旧数字能不能比不靠人眼。实测（no-hyde、seed 0、35 条样本 / 37 个可用子问）：5 块预算下覆盖 **45.9% → 81.1%（+35.1pp）**、样本级全覆盖 **44.0% → 84.0%**；**分路@2（54.1%）就打赢整句@5（45.9%）**，而整句@10 仍是 45.9%（`top_k` 那条路确实死了）；分配口径是**二阶因素**（预算 ≥ 子问数时两臂逐样本完全一致，预算紧张时轮流分名额更稳：2 块时 54.1% vs 48.6%）。**这是「上界」不是线上数字**：拆问用的是 gold 子问题，线上还缺一次 query 分解（多一次 LLM 调用，拆错率与延迟没测，那才是这条路线真正的门槛）；而且分母本身由同一把检索器定义，所以分路臂在严格分母上做到 100% 是**结构性的** —— 有信息量的数字是「29 条单独检索找得到的子问，整句合并丢掉了 13 条（44.8%）」。同轮修了两把量尺：① **「锚点可见」这列会跨文件通胀**（分路臂 100% 里有 7 处是从**别的文件**捞到的同一句模板话 —— 锚点允许全语料出现 ≤2 次，而分路臂每次都拿带锚点的问句去查），结账只看身份口径，另记 `same_file_anchor`（事实在同文件**邻居片**里 —— 切片重叠造成的口径盲区）与 `strict_cross_file`；② **样本级全覆盖的分母错了**：原来除以全部 35 条样本，其中 10 条一个可用子问都没有，改成只数有可用子问的 25 条后，上一轮的 31.4% 在同口径下是 44.0%。边界：装下 ≠ 答对；成本只记检索次数（1.00 → 2.69 次/样本），没有端到端延迟；单配置单种子；兜底路径 0 条。完整数字与边界见 `docs/retrieval-baseline-2026-10-06.md` 的「每问分路检索的对照实验」一节
- **拆问（query 分解）本身（2026-11-08 第五轮）**: `src/decompose.py` 把复合问句拆成能各自检索的子问题，是「每问分路检索」上线前缺的那一步（上一轮 81.1% 用的是 gold 子问）。三条纪律：**只拆出一条时返回原问句**（模型的同义改写只会让检索变差且没有收益，单问提问因此完全不受影响）；**任何故障都退回 `[question]` + WARNING**（缺 key / 抛错 / 空内容 / 解析不出 —— 行为与改造前一致，绝不退成拒答，这是 HyDE 静默失效那次的教训）；**关思考、小预算**（推理模型不关思考会把 `max_tokens=300` 吃光、content 返回空串）。入口有一道廉价闸门 `looks_compound()`（≥2 个问号 / 「还有、以及」这类连接词 / 逗号分隔出 ≥2 段带疑问词），**漏判比误判贵**：误判只白花一次约 0.66 s 的调用（模型照样只返回一条），漏判是整套功能对那类提问静默失效 —— 第一版就漏了省略式追问（「…，境外城市呢？」），把「呢」收进疑问词后 18/18、误判仍是 1/59，调用率 22.9%。输出格式定成 **JSON 数组**（`parse_parts` 兼容代码围栏 / 编号 / 项目符号 / 每行一问的退路，去重并按 `MAX_PARTS=6` 截断）。**拆问质量不按「拆得像不像 gold」量**（词面尺子是本项目栽过三次的坑），只量机械可核的：闸门召回、调用次数、延迟、条数、误拆、解析失败；**真口径是端到端覆盖** —— `scripts/eval_decomposition.py`（结构）与 `scripts/eval_compound_retrieval.py --route decompose`（拿模型拆出来的问句真去检索，看 gold 目标片还在不在）。实测：闸门召回 18/18、单问误拆 0/12、多轮样本 0 触发、解析失败 0/19、延迟中位 0.66 s；端到端三种拼法（`--join-style` qmark/comma/runon）分别是 78.4% / 81.1% / 83.8%，与 gold 分路（81.1%）差在噪声内，平均拆出 2.69 问且条数与 gold 逐条相同。边界：真语料那边的复合问句是模板拼的（边界好认），真实用户复合问句没测；延迟是单点测量；拆漏 3–5 条里混着检索波动。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「拆问（query 分解）本身」一节
- **拆问接进线上（2026-11-08 第六轮）**: `DECOMPOSE_ENABLED`（**默认 0**，与双路融合 / reranker 同一个约定：过了延迟线才谈默认开）。`server.chat` → `_decomposer()`（关着就是 `None`）→ `llm.retrieve_and_build_context(..., decompose=...)` → `llm._retrieve_hits()`，三条边界：① 开关关 / 只拆出一条 → 走**整句检索**（HyDE + 双路融合 + intersect 闸门），逐位不变、连闸门都不跑；② 拆成 ≥2 问 → 每问**单探针**检索（`use_hyde=False`：原问句直接当探针，仍过 `MAX_DISTANCE` 闸门），再按名额轮流合并、预算 = `dynamic_top_k`；③ 分路一条都没捞到 → 退回整句，绝不让一次拆问把问答搞空。**分路各问刻意不调 HyDE**：每问都调会让延迟变成 n 倍（单次 0.7–1.3 s），而拆问只要 0.66 s；代价是丢掉 intersect 的第二票，但只有「已经拆成 ≥2 问」的提问才走这条路，单问提问的精度机制没动。**合并口径只有一份实现** `src/hits_merge.py`（`interleave_hits` / `hit_key`），线上与评测脚本共用 —— 口径写两份必然漂，漂了评测数字就不再说明线上行为。真实 uvicorn 冒烟（on/off 各 3 次）：复合问首字中位 **3.60 s（开）vs 4.71 s（关）**，打开反而更快（省掉一次 HyDE 调用，拆问 0.66 s 被抵消）；单问两列在噪声里（路径完全相同，有测试钉住「只拆出一条时不许改道」）。`sources` 仍先于第一个 token、能答的 `refused=False`、域内无答案仍 `refused` + `searched_files` 且不挂 sources —— 四条链路都验过。**为什么还默认关**：冒烟是单机每条 3 次的单点；「多轮 + 复合」时拆问看的是**未消解的**问句（拆出来的子问题不再回头消解指代），这条路径没样本；也没有并发压测（每问 n 次 embedding 的排队行为未知）。补测了线上预算 10 的口径：模型拆问 86.5%（`？` 分隔）/ 89.2%（「还有」连写），gold 分路 89.2%，整句仍 45.9%。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「拆问接进线上」一节
- **多轮 + 复合：省略式追问的指代路径（2026-11-08 第七轮）**: 第二轮是复合问句、前半是省略式追问（「那这个规定还有效吗？」—— 主语只在上一轮里）时，**光分路不补指代比现状更糟**（省略式那一半 25.0% → 12.5%：把这句话单独拿去检索等于把仅有的上下文也扔了）。三层做法：① `server.chat` 先 `resolve_query`（改造前就有）；② 拆问能看到历史（`decompose_question(question, history=...)`：最近 4 条 user/assistant 压成短文 + 一条「省略式追问要补成能单独检索的问句」规则，**先筛角色再截最近几条**，否则中间夹一条 system 消息会白占名额）；③ **拆出来的每条子问再过一遍 `resolve_query`**（纯 Python 不花 API）—— 实测 12.5% → **75.0%**，这是真正的杠杆。**而「拆问带历史」量到 0 收益**：两个变体逐条逐位相同，因为不带历史时模型把省略原样留下、随后被规则补全，带历史时模型自己补全，殊途同归 —— 保留它只能算**保险**（规则够不到：问句 ≥15 字 / 代词不在 `PRONOUNS` 表里 / 指代助手上一轮的回答），不是已证的收益，而且它偶尔会把锚点硬拼进句子。量尺口径：多轮样本整体覆盖率会骗人（自带问句那一半混进来），**省略式那一半必须单独算**；control 臂（同样拆问、只去掉逐条消解）用来证明「是谁把省略补全了」。脚本入口 `scripts/eval_compound_retrieval.py --followup N --followup-cache <json>`。边界：合成省略式只有一种写法、15 个可用子问、助手回答是占位文本；真实用户的多轮省略没测。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「多轮 + 复合：省略式追问的指代路径」一节
- **并发压测：每问 n 次 embedding 怎么排队（2026-11-08 第八轮）**: 两层量法 —— 进程内基准（`.agents/bench_concurrent_retrieval.py`，120 次调用/配置，单探针路径隔离 API 噪声）+ 真实 uvicorn 压测（`.agents/stress_chat.py`，每个并发档 6–8 个请求）。结论：**embedding 并发是排队不是并行** —— 8 并发只换来 **1.28–1.35×** 吞吐（1 次检索 80.9→103.6 次/秒；3 次检索 84.0→91.8），单次延迟从 12 ms 线性涨到 76–85 ms（同一个模型对象 + GIL，`run_in_threadpool` 只挪阻塞活、不多给 CPU）；**但它不是瓶颈**（8 并发下 3 问分路的检索账 ≈0.26 s，同请求总时长 ≈3.9 s，大头仍是 LLM 往返）；**事件循环没被堵**（8 个并发请求 4.2 s 完成，串行要 17 s）。开关打开**在每个并发档都不慢**（串行档 17.0 s vs 28.1 s —— 分路省掉的那次 HyDE 调用比拆问更贵；首字 p95 3.56 s vs 5.15 s），20 次请求**零错误、零误拒**、`sources` 20/20 先于第一个 token、`done` 全在（并发下最容易静默坏掉的就是事件顺序与半截流，所以专门数了）。边界：并发只到 8（anyio 线程池默认 40）、语料只有 99 块（真实大对话 1199 块，查询部分会变慢）、每次 6–8 个请求不是 soak、n 更大时的放大没测、延迟受 API 波动影响只能看量级。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「并发压测」一节
- **`DECOMPOSE_ENABLED` 默认改成 1（2026-11-08 第九轮定案）**: 判据**跑之前写进脚本**（`.agents/decide_decompose_default.py`），四道门全过才改默认：① 质量门——复合问「两件事都答到」不低于现状；② 拒答门——域外复合问 100% 拒答；③ 延迟门——首字中位增量 ≤ +0.5 s 且 p95 增量 ≤ +1.0 s；④ 安全门——零错误 + `sources` 先于首 token + 拒答不挂来源 + 单问不误伤。方法：**ON/OFF 两个服务同时起、同一条问题交替发**（配对比，压掉 API 抖动；分开跑两轮会把抖动当成开关的作用）。实测 52 次请求：复合问答全 **87.5% → 100%**、域外复合问 4/4 拒答、首字中位 **−0.55 s**、p95 **−1.01 s**、零错误、`sources` 22/22、拒答挂来源 0 → **默认值改为 1**。**第一轮踩到两个测量坑**（都不是模型的错）：① **前置拒答不是 `refused` 事件** —— 检索全空时服务端走 HTTP 400 + 中文错误、根本不进 SSE，判分要认「400 且含『未找到相关内容』= 正确拒答」，否则会把 4 个正确拒答记成 4 个错误请求；② **语料太小质量门是空转** —— `dynamic_top_k = max(10, 文件数×3)`，3 个文件 6 块时预算把全库装下，两臂都 100%（零信息）；要 4 文件 × 8 节 ≈32 块（预算 12）才卡得住。定案后两件事：**关掉仍完全可用**（`=0` 连闸门都不跑、不花 API，行为与改造前逐位一致）；**测试必须自己关掉它** —— 默认变 1 后 `test_chat_flow` 那类用例会顺着 `server._decomposer()` 真去调大模型，已按 reranker 同款加 autouse 隔离（`tests/conftest.py::_never_call_the_real_decomposer`）。边界：52 次请求 / 质量门分母只有 16 个复合问请求；只一种语料规模；拒答门只有 2 条题目；延迟 −0.55 s 的来源是「省一次 HyDE > 多一次拆问」，换机器可能反转。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「把 `DECOMPOSE_ENABLED` 默认改成 1」一节
- **真实写法的复合问句：模板问句把拆问测成了送分题（2026-11-09 第十轮）**: 前九轮的复合问句**全是模板拼的**（子问题恒为 `材料里「<锚点>」讲的是什么`，而锚点是目标正文里逐字截出的 8–14 字窗口）—— 等于**把答案用词预先塞进了问句**，而 `DECOMPOSE_ENABLED` 已经默认 1，等于用一个「问题里含答案」的口径证明一个默认开着的功能。做法是给 `scripts/eval_compound_retrieval.py` 加 `--questions`：**问句来源换成手写**，`question` 是人话原话（带省略、指代、语序跳跃），`parts` 只给 gold 拆分 `ask` 与定位线索 `needle`，**`needle` 绝不进问句**（`question_leaks()` 数出来，本轮 0 处）；线索定位必须**唯一**，找不到或命中多块一律报错退出（gold 标错会让整张覆盖表失真，不许静默取第一个）；手写口径**不套** `validate_sample`（「子问题不许挤在同一文件」是给轮转拼样本用的，真实提问里同一章问两件事很常见），只保留「必须跨 ≥2 个文件」这条实验前提。实测（同一份 4 文件 / 324 块真实语料、预算 5、no-hyde、单 seed）：**模板口径** 整句@5 **40.0%** / 模型拆问 **100.0%** / 可用子问率 **36.4%**（55 个子问里 35 个连自己的目标片都检不到 top-1）；**真实写法口径** 整句@5 **63.3%** / 模型拆问 **96.7%** / 可用率 **71.4%**；gold 分路两臂均 100%，退回整句 0/20。三条结论：① **方向成立** —— 真实写法上拆问 96.7% ≫ 整句 63.3%（**+33.4pp**），不是模板造出来的；② **模板口径同时扭曲两端** —— 整句被低估（40.0% vs 63.3%）、拆问被高估（100.0% vs 96.7%），收益幅度从 +33.4pp 被夸大成 **+60pp**（所以「模板口径让数字虚高」这种笼统说法是错的）；③ **最隐蔽的失真是分母** —— 模板口径那张「分路 81.1%/100%」的表其实只建立在 20 个可评子问上，**以后报分路收益必须同时报可用子问率**。边界：手写问句是「模拟真实写法」不是真实用户日志，且 gold 子问是读过正文之后写的（单问基线 71.4% 偏高，拆问臂下界比真实场景乐观）；单语料单 seed 单模型、no-hyde；96.7% 与 100% 只差 1 条；**生成侧依旧空白**（覆盖率只说「装进了 context」，且 `measure_inline_abstention.py` 目前**没有拆问臂**，要测得先接线）。完整数字与边界见 `docs/retrieval-baseline-2026-10-06.md` 的「真实写法的复合问句」一节
- **生成侧：装下之后到底答没答全（2026-11-12 第十一轮）**: 第十轮把问句来源换成了真实写法，但量到的只是**检索覆盖**（目标片有没有进 context）；而 `measure_inline_abstention.py` **直接调 `retrieval.hyde_retrieve_hits_with_path`，压根没有拆问这条路** —— 于是 `DECOMPOSE_ENABLED=1` 上线之后，「拆问让复合问答得更全」在线上口径下没有任何证据。做法：给脚本加 `--decompose`，两臂**只差这一个开关** —— 新函数 `pick_hits(question, whole, decomposed, decompose)`（`decompose=None` 时连 `decomposed` 都不看一眼，一次多余调用都没有），而 `decomposed` 是**线上 `llm._retrieve_hits` 的薄包装**（「拆出 1 条退回整句」「分路全空退回整句」「每条子问过 `resolve_query`」全归那一份实现，**不在这层重写拆问逻辑**）；两臂共用同一 `TOP_K`、同一装配（`build_context`）、同一批样本、同一份 HyDE 缓存，拆问落 `--decompose-cache`。生成侧 gold 人工标 20 条真实写法复合问 / **42 个子问**，每子问一个「最短事实核」，校验全机器化：线索唯一定位目标片、事实核（或同义写法）真的在那片里、**主核不许出现在同一样本其它子问的目标片里**（否则「答了 A 没答 B」会被判成 B 也答到了）。一条标注约束：`load_queries` 要求**平铺 `answer_terms` 等于 `parts` 拍平结果**，而同义词只加在 `parts` 里会破坏这条一致性校验；平铺字段又是「全命中」语义、塞同义词会把答对判成漏答 —— 所以**每个子问只放一个事实核**。实测（真实语料 4 文件 / 324 块、预算 5）：整句臂 可答项覆盖 **28/42（66.7%）**、**全答到 7/20（35.0%）**、整条拒答 3/20、context 块数中位 **2**（1–4）；拆问臂 **40/42（95.2%）**、**全答到 18/20（90.0%）**、拒答 1/20、context 稳定 **5** 块；20 条全部拆开（退回整句 0 条）、平均 2.20 问。四条结论：① **生成侧的缺口是检索造成的，不是模型不肯答** —— 逐条读原文，整句臂的漏答几乎全是「参考背景里没有涉及文件目录/抖动/缺页率…」，三个整条拒答也都拒得对；② **整句臂连预算都填不满**（中位 2 块 / 预算 5，复合问句向量离任何单一片都远、被 `MAX_DISTANCE` 先砍掉），所以拆问赢的不只是「名额分配更均匀」；③ 用户看见的是**答全率**：覆盖率 66.7%→95.2%，「每个子问都答到」35%→**90%**；④ 剩下 2 处漏答是真的（模型自己写「先说明能回答的两问」把一问丢了 / 目标片没被检回来），不是判词问题。同轮修三处量尺（**假阳性一例**：r05「别让人白等」旧核 `让权等待` 被顺带命中，实为 `有限等待`；假阴性两例：r07 `行代表域`→`域`、r03 `磁盘访问`→`盘块`），修完两臂各 +1 子问，**两臂用同一份标注离线重算**（`--json` 存了回答原文，不重跑模型）——「判词比模型脆」又重演了一次。边界：**这是线上口径对照不是单变量消融**（关着走 HyDE+双路融合，开着时子问走单探针）；20 条 / 42 子问、单语料单 seed 单模型；装配刻意固定成 `build_context`（线上还多一层按块边界装配）；一子问一词仍可能误判同义改写；多轮生成侧、缺口点明率、并发下的生成侧都没测。完整数字与复现命令见 `docs/retrieval-baseline-2026-10-06.md` 的「生成侧：装下之后到底答没答全」一节
- **真实用户日志：12 条提问里没有一条复合问（2026-11-12 第十二轮）**: 前十一轮的复合问样本**全是自己造的**（先模板拼、后手写「模拟真实写法」），真实用户到底怎么问一直空白。新工具 `scripts/analyze_real_questions.py`（只读状态目录、不写盘、不判对错）盘干净了 `state/`：**只有 12 条真实提问 / 4 个对话**，时间戳 2026-07-24 与 2026-10-03 —— **全部早于本项目的检索改造**（10-06 基线、11-05 双路融合、11-08 拆问），所以那几条回答是旧代码产生的，既不能当基线也不能当「现状很差」的证据。四件事：① **复合问闸门 `looks_compound` 触发 0/12**（对照：第十轮手写复合问 20/20 触发）—— 闸门没误判真实提问，但也说明**默认开着的拆问在真实流量上一次都没被用到**，其收益在真实流量上仍未验证；② 真实流量的「多部分」需求是**枚举 / 汇总型**（12 条里 7 条带「列出/所有/分别/几个/哪些/总结/整理/主要内容」这类措辞），这类要的是**跨全部文件取样**，跟「按子问分路」是两种病；③ 真实流量极少（1 个用户，4 个对话，还有 2 条是同一问题问了两遍 —— 第一次没答好）；④ 同一批问题在**今天代码**的线上装配口径（`retrieve_and_build_context` + 真 HyDE + `dynamic_top_k=max(10,文件数×3)` + `MAX_CONTEXT_LENGTH=8000`，只数**装配后**有几个文件）下：2 文件 2/2、3 文件 3/3、5 文件 5/5、**13 文件 12/13**；缺的那一章是**检索排序没捞到**（只装配 14–17 块，离 8000 上限很远），不是装配截断 —— 所以**汇总型问题会静默漏掉整份文件，用户看不出来**。三条结论：① 「真实用户复合问句上的拆问质量」这个待办**当前无法用真实日志回答**（日志里没有复合问），要真回答只能等真实流量出现复合问或做用户研究，**不能再靠手写样本冒充**；② 拆问默认开在真实流量上是**零成本零收益**（闸门 0/12、一次 API 不花），状态安全，但**别把前十一轮的数字当成真实流量的收益**；③ 真实流量的下一个真杠杆是**枚举 / 汇总型的跨文件取样**，与排序、预算都无关。边界：12 条 / 1 个用户 / 两个时间点快照，不是分布；日志早于改造且没重跑生成，评价不了今天的回答质量；`ENUMERATION_MARKERS` 是措辞表、永远脆（「有哪些题型」同时算枚举与单问），只用于分组不判对错；**文件覆盖 ≠ 答得好**。完整数字与复现命令见 `docs/retrieval-baseline-2026-10-06.md` 的「真实用户日志：12 条提问里没有一条复合问」一节
- **跨文件取样：枚举 / 汇总型提问不再静默漏掉整份文件（2026-11-12 第十三轮）**: 治的是上一轮从真实日志量到的唯一真痛点 —— 13 份文件的语料问「该知识库的主要内容是什么」，装配后 context **只覆盖 12 份**，缺的那份是「按相关度 + `MAX_DISTANCE` 距离闸门」筛掉的，**模型和用户都不知道少了**。新增 `src/coverage.py` + `llm.ensure_file_coverage`：① `looks_enumerative()` 识别枚举/汇总措辞（列出/所有/全部/分别/几个/哪些/总结/整理/归纳/概述/主要内容…），**只决定「要不要多花一次补位查询」、不参与判对错**；② `sample_per_file()` 把已有命中**按文件轮流取样**（先每份 1 块再轮到第 2 块，文件间公平、文件内仍按相似度）；③ `missing_files()`/`coverage_report()` 数清差哪几份，补位只补缺的那几份、每份用 `retrieval.per_file_best()` 在**该文件内部**查一次（`where={"file_name": ...}`）且**刻意不过距离闸门**（离问句远的那份正是要补的）。三条硬边界：**开关关着/单文件/非枚举 → 原样返回**（一次查询都不多花，连闸门都不跑）、**块预算不变**（补位块排最前、总数仍截到 `top_k`）、**补位上限** `COVERAGE_BACKFILL_MAX=8`；拆问那条路**不叠**这套取样（已按子问分路分预算，叠着会把预算切碎两次）。实测（真实日志 12 条、真 HyDE）：13 文件的两个提问 **12/13 → 13/13**（17→18 块、14→15 块），**其余 10 条逐位不变**，总缺失 2→0；闸门触发率 **真实提问 7/12、手写复合问只 3/20**（不抢拆问的触发）；代价在 13 文件/1061 块语料上 **13 ms → 27 ms（+14 ms）**、只多 1 次补位查询。**默认仍关**（`COVERAGE_SAMPLING=0`，`.env.example` 已同步）：按项目约定新行为默认关，三道门只过了两道 —— 覆盖门 ✅、代价门 ✅、**质量门 ❌ 没跑**（覆盖变全 ≠ 答得更全，要证明得给这 6 条枚举型提问人工标 gold，本轮没做）。边界：只量文件覆盖不量答案质量；补位超过上限时**仍然静默**（只 INFO 日志，上下文里也没有「本次只覆盖 N/M 份」的提示，那要动 prompt 与 SSE 契约）；老索引无 metadatas 时 `where` 查不到东西；多轮与「同一份文件内部跨节取样」都没测。测试 +23 条（含变异检查：拆掉「关着就原样返回」测试如期变红），pytest 636 → 659。完整数字见 `docs/retrieval-baseline-2026-10-06.md` 的「跨文件取样」一节
- **负样本分两类，误召回要分开报**: `tests/fixtures/eval_queries.json` 的负样本带 `negative_class`：`far`（远域、与语料**零** 2/3-gram 重叠）考「会不会被阈值外的噪声拉进来」，`adjacent`（域内但语料里没有答案、**故意**有词面重叠）考「看起来像就作答」。两条相反的纪律都用测试钉住（far 必须零重叠、adjacent 必须有重叠，见 `tests/test_eval_retrieval.py`），`summarize` 也按类分开统计（`negative_false_positives_far` / `_adjacent`）。2026-11-05 补测（23 条负样本）暴露了原 5 条样本的乐观偏差：**远域误召回 11/15（73%）**、**域内 8/8（100%）**；逐条看 `path` —— **误召回 100% 走向量路径、没有一条是关键词兜底**，元凶是 HyDE 一定会编出「像语料」的假设文本。`intersect` 把远域压回 2/15（= no-hyde 水平），**但域内一条都救不了**：那需要可答性判断（重排分下限 / LLM 判定 context 里有没有答案 / 显式拒答），是当前最大的未解问题
- **双路融合（默认开，闸门 intersect）**: `retrieval._probe_queries` 把**原问题**和 **HyDE 文本**各当一次向量探针，一次 `query(query_texts=[...])` 拿回两条候选列表，再用 RRF（`_rrf_fuse`，k=`RRF_K`=60）合并名次。闸门 `FUSION_GATE` 两种口径都留着：`union`=任一路近就算近（召回优先，实测不降误召回）、`intersect`=**两路都得近**（**默认**，精度优先）。2026-11-05 实测：合成语料两批独立假设文本下 `intersect` 把误召回 2/5、1/5 都压到 **0/5**，recall@5 与文件/section 归属一分不掉，MRR 0.972→1.000 / 0.986；真实语料（1199 块）无关提问代理漏进 17/20→13/20、真实提问 11/12 不变；代价是每问多一次 embedding，**实测 +17 ms**。这条过了「误召回 0/5 且中位延迟 ≤1.5s」的默认开线（reranker 因 3.9s 没过）；**注意上面这两个 0/5 是 5 条容易负样本的结果**，负样本扩到 23 条后同一配置是远域 11/15→2/15、域内 8/8 不变（见上一条）。副作用要记住：闸门收窄后**更多样本落进关键词兜底**（3–4 条 → 7/41），那部分分数是 2-gram 词面分。HyDE 退化（写不出内容、假设文本=原问题）时自动只查一路，不会白花第二次 embedding。**开了重排时闸门不预砍候选**：候选池是融合排序后的全量，最终取舍交给跨编码器
- **关键词兜底的切片走缓存**: `retrieval._cached_pairs` 按「正文内容+切片参数」的 sha1 缓存 `(pairs, 小写正文)`，`PAIRS_CACHE_MAX=8` 条 LRU。真实语料 297KB/1199 块实测：每问重切 8.3 ms、重新 lower 1.0 ms，命中缓存后只剩 0.78 ms 的哈希。缓存键是内容派生的，所以上传/删除文件后**天然失效**，不需要谁去主动清；评测脚本临时改 `pdf_ingestion.CHUNK_SIZE` 也在键里，换切片长度不会吃旧缓存
- **reranker 默认关，因为 CPU 延迟实测 3.9s**: `src/rerank.py` 用 `BAAI/bge-reranker-base`（MIT、278M、1.11GB、**零新依赖**，`sentence-transformers` 自带 `CrossEncoder`）对 query×块 逐对打分：开了它就不再用 `MAX_DISTANCE` 先砍候选，而是取 `RERANK_CANDIDATES=20`、用**原问题**打分（不是 HyDE 文本，跨编码器的训练分布是 question→passage）、留 `score >= RERANK_MIN_SCORE=0.3` 的前 `RERANK_TOP_N=5`。**线上口径（HyDE）实测把负样本误召回 2/5 → 0/5，recall/MRR/归属一分不掉**，但 CPU 3.9 s/次 → 按「误召回 0/5 且延迟 ≤1.5s 才默认开」的约定，`RERANK_ENABLED` 默认 **0**。模型不可用/离线/打分抛错一律 WARNING + 退回距离阈值。国内下权重走 ModelScope（12.6 MB/s）而不是 HF（34–245 KB/s、`huggingface_hub` 会卡在 0 字节），拉到本地目录后把 `RERANK_MODEL` 指过去
- **审查清单与 CI**: 2026-10-01 那次代码审查的原始条目归档在 `docs/code-review-2026-10-01.md`（**编号连续**：P0 1–7 / P1 8–17 / P2 18–21；**不存在 P1-1~7 与 P1-18**），整改状态随代码更新；`.github/workflows/ci.yml` 在 push/PR 上跑 `pytest -q`
- **发版用标签触发 Release**: 流程 = 改 `pyproject.toml` 的 `version`（必须与新标签一致；新功能升次版本、修复升补丁）→ 本地 `python -m build --wheel` 确认 wheel 内容 + `pytest` 全绿 → 提交并推 `main` → `git tag -a vX.Y.Z -m "vX.Y.Z"` + `git push origin vX.Y.Z`。`.github/workflows/publish.yml` 收到 `v*` 标签会构建 wheel 并 `gh release create <tag> dist/*.whl --title "<tag>" --generate-notes`（用仓库自带的 `GITHUB_TOKEN`，本机 `gh` 未登录也能发）。**不要先在网页 UI 手动新建同名 Release**，否则工作流的 `gh release create` 会因同名已存在而失败。**`package-data` 必须覆盖 `templates/vendor/*` 与 `templates/vendor/aiden0z-pptx-renderer-licenses/*`**：漏了 wheel 里就只有 `index.html`、vendor 0 个文件，从 release 附件安装后原文件预览会缺 JSZip/docx-preview/SheetJS/PPTX bundle（2026-10-05 v1.1.0 发版实测踩到并修复）。发布后回读 `git ls-remote --tags origin` 与 GitHub releases API，确认标签/Release/wheel 附件都在，并抽查 wheel 内含 vendor 文件。

## 配置

项目根目录 `.env` 文件（不入 git，参考 `.env.example`）:

```ini
API_KEY="sk-xxx"
BASE_URL="https://api.deepseek.com"
MODEL_NAME="deepseek-flash"
# 可选
EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
CHUNK_SIZE=480
CHUNK_OVERLAP=100
TOP_K=10
MAX_CONTEXT_LENGTH=8000
MAX_DISTANCE=0.55
# 双路融合（原问题 + HyDE 各当一次探针，RRF 合并名次）
FUSION_ENABLED=1
# union=任一路近就算近；intersect=两路都得近（默认，误召回 0/5）
FUSION_GATE="intersect"
RRF_K=60
MAX_UPLOAD_MB=20
LLM_TIMEOUT=60
# 相对路径一律锚定项目根，可填绝对路径
VECTOR_DB_PATH="./chroma_db"
STATE_DIR="./state"
# 监听地址：默认只绑回环；要对外提供才改成 0.0.0.0
HOST="127.0.0.1"
PORT=8080
HF_ENDPOINT="https://hf-mirror.com"
```

LLM API 使用 OpenAI 兼容接口，`BASE_URL` + `API_KEY` 决定实际调用哪个服务。

<!-- superpowers-zh:begin (do not edit between these markers) -->
# Superpowers-ZH 中文增强版

本项目已安装 superpowers-zh 技能框架（20 个 skills），仅限本地使用，未入库。其他开发者 clone 后需单独安装。

## 核心规则

1. **收到任务时，先检查是否有匹配的 skill** — 哪怕只有 1% 的可能性也要检查
2. **设计先于编码** — 收到功能需求时，先用 brainstorming skill 做需求分析
3. **测试先于实现** — 写代码前先写测试（TDD）
4. **验证先于完成** — 声称完成前必须运行验证命令

## 可用 Skills

Skills 位于 `.Codex/skills/` 目录（不入 git），每个 skill 有独立的 `SKILL.md` 文件。

- **brainstorming**: 在任何创造性工作之前必须使用此技能——创建功能、构建组件、添加功能或修改行为。在实现之前先探索用户意图、需求和设计。
- **chinese-code-review**: 中文 review 沟通参考——话术模板、分级标注（必须修复/建议修改/仅供参考）、国内团队常见反模式应对。仅在用户显式 /chinese-code-review 时调用，不要根据上下文自动触发。
- **chinese-commit-conventions**: 中文 commit 与 changelog 配置参考——Conventional Commits 中文适配、commitlint/husky/commitizen 中文模板、conventional-changelog 中文配置。仅在用户显式 /chinese-commit-conventions 时调用，不要根据上下文自动触发。
- **chinese-documentation**: 中文文档排版参考——中英文空格、全半角标点、术语保留、链接格式、中文文案排版指北约定。仅在用户显式 /chinese-documentation 时调用，不要根据上下文自动触发。
- **chinese-git-workflow**: 国内 Git 平台配置参考——Gitee、Coding.net、极狐 GitLab、CNB 的 SSH/HTTPS/凭据/CI 接入差异与镜像同步配置。仅在用户显式 /chinese-git-workflow 时调用，不要根据上下文自动触发。
- **dispatching-parallel-agents**: 当面对 2 个以上可以独立进行、无共享状态或顺序依赖的任务时使用
- **executing-plans**: 当你有一份书面实现计划需要在单独的会话中执行，并设有审查检查点时使用
- **finishing-a-development-branch**: 当实现完成、所有测试通过、需要决定如何集成这份工作时使用
- **mcp-builder**: MCP 服务器构建方法论 — 系统化构建生产级 MCP 工具，让 AI 助手连接外部能力
- **receiving-code-review**: 收到代码审查反馈后、实施建议之前使用，尤其当反馈不明确或技术上有疑问时——需要技术严谨性和验证，而非敷衍附和或盲目执行
- **requesting-code-review**: 完成任务、实现重要功能或合并前使用，用于验证工作成果是否符合要求
- **subagent-driven-development**: 当在当前会话中执行包含独立任务的实现计划时使用
- **systematic-debugging**: 遇到任何 bug、测试失败或异常行为时使用，在提出修复方案之前执行
- **test-driven-development**: 在实现任何功能或修复 bug 时使用，在编写实现代码之前
- **using-git-worktrees**: 当需要开始与当前工作区隔离的功能开发，或在执行实现计划之前使用——通过原生工具或 git worktree 回退机制确保隔离工作区存在
- **using-superpowers**: 在开始任何对话时使用——确立如何查找和使用技能，要求在任何响应（包括澄清性问题）之前调用 Skill 工具
- **verification-before-completion**: 在宣称工作完成、已修复或测试通过之前使用，在提交或创建 PR 之前——必须运行验证命令并确认输出后才能声称成功；始终用证据支撑断言
- **workflow-runner**: 在 Codex / OpenClaw / Cursor 中直接运行 agency-orchestrator YAML 工作流——无需 API key，使用当前会话的 LLM 作为执行引擎。当用户提供 .yaml 工作流文件或要求多角色协作完成任务时触发。
- **writing-plans**: 当你有规格说明或需求用于多步骤任务时使用，在动手写代码之前
- **writing-skills**: 当创建新技能、编辑现有技能或在部署前验证技能是否有效时使用

## 如何使用

当任务匹配某个 skill 时，使用 `Skill` 工具加载对应 skill 并严格遵循其流程。绝不要用 Read 工具读取 SKILL.md 文件。

如果你认为哪怕只有 1% 的可能性某个 skill 适用于你正在做的事情，你必须调用该 skill 检查。
<!-- superpowers-zh:end -->
