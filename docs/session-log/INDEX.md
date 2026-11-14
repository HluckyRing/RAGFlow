# docs/session-log/INDEX.md — 历史索引、现状长文与速查

> 本文件是 `MEMORY.md` 的细节层：MEMORY.md 只保留开工必读的现状/待办（受 dsh-memory 注入上限约束），
> 长文索引、历史快照、环境/提交/推送速查都放这里，按需再读。

## 👉 详细历史看这里（本文件只做索引，别在这里囤细节）

- **2026-10-06 检索优化全程**：`docs/session-log/2026-10-06-retrieval-optimization.md`
  —— 14 个提交、4 个「静默失效」真 bug、所有测量数字与决策理由、环境坑、验证纪律。
- **2026-11-05~07 检索质量第二轮**：`docs/session-log/2026-11-07-answerability-and-refusal.md`
  —— 双路融合（默认 intersect）、负样本 5→23 证伪旧结论、重排只救远域、可答性判断从「实现」到
  「证明多余」、`refused` 事件与「拒答不挂来源」。9 个提交、数字总表、踩过的 6 个坑。
- **2026-11-08 收尾三件**：`docs/session-log/2026-11-08-partial-answer-and-frontend-tests.md`
  —— 部分可答（过度拒答 5/8→0/8）、拒答报「已检索 N 个文件」、前端拒答标签的 Node 真渲染测试。
  6 个提交、坑又多了 6 条（命令行套 Python、CRLF、冒烟选问题、分类桶不够细、标注错、git stat 缓存）。
- **2026-11-08 复合问题的形状（两轮）**：`docs/session-log/2026-11-08-compound-question-shapes.md`
  —— 样本 8 → **24 条 / 7 种 shape**（两问 / 三问 / 缺口在中间 / 四问 / 缺口多处 / 多轮补问 /
  多轮+部分可答），判定口径五桶 → **九桶 + 三种逐处量尺**（逐项覆盖 / 缺口后一问存活 / 缺口全点明）；
  真实语料上首次量了复合检索覆盖：**覆盖@5 = 覆盖@10（45.9%，+0.0pp）→ 调 TOP_K 不是杠杆**。
  8 个提交、`pytest` 437→533、坑又多了 8 条（其中「量尺比模型脆」重复了三次）。
- **2026-11-08 查询分解（六轮）**：`docs/session-log/2026-11-08-query-decomposition.md`
  —— 每问分路检索对照（**45.9% → 81.1%**，分路@2 就打赢整句@5）→ 修两把量尺 →
  拆问本身（闸门 18/18、延迟 0.66 s、条数与 gold 逐条相同）→ 接进线上 → 多轮+复合的
  指代（省略式那一半 12.5% → 75.0%，补全靠逐条 `resolve_query`）→ 并发压测（排队不是并行）
  → `DECOMPOSE_ENABLED` **默认改成 1**（四道门判据先写死，52 次请求配对实测）。
  12 个提交、`pytest` 533→606、坑八条（「量尺比模型脆」第四次、前置拒答不是事件、
  语料太小质量门空转）。
- **2026-11-12 真实写法 → 覆盖取样（八轮）**：`docs/session-log/2026-11-12-real-writing-and-coverage.md`
  —— 复合问句换成**手写真实写法**（收益从「+60pp」缩到 **+33.4pp**，模板口径同时低估整句、
  高估拆问）→ **生成侧**答全率 **35.0% → 90.0%** → 盘**真实用户日志**（12 条 / 1 用户 /
  复合问闸门 **0/12**，多部分是枚举型 7/12）→ **跨文件取样**（12/13 → 13/13 文件，+14 ms）→
  **质量门 no-op**（24/38 → 24/38，默认关）→ **导航页降权掉分**（24/38 → 20/38，查下来是
  gold 坏了）→ **同文件跨节取样**（「整理第一章」2/5 拒答 → **5/5 不拒答**，合计 28/38）→
  **两条风险加固**（启动 fail-closed + 闸门措辞表可插拔）。8 个提交、`pytest` 606 → **721**、
  坑九条（坏 gold 比坏特性贵、预算不对齐线上出假胜利、评测臂绕过线上入口）。**234 行**。
- **基线与结论的权威版本**：`docs/retrieval-baseline-2026-10-06.md`（**2262 行**，逐条实测表）
- **约定**：`AGENTS.md` / `CLAUDE.md`（两份必须逐字一致，当前均 187 行、仅 4 行工具相关文本不同
  = `git diff --no-index -U0` 正好 8 行差异）；
  审查清单归档 `docs/code-review-2026-10-01.md`（P0 1–7 / P1 8–17 / P2 18–21，编号连续）。

## 现状长文（2026-11-14 快照，迁自 MEMORY.md）

> **本节是快照，不是当前状态**：工作记忆（`MEMORY.md`）按约定不入库，所以仓库里没有
> 「当前状态」文件 —— 最新状态看 `git log`。第十~十七轮（`da8e736`..`d498395`）的数字见
> `docs/retrieval-baseline-2026-10-06.md` 对应小节（「真实写法的复合问句」「生成侧」
> 「真实用户日志」「跨文件取样」「排除导航页」「同文件内部跨节取样」「两条小风险的加固」）。

## 现状（2026-11-14 复核：8 个提交已推送，工作区同步）

- **已同步**：本地 `main` = `origin/main` = **`d498395`**（`git status -sb` 无 ahead/behind），
  其上另有 2026-11-14 的「覆盖取样拆开关」提交；全量 `pytest tests -q` → **726 passed**
  （2026-11-14 重跑，项目自带 `.venv`）；工作区只有刻意的 `?? .agents/`。
- **覆盖取样（两个开关都默认 0，2026-11-14 拆分）**：`COVERAGE_CROSS_FILE`（跨文件取样）在**线上
  预算**下是 **no-op**（实体覆盖 24/38 → 24/38）；导航页降权实现完整（`is_navigational` /
  `prefer_content`）但**无收益证据**，且暴露「该知识库的主要内容」那条 gold **无效**（章节级
  答案 vs 节级 gold 无干净解）。`COVERAGE_INTRA_FILE`（**同文件内部跨节取样**：
  `pick_target_file` / `sample_evenly` / `blocks_that_fit`）是这条线上**第一次质量门正面结果**：
  「帮我整理第一章的内容」2/5 拒答 → **5/5 不拒答**、其它 4 条逐位不变、+16 ms；合计 24/38 →
  28/38。仍默认关（有效样本只 1 条）。老开关 `COVERAGE_SAMPLING` 保留为「两个都开」的兼容别名
  （带 WARNING）；只开跨文件时点名章号的提问退回第十三~十五轮那条路，只开文件内时未点名章号的
  枚举型提问原样返回。
- **真实流量（第十二轮）**：`state/` 只有 **12 条真实提问 / 4 个对话 / 1 个用户**，全早于检索
  改造；复合问闸门 **0/12**（对照手写 20/20）→ 拆问在真实流量上**一次没被用到**；真实的多部分
  需求是**枚举 / 汇总型（7/12）**，与「按子问分路」是两种病。
- **安全 / 可维护性（第十七轮）**：`src/security.py::startup_error()` —— 对外监听且无
  `AUTH_TOKEN` → **拒绝启动**（退出码 2），回环 + 空 token 保持零摩擦；`/api/chat` 按**客户端 IP**
  限流 `CHAT_RATE_LIMIT=20` / `CHAT_RATE_WINDOW=60`（429 + `Retry-After`，判断在最前面）；
  四张闸门措辞表外置成 `src/gates/zh.py` + `GATE_PACK`（未知包退回 `zh` 并 WARNING、缺字段按
  空表），加 `compound`/`enumerative` 的 hit|miss 日志与「整批 0 触发」告警，再补**语言无关**的
  结构页判据（4-gram Jaccard ≥ 0.6 且跨 ≥3 份文件）。**仍无真身份体系**（拿到 `sid` 等于拿到
  全部数据）、限流是**进程内**的。

## 现状（2026-11-09 复核：归档后已推送，工作区同步）

- **已同步**：本地 `main` = 远端 `origin/main` = `6a63b7c`（`git ls-remote origin -h refs/heads/main`
  回读一致，`git status -sb` 无 ahead/behind）；全量 `pytest tests -q` → **606 passed**（2026-11-09 重跑，
  用项目自带 `.venv`，不沿用旧结果）；工作区只有刻意的 `?? .agents/`。
  归档已入库：`docs/session-log/2026-11-08-query-decomposition.md`（190 行）。
- 查询分解那轮的 12 个提交**归档时只在本地，归档后已推送**（别再按「未推送」找它们）。其后又加了
  两个纯文档提交：`c6eb9d8`（README 补英文摘要与实测数据）、`6a63b7c`
  （`docs/assets/social-preview-1280x640.png`，用于仓库 Settings → Social preview）。
- **拆问（query 分解）已上线且默认开**（`DECOMPOSE_ENABLED=1`，`src/config.py` + `.env.example`）：
  `server.chat` → `_decomposer()` → `llm.retrieve_and_build_context(..., decompose=..., history=...)`
  → `llm._retrieve_hits()`。三条边界：只拆出 1 条 / 开关关 → **整句检索逐位不变**（连闸门都不跑、
  一次 API 不花）；拆成 ≥2 问 → 每问**单探针**（不调 HyDE，否则延迟 n 倍）+ 轮流分名额合并，
  预算 `dynamic_top_k`；分路全空 → 退回整句。**合并口径只有一份实现**：`src/hits_merge.py`。
- **多轮 + 复合**：拆问吃 `history`（`decompose_question(question, history=...)`），而且**拆出来的
  每条子问再过一遍 `resolve_query`** —— 实测把省略式追问那一半从 12.5% 提到 **75.0%**；
  「拆问带历史」本身量到 **0 收益**（只能算保险，别写成提升）。
- **旧的默认值仍在**：`FUSION_ENABLED=1` + `FUSION_GATE=intersect`；`RERANK_ENABLED=0`
  （CPU 3.9 s/问，没过延迟线）；`ANSWERABILITY_CHECK=0`（生成自己会拒答，省 0.78 s）；
  `CHUNK_SIZE=480`、`MAX_DISTANCE=0.55`。

- **上一轮（复合问题形状那轮）结束时的状态**（历史快照，留作对照）：`53db0d1`
  （`fe040f1..53db0d1` 共 8 个提交）、全量 `pytest` → **533 passed**。
  归档已入库：`docs/session-log/2026-11-07-answerability-and-refusal.md`（219 行）、
  `docs/session-log/2026-11-08-partial-answer-and-frontend-tests.md`（133 行）与
  `docs/session-log/2026-11-08-compound-question-shapes.md`（212 行）。
- 检索链路：`CHUNK_SIZE=480`；`MAX_DISTANCE=0.55`（**保持不动**，四种口径下正负距离带都重叠）；
  HyDE **已真正生效**（此前被推理模型静默关掉，`24f3661` 修复）；reranker 已接入但
  **默认关闭**（`RERANK_ENABLED=0`；要开需 `RERANK_MODEL` 指本地权重目录）。
- **2026-11-05 新增：双路融合默认开 + `FUSION_GATE=intersect`**（原问题与 HyDE 各一路探针，
  RRF 合并名次）。最初 5 条负样本上「误召回 2/5、1/5 → 0/5」，**负样本扩到 23 条后真实口径是
  远域 11/15 → 2/15、域内 8/8 不变**（见最后一条）；recall@5 与归属始终不掉、MRR 0.972→1.000；
  真实语料无关代理漏进 17/20→**13/20**；代价 +17ms/问；**副作用**：闸门收窄后更多样本落进
  关键词兜底（41 条时 3–4 → 7；59 条时 4 → 15）。开关：`FUSION_ENABLED` / `FUSION_GATE`
  （`union` 召回优先）/ `RRF_K`。同轮加了关键词兜底的切片缓存
  （真实语料 8.3ms+1.0ms → 0.78ms，`PAIRS_CACHE_MAX=8`）。
- 评测入口：`scripts/eval_retrieval.py`（合成 **59 条 = 36 正 + 23 负**；`--hyde` / `--rerank` /
  `--calibrate` / `--fusion` / `--fusion-gate` / `--hyde-cache` / `--json`）与
  `scripts/eval_real_corpus.py`（真实语料，只读、只记数字、不落文件；也支持 `--fusion` / `--hyde-cache`）。
  **对比不同开关时必须带 `--hyde-cache`**，否则比的是大模型随机性。
- 本轮的原始证据留在 `.agents/`（未入库）：`eval-sample{1,2}-{off,union,intersect}.json`、
  `eval-neg23-{off,union,intersect,nohyde}.json`、`hyde-cache*.json`、
  `bench_split.py` / `bench_keyword.py`（切片与打分的耗时基准）、`add_negatives.py`
  （负样本扩样脚本，含词面自检）、`smoke_fusion.py`（真实 uvicorn 冒烟脚本）。
- **负样本已扩到 23 条（15 远域 + 8 域内无答案）**，原 5 条被证明太容易：同一配置下远域
  误召回 **11/15（73%）**（原报 2/5）、域内 **8/8（100%）**。逐条看 `path`：**误召回 100%
  走向量路径，没有一条是关键词兜底** → 元凶是 HyDE 必会编「像语料」的假设文本。
  `intersect` 闸门把远域压回 **2/15**（= no-hyde 水平），域内一条都救不了。
- **2026-11-07 新增：拒答可见（`refused` 事件）**。inline 拒答原来在前端是普通回答，现在改成
  **结构化契约**：`prompts.REFUSAL_MARKER`（`【无答案】`）+ 服务端流式开头缓冲判定
  （`llm.refusal_state` 三态，标记被拆在 `【无答`+`案】` 两个 token 里也要判出）+ 推
  `{"type":"refused"}` + 落盘 `refused: true` + 前端 `⚠️ 知识库里没有能回答这个问题的内容` 标签。
  换提示词后复测：10/10 条拒答都带标记、36 条正样本 0 误标（生成中位 1.72 s，属 API 波动）。
  真实 uvicorn 冒烟：能答 refused=False、不能答 refused=True 且正文无标记残留、落盘标记正确。
- **2026-11-08 新增：部分可答的提问（67 条 → fixture 里加 `kind: "partial"` 8 条）**。
  第一版提示词把「有一半答不了」的复合问题**整条**打上 `【无答案】`（5/8 过度拒答，正文其实答对了
  可答部分）→ 提示词改成三档（完全没涉及→写标记；只答得了一部分→正常作答+另说缺口、不许写标记；
  完整可答→正常回答），并划清「相关话题 ≠ 回答了问的那件事」。改后**四项指标同时达标**：
  过度拒答 5/8→**0/8**、答了可答部分+点明缺口 3/8→**8/8**、域内 8/8、远域 2/2、正样本 0/36。
  顺带修了一处**标注错误**（p08 的 `answer_terms` 写成了「分三段缓慢抬升」，但那句讲的是升温，
  问题问的是降温原因 → 改「产生应力」）。
- **2026-11-08：前端拒答标签有了真渲染测试**。`tests/test_frontend_render.py`：从 `index.html`
  抽出真实函数源码（花括号配对切 `buildRefusedTag`/`showRefusedTag`）→ Node 内置 `vm` + 极小 DOM
  替身跑一遍 → 断言 class/文本/「非数字不加计数」/重复调用不叠标签/拒绝 `innerHTML`。
  **不引 npm 依赖、不装浏览器**；没 `node` 时整组 skip（CI ubuntu runner 自带）。反向对照：临时改成
  `tag.innerHTML = text` → 4 条全红，还原后全绿。
- **2026-11-08：拒答时报「已检索 N 个文件」**。`refused` 事件与落盘都带 `searched_files`
  （按**文件名**去重、取自真正进了 context 的块），前端标签变成
  「⚠️ 知识库里没有能回答这个问题的内容（已检索 N 个文件）」—— 只给计数不给文件名，
  避免把某个文件跟「答案出处」绑在一起。冒烟：上传 2 文件问「恢复演练要请哪些人参加」→
  `searched_files=1`、无 sources 事件、落盘 `(refused, searched_files, has_sources)=(True, 1, False)`。
- **2026-11-07：拒答不再挂 sources**。`sources` 事件推迟到「判成正常回答」之后才推（token 本来就
  缓冲着，正常回答仍「sources 先于第一个 token」），判成拒答则既不推也不落盘 —— 材料答不了这个
  问题，贴来源只会让人以为那就是答案出处。冒烟核对 `落盘 (refused, has_sources) = [(False, True),
  (True, False)]`；用例 `test_refusal_carries_no_sources` +
  `test_normal_answer_still_pushes_sources_before_the_first_token`。
- **2026-11-06：那 0.78 s 省掉了 —— 默认改成「让生成自己拒答」**（`ANSWERABILITY_CHECK=0`）。
  三条路实测：① 用距离门槛跳过判定是死路（省 31/44 判定要漏 3/8 域内负样本，两者距离完全交织）；
  ② 缩短判定输入没用（top1 1.01 s / top2 0.92 s / 全长 0.78 s —— 延迟是 API 往返不是 prefill）；
  ③ **QA 自己就会拒答**：`scripts/measure_inline_abstention.py` 在 59 条上真跑 46 次生成，
  域内 **8/8**、远域 **2/2** 正确拒答、正样本 **0/36** 误拒，零额外调用，生成中位 1.30 s
  （带判定 ≈2.1 s）。独立判定保留为严格模式（400 + 不生成）。新遗留：inline 拒答在前端是
  普通回答（带 sources），要显式标注就得加 `refused` 事件。

- **2026-11-08（第二轮）：复合问题从「两问」扩到 24 条 / 7 种形状**。fixture 的 partial 样本
  现在用 `shape` + 有序 `parts` + `history` + `expect_refused` 描述：`two_parts` 8 / `three_parts` 3 /
  `gap_in_middle` 3 / `four_parts` 2 / `multi_gap` 2 / `followup` 4 / `followup_partial` 2。
  `classify_partial` 五桶 → **九桶**（新增 `dropped_part_with_gap`/`dropped_part_silent`
  「点明缺口却吞掉某个本可答子问题」、`refused_without_marker`/`answered_without_refusal`
  「该拒的样本没写标记 / 直接硬答」）。三种**逐处量尺**：`part_coverage()`（逐子问题判生死）、
  `post_gap_coverage()`（**缺口后面那一问还活着吗**）、`gap_coverage()`（**缺口全点明**——
  扁平 `absent_terms` 是 `any()` 语义，多缺口样本只点明第一处也会判达标，这是结构不是 bug；
  且该拒的样本不进这个分母）。多轮必须先过 `resolve_query`（检索用消解后问句、生成用原问句+history）。
  实测（口径修好后两轮一致）：24 条里 **0 过度拒答 / 0 漏答可答项 / 0 没写标记 / 0 硬答**，
  该拒的 3 条都写了标记，域内 8/8、远域 2/2、正样本误拒 0/36 一分不掉。
- **2026-11-08（第二轮）：第一次量真实语料上的复合问题**。`scripts/eval_compound_retrieval.py`
  （`state/*.json` 只读、gold 现场派生、命中=同片或同文件同节、单问基线没命中的子问不进分母）：
  2/3/4 问的合并覆盖@5 = 66.7% / 38.5% / 22.2%，**覆盖@5 与 @10 完全相同（+0.0pp）**。
  原因干净：k 被填满（每问命中块数 @5=4.94 / @10=9.77），未覆盖子问的目标片名次**双峰**
  （第 1 附近或 20 名开外，6–10 名的 0 条）→ **调 `TOP_K` 不是杠杆，方向是每问分路检索 / query 分解**。
  边界：装下 ≠ 答对；锚点不是可靠定位器（单问基线 39.4%，有效分母仅 37 个子问）；单配置单种子。
- **判定/标注口径的改动可以离线重算**（本轮三次救命）：分桶只依赖「回答文本 + 标注」，
  所以修判定器、改锚点后拿 `--json` 存下的回答重算即可（`scripts/rejudge_abstention.py`，已入库有单测；
  它比对分桶 / 逐项覆盖 / 缺口后存活 / 缺口点明数**四样**，只比分桶会漏掉「分桶没变但明细变了」）。
  只有改**提示词**或改**检索**才必须真跑。前提：每轮把 `answer` 原文写进 `--json`。

## 环境 / 提交 / 推送速查（六条最容易踩的）


1. **pwsh 默认沙箱必失败**（`SetNamedSecurityInfoW … grantWrite`）→ 加一次性
   `sandbox_permissions: danger-full-access` 即可；bash 工具在本机不可用。
   **跑测试必须用项目自带的 `.venv\Scripts\python.exe`**：PATH 上的 `python` 是 anaconda base，
   缺 `openai`/`fastapi`，`pytest` 会在 collection 阶段就报 23 个 ImportError（不是代码坏了）。
   **数文件行数别用 `Get-Content`**：它按 ANSI 解码，中文 UTF-8 文档会少算（190 行报成 118 行），
   用 `.venv` 的 Python 显式 `decode('utf-8')` 再 `splitlines()`。
2. **降级必须出声**：HyDE 曾因推理模型吃光 `max_tokens=200` 静默退回原问题，长期无人发现。
   任何「--hyde 口径」的数字先确认它真生效（评测脚本已有自检）。
3. **国内下模型走 ModelScope**（实测 12.6 MB/s），HF 只有 34–245 KB/s 且 `huggingface_hub`
   会卡在 `.incomplete` 0 字节 → 拉到本地目录后让 `RERANK_MODEL`/`EMBEDDING_MODEL` 指过去。
4. **测试绝不加载真模型**：`tests/conftest.py` 有 autouse 关掉 `RERANK_ENABLED`。
5. **提交纪律**：`type(scope): 中文` + `验证:` 证据；UTF-8 无 BOM 草稿 +
   `git commit --cleanup=verbatim -F`；提交前重跑验证、不复用旧结果；**只本地提交不推送**
   （推送需用户显式授权，推后回读 `git ls-remote` 核对）。
6. **推 GitHub 只能走 SSH over 443**：`github.com` 与 `api.github.com` 的 HTTPS在这台机器上
   被挡（直连/走 Clash 7897 都是 000，Clash 上游节点还经常整体挂掉），`codeload` 与
   `ssh.github.com:443` 可达。本仓库 `origin` 已改为
   `ssh://git@ssh.github.com:443/HluckyRing/RAGFlow.git`，密钥 `~/.ssh/id_ed25519_ragflow`
   （无口令，已加到 GitHub 账号），并把 `core.sshCommand` 设成
   `ssh -i "C:/Users/<本机用户名>/.ssh/id_ed25519_ragflow" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=C:/Users/<本机用户名>/.ssh/known_hosts"`。
   **两个坑**：路径必须用**正斜杠**（反斜杠会被 MSYS 的 sh 吃掉）；`UserKnownHostsFile`
   必须显式给（中文用户名会被 MSYS 转义成 `\272\306\310\312`，写不进 known_hosts）。
   这些都是**本仓库 `.git/config` 里**的本地设置，不进版本库；换机器要重配一遍。

## 待办全文（2026-11-14 版）

> 2026-11-09 版的前三项已全部做过一刀：① 真实写法复合问句（第十轮 `da8e736`）→ 收益从 +60pp
> 缩到 **+33.4pp**；② 生成侧（第十一轮 `16d132a`）→ 答全率 **35% → 90%**；③ 真实用户日志
> （第十二轮 `71ffb54`）→ **12 条里一条复合问都没有**。所以「拆问的真实收益」这条**仍然无法
> 验证**，只能等真实流量或做用户研究 —— **不许再拿手写样本冒充真实流量**。
> 真实日志暴露出的下一刀是**枚举 / 汇总型覆盖取样**（第十三~十六轮），目前的状态是：
> 跨文件取样在线上预算下 no-op、导航页降权无收益、**同文件跨节取样**是唯一有正面结果的一条
> （有效样本 1 条，故默认仍关）。

## 待办

1. **覆盖取样收尾**：把「同文件跨节」的 gold 扩到 5–6 条再定 `COVERAGE_INTRA_FILE` 默认值
   （开关已在 2026-11-14 拆成 `COVERAGE_CROSS_FILE` / `COVERAGE_INTRA_FILE`，两个都默认 0；
   `COVERAGE_SAMPLING` 只剩兼容别名）；补位超上限时要给「只覆盖 N/M 份」的提示
   （动 prompt 与 SSE 契约，第十三轮起欠着）。
2. **拆问的真实收益仍无法验证**：等真实流量出现复合问，或做用户研究。生成侧没量的两头：
   **多轮 + 复合**、**缺口点明率**（第十一轮样本全部可答，`absent_terms` 为空）。
3. **章节级汇总的量尺**：第十五轮证明「该知识库的主要内容」那条 gold 无效（章节级答案 vs
   节级核无干净解），要量「整理 / 总结某一份文件」得另做一套量尺（章节级 + 人工读答案）。
4. **更大语料规模**：分路收益取决于 `dynamic_top_k` 是否真被卡住（库里块数 ≫ 预算）；
   另一头只量到 32–99 块 / 最多 13 文件 1061 块。库小到「预算覆盖全库」时整个功能对覆盖率是空操作。
5. **关键词兜底路径下的复合覆盖**：仍 0 条走兜底，那条「最多回 3 块」的天花板没数据。
6. **多 seed 的方差**；**五问以上 / 三处缺口**；并发只压到 8（anyio 线程池默认 40），
   长时间 soak、内存与连接复用没测。
7. **安全侧剩下的大头**：真身份体系（账号 + 会话归属校验 + CSRF）；限流换共享存储（多实例）；
   结构页阈值（4-gram 0.6 / 跨 3 份文件）换语料要重校；`NAVIGATION_MARKERS` 仍是中文措辞表。
8. 拒答的「已检索 N 个文件」只数进了 context 的文件；真实语料的 reranker 分数带；
   reranker 性能路线（小 `RERANK_CANDIDATES` / 量化 ONNX）；xlsx 工作表名支路样本；
   「Node vm + DOM 替身」推广到来源 chip / 消息操作按钮。
9. **换机器/换模型必须重跑两道线**：双路融合的延迟线与拆问默认值的四道门 ——
   它们都是在这台机器上量出来的（`.agents/decide_decompose_default.py` 可原样复用）。
