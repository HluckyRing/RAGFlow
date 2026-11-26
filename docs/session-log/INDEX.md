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
- **2026-11-19 范围词 → 文件过滤器（`SCOPE_FILTER` 默认开）**：
  `docs/session-log/2026-11-19-scope-filter.md`
  —— 起点是一次真实提问：「结合第2章课件，…」拒答、去掉前缀就答对。逐块复现定位到两层原因：
  ①「第2章/课件」这类**元信息词被当检索词**，把答案页从第 8 名挤到第 16 名（距离 0.40 ≪ 0.55，
  不是阈值）；② `intersect` 闸门要求两路都召回，交集塌成 1 块。改法：`src/scope.py`（章号→文件
  定位 + 探针改写，复用 `coverage` 同一份判据）+ `where` 过滤 + 空命中退回全库；另加
  `FUSION_CANDIDATES` 候选池。实测（12 条带范围词提问 / 257 块 / 真 HyDE）：事实核 11/14→**13/14**、
  整条答到 10/12→**11/12**、目标文件 11/12→**12/12**、逐条无回归；负样本误召回不变；+11 ms。
  `pytest` 750→**768**（+15 范围词 +2 文档漂移守卫 +1 同步守卫）、变异检查 8 项全红；README 的机械部分改由 `scripts/sync_readme.py` 生成（CI `--check`）。坑：`git checkout`
  抹掉未提交改动、延时被 HyDE 污染、默认语料选错、误召回没分路、**文档里的硬数字没人管**（README
  的 750 passed / 1998 行 / 「这个服务没有鉴权」全部过期，本轮补了两条守卫）。
- **2026-11-23 发版 v1.2.1（缺口清单）**：`docs/release-notes-1.2.1.md`
  —— 功能提交 `ceec1ea` 之后打补丁版：`196fa8a`（version 1.2.0→1.2.1 + 手写 notes）→
  标签 `v1.2.1` → Release 30 s 建好（附件 794 350 字节，37 entries / `index.html` /
  12 vendor / `Version: 1.2.1`）→ `gh release edit --notes-file` 换掉自动正文 →
  `gh release list` 显示 Latest。**同一轮自查抓到日期写错**（本轮工作在 11-23，我按上一轮
  印象写成 11-20）：`git mv` 改名 + 五处正文改正 + 会话日志补「跨天会话要当场 Get-Date」
  这条教训。
- **2026-11-26 上传进度条：真实阶段 + 真实百分比**：
  `docs/session-log/2026-11-26-upload-progress.md`
  —— 上传 250 块要 13–27 s，界面上只有静态的「仍在解析中…」。改成后端上报真实阶段
  （`parse` 第几个文件 → `index` 已嵌入几块）+ 前端 400 ms 轮询 `/api/upload/progress`，
  拿不到 total 就显示不确定态、绝不假装百分比。**不引入任务队列**：同步上传接口 + 内存进度表
  （TTL 300 s，`finally` 删条目）。踩到真竞态：**第一帧就是 404**（轮询比后端建条目更早），
  第一版「404 就停轮询」导致帧数 0 → 改成错误分支不终止、只由调用方 stopPolling 收起。
  变异 5/5 红（其中 D 第一版被自己的注释骗过，守卫收紧成断言 `form.append('upload_id', ...)`）；
  实测 26 万字/160 块：99 帧，parse 1/1 → index 0/64/128/160；`pytest` 872→**882**。
- **2026-11-26 删文件要等 3 分钟：定向删除 + 批量 embedding + 前端即时反馈**：
  `docs/session-log/2026-11-26-delete-fast.md`
  —— 用户反馈「删文件很慢也没自动刷新」。真因两层：① 删一个文件走 `_rebuild_collection`，
  把**剩下所有文件**重新切片 + embedding（真实语料 811 块 ≈ 3 分钟），而删除根本不需要 embedding；
  ② `_rebuild_collection` 是**逐块** `collection.add`（811 块 = 811 次调用）；③ 前端点确认后弹窗
  立刻关、`refreshView` 要等 DELETE 返回，慢的时候等于「点了没反应」。改法：定向
  `delete(where={"file_name":...})`（毫秒级，但**删完要数一遍块数**——老索引无 `file_name`
  元数据时 where 会静默删 0 条，比慢更糟）→ 批量 `EMBED_BATCH=64` → 前端乐观更新 + 处理中态 +
  失败回滚。实测删文件 4.37s→**0.08s**、上传 250 块 估 62s→**13.6s**；变异 5/5 红；
  `pytest` 864→**872**。
- **2026-11-25 Docker 部署：Dockerfile + compose，镜像实测跑通**：
  docs/session-log/2026-11-25-docker-deploy.md
  —— 真构建（2.82 GB）+ 真起容器 + 走完「登录 → 建对话 → 上传 → 解析 → 离线向量化 →
  \docker restart\ 后数据还在」。三个坑：容器必须 \HOST=0.0.0.0\ 于是 \AUTH_TOKEN\ 变成硬要求
  （**不给默认值**，否则等于绕过启动闸门）；模型（93 MB）**构建期烤进镜像**、运行时零下载；
  state/uploads/chroma 统一挪到 \/data\ 走命名卷。torch 装 CPU 版省约 2 GB。
  4 条新守卫钉住「密钥/用户数据不进镜像 + 非 root + volume + 只发布回环」。\pytest\ 835→**839**。
- **2026-11-25 安全两把小刀：cookie `Secure` + 旧会话迁移窗口（待办 ⑦ 遗留）**：
  `docs/session-log/2026-11-25-security-cookie-secure.md`
  —— cookie 以前硬编码不带 `Secure`（HTTPS 后能被明文带出去），但无脑加又会让本机 http
  「刷新就掉会话」→ `COOKIE_SECURE=auto|1|0`，`auto` 按**这次请求的真实 scheme**（直连看
  scheme、反代看 `X-Forwarded-Proto` 首个值）判断。旧会话首次绑定是**已泄漏旧 sid 的抢先
  绑定窗口** → `SESSION_BIND_LEGACY=0` 可关（默认 1 保升级平滑，关掉后旧 sid 换新会话）。
  6 条新单测 + 变异检查 4/4 红；`pytest` 829→**835**。随后发 **v1.2.2**（顺手验通真·Actions
  发版链路，闭环待办 ⑧）。
- **2026-11-25 缺口清单第二刀：行政页 / 只报开头 / 4 条上限（待办 ① 遗留）**：
  `docs/session-log/2026-11-25-gap-list-second-cut.md`
  —— 上轮自己留的三条「没量」口子在**真实 `state/` 语料（36 份文件 / 2886 块）**上量掉：
  ① 行政页霸榜（频次前三 `课程思政小讲堂`19 / `学 习 进 步 ！`19 / `第二次作业`17，噪音表用
  子串匹配、真实文本逐字带空格 → 一个没命中）→ **整行去空白归一化精确匹配**（不做子串，
  否则 `作业调度算法`/`两级文件目录` 被误杀）；② 「前 4 条」只覆盖文件开头 → **按位置均匀
  铺开（首尾必含）**；③ 「4 条」是拍的 → **字符预算 300 字 + 硬上限 12**（与 `blocks_that_fit`
  同口径）。另补：34.6% 的块解析不出标题（其中 628 块有页码）→ 「第 N 页」兜底命名，但**稳定
  分区排在标题之后**；结构页/行政页既不命名也不计入缺口。实测点名条目 84→244、行政页 1→0、
  每份清单 ≤300 字；`pytest` 822→**829**。**仍不判轻重**、md/docx 无分页语料的「页标题」仍是
  块首行（两个尾巴都写进待办①）。
- **2026-11-24 会话归属：HttpOnly cookie + 服务端令牌（待办 ⑦ 第二刀）**：
  `docs/session-log/2026-11-24-session-ownership.md`
  —— 上一刀堵住匿名访问，这一刀堵剩下的真洞：`session_id` 客户端生成、服务端对任何 sid 都无条件
  加载状态 → sid 泄漏就能读那份会话。做法：`/api/session` 下发 **HttpOnly** cookie（`sid.token`，
  状态里**只存 sha256**），其余接口 `_session_or_403` 校验 sid+token；一次调用涵盖新建 / 恢复 /
  旧会话首次绑定；前端启动带上 localStorage 的 sid、遇 403 丢本地 sid 重开。判据：别人的 sid 403、
  伪造 token 403、A 的 cookie 开不了 B、状态无明文、旧会话绑定后无 cookie 进不来、同浏览器恢复同一会话；
  变异 **5/5 红**；真 uvicorn+curl 冒烟 **10/10**；`pytest` 814 → **822**。
  顺带改了三处既有测试（一个 TestClient = 一个浏览器），并记下「旧会话首次绑定」这个一次性迁移窗口。
- **2026-11-24 AUTH_TOKEN 逐请求校验 + CSRF（待办 ⑦ 第一刀）**：
  `docs/session-log/2026-11-24-auth-basic.md`
  —— 读代码读出真洞：`AUTH_TOKEN` **只在启动时**被检查（对外监听必须设），请求到达时一次都没校验 →
  `HOST=0.0.0.0` + 设了 token 的部署其实**对外完全开放**。改法：`server.py` 加一道 http 中间件，
  设了 token 就要求 **HTTP Basic**（密码=token，`hmac.compare_digest`，401 带 `WWW-Authenticate`，
  浏览器自动带凭证、前端零改动）+ 写操作 `Origin`/Host 不一致一律 403；**没设 token 逐位不变**。
  判据：401/200/403 三类 + 错误体不泄漏 token + 首页也受保护；变异检查 **4/4 红**；`pytest` 801 → **814**。
  坑三条：测试碰 `server` 会写真实 `state/` 与 chroma（实测留下 17 个空会话 + 5 个空集合，已清）；
  `"torch" not in sys.modules` 是顺序依赖的假红（改成差集）；读源码找字符串的守卫被 docstring 骗（改行为 spy）。
  **仍未解决**：会话归属（sid 泄漏=能读那份会话）、多用户账号、Basic 明文需 HTTPS、限流进程内。
- **2026-11-23 指代式定位：标题词唯一匹配（待办 ④）**：
  `docs/session-log/2026-11-23-scope-deixis.md`
  —— 先量现状：待办里的「第二章那份材料」**本来就已能定位**（有章号），真正失效的是「点了文件
  标题却没有章号」那一类；「这份材料」「上面那个实验」不是没实现而是**没有依据**。
  新增第二条定位路：指代+材料标记 + 标题前缀 ≥3 字 + **最长命中优先 + 证据串在语料里唯一** →
  才加 `where` 过滤，并把「那份 PPT / 那个课件」从探针里删掉。判据：**误定位 0**（22 条标注）、
  新能力 5/5 + 3/3、对照零变化（12 条 fixture 0 差异）、变异 5/5 红、`pytest` 786 → **801**。
  返工一次：只做「最长命中」时「存储器那份材料」会锁死到 `存储器管理`（「存储器」也出现在
  `虚拟存储器` / `磁盘存储器管理` 里）→ 补了「证据串唯一」这一级。
- **2026-11-23 发版正文硬门：Release 必须用仓库里的手写 notes（待办 ⑧）**：
  `docs/session-log/2026-11-23-release-notes-guard.md`
  —— `--generate-notes` 在直推 main 的仓库里只产出「Full Changelog」一行（v1.1.0 / v1.2.0 /
  v1.2.1 三次都验证），这次改成硬门：逻辑搬进可测的 `scripts/check_release_notes.py`
  （标签 → `docs/release-notes-<版本>.md`，缺文件或短于 400 字符就 `::error::` + 退出 1，
  通过则把 `NOTES` 写进 `$GITHUB_ENV`），工作流只调用它 + `gh release create --notes-file`，
  `--generate-notes` 已删。+6 条守卫（pytest 780 → **786**），变异检查 4/4 全红。
  **量不到**：真·Actions 运行没验（本机 bash 指向没有发行版的 WSL），
  `$GITHUB_ENV → $NOTES` 这条链要等下次真实打标签才算验过。
- **2026-11-23 ML 语料扩到 4 章 + 整章取样质量门重跑（待办 ②）**：
  `docs/session-log/2026-11-23-ml-4ch-gate.md`
  —— 用户只有 4 章（第4章 - 决策树.pdf 91 页新入），语料 3 章 257 块 → **4 章 348 块**；
  **旧数 93%（3 章 14/15）作废**，同口径重跑两遍两臂：ON **18/20（90%）两次一致**、
  整章答全 3/4、拒答 0/4、context 7 535 字符 / 47 块；OFF 2/20、1/20、context 434 字符 / 3 块。
  逐条：第1章 3/5（漏「非监督学习」「sklearn」，两遍一致=真漏）、第2/3/4 章 5/5（**新章满分**）。
  判据改成「≥90% 且零回归 + 逐条都能过」——4 条样本算不出百分比精度；待办 ② 结掉，
  语料上限就定在 4 章。第4章的 5 个事实核全部取自正文页（首次出现 p29 之后），不用提纲页送分。
- **2026-11-23 覆盖说明点名「缺的是哪几页」（待办 ①）**：
  `docs/session-log/2026-11-23-gap-list.md`
  —— 待办第 1 条，但**结论与设想不同**：先做了「小节编号归属」的原型，实测三种坏法（第1章
  `1.6 操作系统的结构设计` 正文页首行是「操作系统结构设计」→ 整节被剔除、第4章算出 `p12-0`
  负区间、ML 的 PDF 没有编号大纲还把正文算式当标题）→ 判定「猜错的缺失小节比不说更糟」，
  换成**页标题级**缺口清单：标题 = 块前 3 行第一个像标题的行（去页码前缀 / 跳过噪音 / 结构页
  就地判死 / >24 字当正文放弃 / 截 18 字），缺失 = 没进 context 的块标题 − 进了 context 的块标题
  （同标题连排多页读到 1 页就算摸到）。四条判据全过：精确率 100%（6 份真实文件 + 变异检查）、
  解析不出时逐字退回老话术、字数开销 +64（上限 150）、6 条整章 gold 事实核 **29/30 与 28/30
  两遍 vs 老话术 28/30 与 29/30 两遍**（持平，答全 4–5/6、拒答 0/6）。**诚实结论：这是透明度
  改进、不是质量杠杆**。`pytest` 768 → **780**；变异检查 4 项全红（第一版 A 项没红 → 池子缺
  重复标题，补上才红）。
- **2026-11-20 发版 v1.2.0（拆问 / 覆盖说明 / 范围词 / 安全闸门）**：
  `docs/session-log/2026-11-20-v1.2.0-release.md`，正文看 `docs/release-notes-1.2.0.md`
  —— 判据：落后 **79 提交**（43 个 feat/fix/perf/refactor）、`main == origin/main`、CI success、
  `pytest` **768 passed**、`sync_readme --check` 一致；待办 1–7 全是长线研究，**不阻塞发版**，
  但全部写成「已知限制」。一个提交 `2157668`（version 1.1.0→1.2.0 + 手写 notes）→ 标签
  `v1.2.0` → `publish.yml` 18 s 出 Release（wheel 791 593 字节）→ `gh release edit --notes-file`
  换掉自动正文 → 回读 Latest / 标签 / 附件内容（37 entries、`index.html`、12 vendor、
  `Version: 1.2.0`）/ CI success。坑五条，头两条是通用的：**`--generate-notes` 对直推 main
  的仓库只有一行 Full Changelog**（v1.1.0 与本轮都验证）、**别拿 wheel 字节数比对**
  （Windows 792 031 vs Linux 791 593，要抽查内容）；另有 PowerShell 拆 `gh` 参数、
  破坏性变更（公开监听无 `AUTH_TOKEN` 拒绝启动、`CHUNK_SIZE` 600→480）必须进 notes。
- **2026-11-14 覆盖取样拆开关 + 文件内跨节质量门（两件）**：
  `docs/session-log/2026-11-14-coverage-switch-and-intra-gold.md`
  —— ① 把混在一起的 `COVERAGE_SAMPLING` 拆成 `COVERAGE_CROSS_FILE`（线上预算下已证 no-op）与
  `COVERAGE_INTRA_FILE`（唯一有效的一条），都默认 0，老名字降为兼容别名（= 两个都开 + WARNING）；
  ② 把文件内跨节的 gold 从 **1 条扩到 6 条「整理第N章」/ 30 个事实核**，事实核覆盖
  **4/30（13%）→ 27/30（90%）**、整章答全 0/6 → 4/6、拒答 4/6 → 0/6，代价是首 token 2.0 → 5.6 s、
  生成 3.6 → 13.6 s（回答从「拒答」变成 2–4k 字整章框架）；**默认仍 0**，判据写死（换 md/pdf
  语料同口径 ≥90% 且无回归 → 改 1）。顺带抓到一个真 bug：目标文件一块都没命中（关键词兜底回来
  的块没有 metadatas）时 `0 < 0` 导致返回**空 context / 空回答**，已修 + 回归测试 + 变异检查。
  2 个提交、`pytest` 721 → **727**、坑六条（一个开关管两套行为两套都评估不了、样本一多就踩到
  新失败模式、量尺假阴性又来两次）。**180 行**。
- **2026-11-15 机器学习语料侦察 + 整章枚举（A+B）**：
  `docs/session-log/2026-11-15-intra-file-enumerate-and-ml-corpus.md`
  —— 新上传的三份机器学习 PDF（96/110/51 页）成了第二份语料（也是第一种非 PPT 语料）；
  文件内跨节取样在它上面只有 **73%**，逐条读原文定位到取样侧两处（候选池只有**最相似的 40 块**、
  块数**写死 30**），改成 `retrieval.file_blocks` **整章枚举**（`collection.get`，不花 embedding）
  + 块数**按字符预算反推**后：**PPT 13 文件 90% → 97%（29/30）、整章答全 5/6、无回归**；
  **机器学习 3 PDF 73% → 93%（14/15）**。代价：context 中位 **7 560 字符**（逼近 8000）、
  生成 13.6 → **17.8 s**。判据「md/pdf ≥90% 且无回归」**已达成**，默认值仍 0（差的不是质量而是
  「延迟与 context 预算要不要付」）。剩下的 1 处真漏是**物理上限**（整章 10 541 字符装不进 8000）。
  方法学：ML 语料的 gold **不许用「内容提要」页上的小节名**（知识导图送分翻版，脚本会拒绝）。
  收尾（同日晚）：**覆盖说明上线**（取样填 `{file_name, covered, total}` → context 最前面插
  【覆盖说明】+ SSE `coverage` 事件 + 前端小标签 + 落盘；拒答不推），据此
  `COVERAGE_INTRA_FILE` **默认 0 → 1**（跨文件仍 0；老开关显式写了以它为准）。
  2 个提交、`pytest` 727 → **741**、变异检查 8 项全红、坑六条。**153 行**。
- **基线与结论的权威版本**：`docs/retrieval-baseline-2026-10-06.md`（**2600 行**，逐条实测表）
- **约定**：`AGENTS.md` / `CLAUDE.md`（两份必须逐字一致，当前均 191 行、仅 4 行工具相关文本不同
  = `git diff --no-index -U0` 正好 8 行差异）；
  审查清单归档 `docs/code-review-2026-10-01.md`（P0 1–7 / P1 8–17 / P2 18–21，编号连续）。

## 现状长文（2026-11-14 快照，迁自 MEMORY.md）

> **本节是快照，不是当前状态**：工作记忆（`MEMORY.md`）按约定不入库，所以仓库里没有
> 「当前状态」文件 —— 最新状态看 `git log`。第十~十七轮（`da8e736`..`d498395`）的数字见
> `docs/retrieval-baseline-2026-10-06.md` 对应小节（「真实写法的复合问句」「生成侧」
> 「真实用户日志」「跨文件取样」「排除导航页」「同文件内部跨节取样」「两条小风险的加固」）。

## 现状（2026-11-15 收尾：A+B + 覆盖说明，`COVERAGE_INTRA_FILE` 默认开）

- **已同步**：本地 `main` 起点 `5a330ef`（= `origin/main`），其上另有 2026-11-15 的三个提交
  （整章枚举 A+B / 覆盖说明 + 默认改开 / `.agents/` 进 `.gitignore`）；全量 `pytest tests -q` →
  **750 passed**。
- **用户数据不会被误提交**：`uploads/` / `state/` / `data/` / `chroma_db/` 一直在 `.gitignore` 里，
  2026-11-15 又把 **`.agents/`** 加进去 —— 它里面是**用户上传文件的解析文本**、模型回答全文与
  各轮原始 JSON，以前只靠「约定不入库」，一次 `git add -A` 就会推进公开仓库。守卫是**功能断言**
  （`tests/test_repo_hygiene.py` 直接问 `git check-ignore`，另有一条挡「已跟踪再加忽略无效」），
  不是「.gitignore 里有这行字符串」。改完 `git status` 不再出现 `?? .agents/`。
- **覆盖取样**：`COVERAGE_INTRA_FILE` **默认 1**（2026-11-15 过质量门 + 覆盖说明上线），
  `COVERAGE_CROSS_FILE` **仍 0**（线上预算下是 no-op）；老开关 `COVERAGE_SAMPLING` 显式写了
  就以它为准 + WARNING。文件内取样 = **整章枚举**（`retrieval.file_blocks`）+ **块数按字符预算
  反推**；实测 **PPT 29/30（97%）**、**机器学习 3 PDF 14/15（93%）**、无回归；代价 context 中位
  **7 560/8 000 字符**、生成 **17.8 s**（回答从「拒答」变成 2–4k 字整章框架）。
- **覆盖说明**（同日晚）：只摘到一部分时 ① context 最前面插【覆盖说明】告诉模型「别假装看全了」；
  ② SSE 推 `coverage` 事件并落盘；③ 前端渲染「本次只摘录了《X》的 43/110 段」小标签；
  ④ **拒答不推也不落**（与「拒答不挂来源」同一条纪律）。默认配置冒烟：`43/110`、7 067 字符。
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
   **别用 `git add -A` / `git add .`**：只按路径 `git add` 具体文件 —— 用户数据目录虽然都在
   `.gitignore` 里（含 2026-11-15 起显式忽略的 `.agents/`），但白名单式添加才是最后一道保险。
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
> （有效样本 1 条，故默认仍关）。**2026-11-14 补**：开关已拆开、gold 扩到 6 条「整理第N章」
> （事实核覆盖 13% → 90%、拒答 4/6 → 0/6），默认仍 0，判据写成「换 md/pdf 语料同口径 ≥90% 且
> 无回归 → `COVERAGE_INTRA_FILE` 改 1」。**2026-11-15 补**：机器学习 3 份 PDF 上侦察到 73%，
> 定位到「候选池只有 40 块 + 块数写死 30」并改成**整章枚举 + 预算反推**后，
> **PPT 97%（29/30，无回归）/ ML 93%（14/15）** —— **判据已达成**，默认值只差一个产品决策
> （+2~4 s 生成、context 逼近 8000）。

## 待办

1. ~~**「缺的是哪一节」**~~ → **2026-11-23 做完（脚页标题级缺口清单）、2026-11-25 补第二刀**
   （行政页归一化匹配 + 均匀铺开 + 字符预算 + 页码兜底）：见
   `2026-11-23-gap-list.md`、`2026-11-25-gap-list-second-cut.md`。剩下：不判轻重、
   md/docx 无分页语料的「页标题」其实是块首行。
2. **机器学习语料补到 5–6 章**：现在只有 3 条样本（一条 6.7pp），第1章那份 PDF 前 4 页还是课程介绍。
3. ~~**「内容提要」页要不要进 `NAVIGATION_MARKERS`**~~ → **2026-11-19 判定作废**。原因：
   `NAVIGATION_MARKERS` / `is_navigational()` / `prefer_content()` **只在 `llm.ensure_file_coverage`
   里用**（主检索排序不看它），而当初要治的是**检索层**名额被前言页吃掉（图一那条提问没走覆盖
   取样，加词对它零作用）；范围词场景已由 `SCOPE_FILTER` 解决（事实核 11/14 → 13/14，默认开）。
   如果将来要治「纯章号 / 枚举汇总」提问的前言页，应新开「**检索层**前言页降权」这条（历史上
   导航降权实测是 no-op 甚至掉分，且当时 gold 坏了），不要塞回措辞表。
4. **章号之外的指代定位**：「这份材料」「上面那个实验」现在一律不定位（原样返回）；要做得先有
   文件级指代消解，否则定位错会把整章取成另一份文件。
5. **拆问的真实收益仍无法验证**：等真实流量出现复合问，或做用户研究。生成侧没量的两头：
   **多轮 + 复合**、**缺口点明率**（第十一轮样本全部可答，`absent_terms` 为空）。
6. **章节级汇总的量尺**：第十五轮证明「该知识库的主要内容」那条 gold 无效（章节级答案 vs
   节级核无干净解），要量「整理 / 总结某一份文件」得另做一套量尺（章节级 + 人工读答案）。
7. **更大语料规模**：分路收益取决于 `dynamic_top_k` 是否真被卡住（库里块数 ≫ 预算）；
   另一头只量到 32–99 块 / 最多 13 文件 1061 块。库小到「预算覆盖全库」时整个功能对覆盖率是空操作。
8. **关键词兜底路径下的复合覆盖**：仍 0 条走兜底，那条「最多回 3 块」的天花板没数据
   （本轮倒是抓到了兜底块**没有 metadatas** 导致文件内取样返回空 context 的真 bug，已修）。
9. **多 seed 的方差**；**五问以上 / 三处缺口**；并发只压到 8（anyio 线程池默认 40），
   长时间 soak、内存与连接复用没测。
10. **安全侧剩下的大头**：真身份体系（账号 + 会话归属校验 + CSRF）；限流换共享存储（多实例）；
   结构页阈值（4-gram 0.6 / 跨 3 份文件）换语料要重校；`NAVIGATION_MARKERS` 仍是中文措辞表。
11. 拒答的「已检索 N 个文件」只数进了 context 的文件；真实语料的 reranker 分数带；
   reranker 性能路线（小 `RERANK_CANDIDATES` / 量化 ONNX）；xlsx 工作表名支路样本；
   「Node vm + DOM 替身」推广到来源 chip / 消息操作按钮。
12. **换机器/换模型必须重跑两道线**：双路融合的延迟线与拆问默认值的四道门 ——
   它们都是在这台机器上量出来的（`.agents/decide_decompose_default.py` 可原样复用）。
