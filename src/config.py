import os
import logging
import warnings
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("ai_rag")

os.environ.setdefault('HF_ENDPOINT', os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value, default_name):
    """把路径锚定到项目根：相对路径（含默认值）都以项目根为基准，绝对路径原样保留。

    这样不管从哪个目录启动（python src/server.py、还是先 cd 到别处再运行），
    向量库与状态目录都落在同一个地方，不会各自悄悄生出一份。
    """
    if value:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else PROJECT_ROOT / path)
    return str(PROJECT_ROOT / default_name)


API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-flash")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")

# 切片长度受 embedding 模型的 token 窗口约束：bge-small-zh-v1.5 的
# max_seq_length=512，中文约 1 字 1 token，600 字会被 tokenizer 静默截掉尾部。
# 480 字给特殊 token 与中英/数字混排的拆词留了余量。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "480"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
# 向量检索相关性阈值（l2 距离，越小越严格）。实测本项目语料上相关查询 max≈0.47、
# 无关查询 min≈0.62，默认取分离带中点 0.55；换语料/换 embedding 模型时应重新标定。
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.55"))

# 双路融合：把**原问题**和 **HyDE 文本**各当一次向量探针，两路名次用 RRF 融合。
# 动机：HyDE 会把短问句改写成「像文档的片段」，关键词被揉进长句后那一侧偶尔整段超阈值，
# 而原问题侧还贴着答案（反之亦然）；单探针遇到这种情况只能退到关键词兜底。
#
# 默认开启、闸门用 intersect（**两路都得近**）。依据是 2026-11-05 的实测：
#   - 合成语料两批独立假设文本：误召回 2/5→0/5、1/5→0/5，recall@5 与文件/section 归属
#     一分不掉，MRR 0.972→1.000、0.972→0.986；
#   - 真实语料（1199 块）：无关提问代理漏进 17/20→13/20，真实提问 11/12 不变；
#   - 代价：每条提问多一次 embedding 查询，实测 +17 ms（41 条 1.1s→1.8s）。
# 这条正好过「误召回压到 0/5 且中位延迟 ≤1.5s」那条默认开线，而 reranker 因为 3.9s 没过。
# union（任一路近就算近）召回更宽，但误召回一点没降；想换回来只改 FUSION_GATE。
# 数字与逐条明细见 docs/retrieval-baseline-2026-10-06.md「双路融合」一节。
FUSION_ENABLED = os.getenv("FUSION_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# RRF 的平滑常数：越大越平缓（标准值 60）。两路各只取 top_k/RERANK_CANDIDATES 条，
# 60 让「名次的相对差」不至于被单路的第一名吃掉。
RRF_K = int(os.getenv("RRF_K", "60"))
# 融合闸门口径：`intersect` = 两路都得近才算命中（默认，精度优先）；
# `union` = 任一路近就算近（召回优先）。实测 union 不降误召回，见上。
FUSION_GATE = (os.getenv("FUSION_GATE", "intersect").strip().lower() or "intersect")
if FUSION_GATE not in ("union", "intersect"):
    logger.warning("FUSION_GATE=%r 不认识（只认 union / intersect），按 intersect 处理", FUSION_GATE)
    FUSION_GATE = "intersect"

# 跨编码器精排（见 src/rerank.py）：把「向量距离」换成 query×块 的逐对匹配分。
# **默认先关**：等真实测出 CPU 延迟与正负分数带之后再决定默认值（2026-10-06 约定），
# 在那之前开它等于让每个提问白付一次 1.1GB 模型推理，阈值也只能靠猜。
# 实测数据与最终默认值记在 docs/retrieval-baseline-2026-10-06.md。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0").strip().lower() not in ("0", "false", "no", "")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# 送进重排的候选数：先放宽召回，再由重排决定留谁（开了重排就不再用 MAX_DISTANCE 先砍）
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
# 重排后最多留几块进 context
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
# 重排分数下限：低于它的候选直接丢。用实测正负分带标定，见文档。
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.3"))
# 可答性判断（见 src/answerability.py）：材料检索到之后，再让模型判一次「这段材料能不能
# 回答这个问题」，判「不能」就走已有的「未找到相关内容」拒答路径。
# 动机（2026-11-05 实测）：域内「看着像但没答案」的提问 8/8 全被作答，而阈值扫描与兜底
# 命中强度闸门都分不开（重排分里域内负样本 0.0004–0.9529 与正样本 0.0284–0.9993 完全重叠）。
#
# **默认关（2026-11-06 实测）**：这次调用每条提问固定 +0.78 s，而缩短判定输入并不省时间
# （top1 1.01 s / top2 0.92 s / 全长 0.78 s，延迟是 API 往返不是 prefill）。同时实测发现
# **生成那一次调用自己就会拒答**：`QA_SYSTEM_PROMPT_TEMPLATE` 第 1 条要求「背景没有答案就说
# 背景中没有提到相关内容」，59 条语料上域内负样本 **8/8**、远域 **2/2** 正确拒答，正样本
# **0/36** 误拒 —— 与独立判定分毫不差，却零额外调用（`scripts/measure_inline_abstention.py`）。
# 所以默认走 inline 拒答；要「硬 400 + 不生成」的严格口径再设 ANSWERABILITY_CHECK=1。
ANSWERABILITY_CHECK = os.getenv("ANSWERABILITY_CHECK", "0").strip().lower() not in ("0", "false", "no", "")
# 可答性判断要输出「能：<原文依据>」或「不能」，32 个 token 够写一小段依据；
# 关思考后这点预算不会被 reasoning 吃掉。
ANSWERABILITY_MAX_TOKENS = int(os.getenv("ANSWERABILITY_MAX_TOKENS", "32"))

# 拆问（query 分解）：复合问句先拆成若干子问题，各自检索再合并 context。
# **默认开**（2026-11-08 第九轮定案）：判据是跑之前就写死的四道门，实测全部通过 ——
# ① 复合问「两件事都答到」87.5% → **100.0%**；② 域外复合问 **4/4 正确拒答**；
# ③ 首字延迟中位 **−0.55 s**、p95 **−1.01 s**（分路省掉的 HyDE 调用比拆问那次更贵）；
# ④ 零错误、`sources` 22/22 先于首 token、拒答不挂来源、单问不误伤。
# 关掉也支持（`DECOMPOSE_ENABLED=0`）：那时**连闸门都不跑**，一次 API 都不花，
# 行为与改造前逐位一致。
DECOMPOSE_ENABLED = os.getenv("DECOMPOSE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
# 拆问最多拆几问（再多就不是「一个问题」，而是把整段话贴进来了）
DECOMPOSE_MAX_PARTS = int(os.getenv("DECOMPOSE_MAX_PARTS", "6"))

# 覆盖取样拆成**两个开关**（2026-11-14）：两套行为差得很远，一个开关管两件事就没法
# 分别评估。实测（第十三~十六轮，全部按线上预算）：
# ① 跨文件取样（`COVERAGE_CROSS_FILE`）：枚举 / 汇总型提问保证每份文件都有代表块。**线上预算下
#    是 no-op** —— 实体覆盖 24/38 → 24/38，导航页降权还掉到 20/38（那边是 gold 坏了，不是特性坏）。
#    所以它**仍然默认关**：开了只是多花补位查询。
# ② 同文件跨节取样（`COVERAGE_INTRA_FILE`）：问句点名章号（「帮我整理第一章的内容」）时把**那一份
#    文件**按节均匀铺开。**这条线上唯一有正面结果的一条**，2026-11-15 过了质量门（PPT 29/30、
#    机器学习 PDF 14/15，无回归）并补上了「只覆盖 N/M 段」的说明，所以**默认改成开**。
#    它只在「枚举 / 汇总措辞 + 明确章号」时触发，单问事实问答完全不受影响。


def coverage_switches(env=None):
    """解析覆盖取样的两个开关，返回 `(cross_file, intra_file, legacy_present)`。

    老开关 `COVERAGE_SAMPLING` 的语义是「两个都开」（第十三~十六轮的评测脚本都按这个名字
    跑的），保留兼容但要**出声**——它曾经代表两件事，现在只当作一次性别名。**显式写了就
    以它为准**（`COVERAGE_SAMPLING=0` 仍然是「两个都关」），否则默认值按新口径走。

    单独抽成函数是因为「import 时读一次环境变量」在测试里没法干净地重放：
    这条兼容语义必须能被钉住，而 `importlib.reload` 会把整个模块（含 OpenAI client）重造一遍。
    """
    env = os.environ if env is None else env

    def on(name, default=False):
        value = env.get(name)
        if value is None:
            return default
        return value.strip().lower() not in ("0", "false", "no", "")

    cross_file = on("COVERAGE_CROSS_FILE")                      # 默认关（线上预算下是 no-op）
    intra_file = on("COVERAGE_INTRA_FILE", default=True)        # 默认开（2026-11-15 过质量门）
    legacy_present = env.get("COVERAGE_SAMPLING") is not None
    if legacy_present:
        cross_file = intra_file = on("COVERAGE_SAMPLING")
    return cross_file, intra_file, legacy_present


COVERAGE_CROSS_FILE, COVERAGE_INTRA_FILE, _COVERAGE_LEGACY_SWITCH = coverage_switches()
if _COVERAGE_LEGACY_SWITCH:
    logger.warning("COVERAGE_SAMPLING 已拆成 COVERAGE_CROSS_FILE / COVERAGE_INTRA_FILE"
                   "（按旧语义覆盖两者的默认值）；建议改用新名字")
# 最多给几份缺失文件补位（每份多一次查询）。文件很多时不能让这一次提问变成 n 次检索。
COVERAGE_BACKFILL_MAX = int(os.getenv("COVERAGE_BACKFILL_MAX", "8"))
# 每份缺失文件取几个候选再挑「正文页」：一份文件里最像「内容是什么」的那块往往正是它的
# 目录页，只取 1 个就没有挑的余地（2026-11-12 第十四轮实测：头部 14 块里 10 块是知识导图）。
COVERAGE_PER_FILE_CANDIDATES = int(os.getenv("COVERAGE_PER_FILE_CANDIDATES", "3"))
# 同一份文件内部跨节取样：问句点名了章号（「整理第一章的内容」）时，**在这份文件内部**
# 按页均匀铺开。下面这个数是**硬上限**，实际取几块由字符预算反推（`coverage.blocks_that_fit`）：
# PPT 每页才 ~140 字，取 30 页也只有 4k 字符；md 每块 480 字时只装得下 ~15 块。
# 2026-11-15 实测（机器学习 3 份 PDF，96–110 页/章）：**写死 30 块**会让 100 页的章只铺开
# 约 27%，章尾整块落空（第1章 sklearn 94–96 页、第2章 决定系数 88–101 页都没进取样）。
# 所以默认不再写 30，而是放到字符预算允许的上限（8000 字 ≈ 44 块/140 字一页），由预算说话。
COVERAGE_INTRA_FILE_BLOCKS = int(os.getenv("COVERAGE_INTRA_FILE_BLOCKS", "200"))
# 整章枚举的上限：把这**一份文件的所有块**取回来（`retrieval.file_blocks`，用
# `collection.get(where=...)`，不花 embedding）。一页一块的课件 100 页也不到 400，
# 只有超大文件才会在这里被截断。2026-11-15 之前这里是「取最相似的 40 个候选」，
# 那正是「整理整章只整理了前 2/3」的原因。
COVERAGE_INTRA_FILE_CANDIDATES = int(os.getenv("COVERAGE_INTRA_FILE_CANDIDATES", "400"))


VECTOR_DB_PATH = resolve_path(os.getenv("VECTOR_DB_PATH"), "chroma_db")
# 上传原件（不是抽取文本）的存放根目录。原文件预览要读它；同样锚定项目根，
# 测试/冒烟用 UPLOAD_DIR 指向临时目录，避免落到真实用户数据上。
UPLOAD_DIR = resolve_path(os.getenv("UPLOAD_DIR"), "uploads")

# 单次上传大小上限（MB）。解析 PDF/DOCX 又慢又吃内存，超限直接挡在入口。
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# 调用大模型的超时（秒）。没有它，网络挂起时 SSE 会一直不出声。
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))

# 服务监听地址：默认只绑回环，避免这个没有鉴权的服务被同网段直接访问。
# 确实要对外提供（局域网/容器）时显式设 HOST=0.0.0.0。
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))

# 鉴权令牌：**只有对外监听时才是必需的**（`src/security.py` 的启动自检会强制这一点）。
# 本机回环 + 空 token 保持零摩擦，和改造前一模一样。
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()

# 归属 cookie 的 `Secure` 标记（2026-11-25 第二十四轮）。
# `auto`（默认）= 按**这次请求的真实 scheme** 决定：直连看 `request.url.scheme`，反代看
# `X-Forwarded-Proto`。本机 http 照样能用（加 `Secure` 会让浏览器直接丢掉 cookie，
# 「刷新就掉会话」），套了 HTTPS 又自动生效 —— 一个配置同时满足两种部署。
# 强制三态：`auto` / `1` / `0`（写错按 auto 处理并 WARNING，不静默当成 1 或 0）。
COOKIE_SECURE = (os.getenv("COOKIE_SECURE", "auto") or "auto").strip().lower()
if COOKIE_SECURE not in ("auto", "1", "0", "true", "false", "on", "off", "yes", "no"):
    logger.warning("COOKIE_SECURE=%r 不认识，按 auto 处理（可选 auto/1/0）", COOKIE_SECURE)
    COOKIE_SECURE = "auto"

# 旧会话的**首次绑定**要不要继续放行（2026-11-25 第二十四轮）。
# 病根：本刀（2026-11-24）之前建的会话状态里没有令牌哈希，谁先带着那个 sid 来，
# 服务端就给它发一把钥匙并绑定 —— 对**已经泄漏过的旧 sid** 是一次性抢先绑定窗口。
# 默认 1（继续迁移，不然升级后所有老用户的会话当场失效）；把老会话清干净以后设 0，
# 之后旧会话一律**换新会话**（前端拿到新 sid 会自动切过去，不报错、不丢新数据）。
SESSION_BIND_LEGACY = int(os.getenv("SESSION_BIND_LEGACY", "1"))

# 闸门措辞包（`src/gates/`）：闸门靠「措辞」判断意图，换语言/换语料就可能全部失效。
# 失效是 fail-safe 的（闸门不触发 → 退回改造前行为），但必须**看得见** —— 见
# `scripts/analyze_real_questions.py` 的「0 触发」告警。加一门语言 = 加一个包文件。
GATE_PACK = os.getenv("GATE_PACK", "zh").strip().lower() or "zh"
# /api/chat 限流：每问一次都要花钱调大模型，被刷就是真金白银。
# 0 = 关闭。按**客户端 IP** 计数（session_id 是客户端生成的，换一个就绕过）。
CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT", "20"))
CHAT_RATE_WINDOW = float(os.getenv("CHAT_RATE_WINDOW", "60"))

# 范围词 → 文件过滤器（`SCOPE_FILTER`，2026-11-19 第十九轮）。
# 动机（真实对话逐块复现）：问「结合第2章课件，在同一训练集上不断提高多项式回归的阶数，
# 通常可以观察到什么」时只命中 1 块（第1章 p46 的通用框架页）→ 老实拒答；同一条问题
# **去掉前缀**却命中 3–4 块并答对。根因有两层：①「第2章/课件」这类**元信息词**被当检索词，
# 而向量空间里离它最近的是封面页 / 内容提要页 / 目录页，把答案页从第 8 名挤到第 16 名；
# ② 双路融合的 `intersect` 闸门要求两路都召回，交集于是塌成 1 块。
# 打开后：用章号定位到**那一份文件**（`scope.scope_target`，复用 `coverage.pick_target_file`），
# 检索加 `where={"file_name": ...}`，并把范围短语从探针文本里删掉。
# **默认开**（2026-11-19 实测过了四道门，见 docs/retrieval-baseline-2026-10-06.md「范围词」一节）：
# ① 质量：12 条「结合第N章课件，…」提问的事实核 11/14 → 13/14、整条答到 10/12 → 11/12、
#    命中目标文件 11/12 → 12/12，逐条无一变差（那条真实拒答 s01 从 0/2 变 2/2）；
# ② 负样本：误召回 10/15 不变（走向量那条 3/15 不变）；
# ③ 代价：检索中位 50 ms → 61 ms；④ 回退：开关关着逐位不变，整章枚举那条路两臂完全相同
#    （47 块 / 7 441 字符），认不准（无章号 / 章号匹配多份文件）一律不动手。
# 触发条件很窄：**问句点名章号 + 唯一匹配到一份文件**；没点名章号的提问一个字都不改。
SCOPE_FILTER = os.getenv("SCOPE_FILTER", "1").strip().lower() not in ("0", "false", "no", "")
# 双路融合的候选池（`FUSION_CANDIDATES`）：每路先取这么多候选再融合 + 过 `intersect` 闸门，
# 最终仍只留 `top_k` 块。0 = 维持现状（每路只取 top_k，闸门就是「两路各 top_k 的交集」）。
# 动机同上：交集对**名次扰动**极度敏感 —— 同一句图一，只把候选池从 10 放到 20，
# 交集就从 1 块变 5 块（实测见 docs/retrieval-baseline-2026-10-06.md「范围词」一节）。
# 代价是每路多召回、闸门外候选变多（embedding 次数不变，只是 n_results 变大）。
FUSION_CANDIDATES = int(os.getenv("FUSION_CANDIDATES", "0"))


def build_client(api_key, base_url, timeout):
    """构造 OpenAI 客户端。

    缺 API_KEY 时返回 None 让上层降级（关键词检索照常可用），
    而不是让 openai SDK 在 import 阶段抛一个英文异常把整个服务带崩。
    """
    if not api_key:
        logger.error(
            "未配置 API_KEY：请在项目根目录的 .env 里填入 API_KEY=sk-xxx 后重启服务；"
            "当前只能用关键词检索，问答与 HyDE 不可用"
        )
        return None
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


client = build_client(API_KEY, BASE_URL, LLM_TIMEOUT)
