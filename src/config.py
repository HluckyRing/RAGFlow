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

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "10"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
# 向量检索相关性阈值（l2 距离，越小越严格）。实测本项目语料上相关查询 max≈0.47、
# 无关查询 min≈0.62，默认取分离带中点 0.55；换语料/换 embedding 模型时应重新标定。
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.55"))
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
