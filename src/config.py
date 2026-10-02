import os
import logging
import warnings
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
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
