# RAGFlow —— 本地知识库问答（FastAPI + SSE + 本地向量检索）
#
# 三个必须知道的点（都写在 README「Docker 部署」一节里）：
# 1. 容器里必须 HOST=0.0.0.0，而 src/security.py 的启动闸门规定「对外监听必须有
#    AUTH_TOKEN」—— 所以**不设 AUTH_TOKEN 容器会直接退出（退出码 2）**，这是故意的；
# 2. 向量模型（BAAI/bge-small-zh-v1.5）在**构建期**就烤进镜像，运行时不需要联网；
#    重排模型（bge-reranker-base，1.1 GB）默认关（RERANK_ENABLED=0），没有烤进来；
# 3. state / uploads / chroma_db 三个目录必须挂 volume（/data 下一份），否则容器一删
#    会话和索引全没。
FROM python:3.11-slim

# PYTHONUNBUFFERED: 日志不缓冲，docker logs 才实时；
# HF_HOME: 模型缓存固定位置，构建期烤进来、运行时直接读；
# PIP_NO_CACHE_DIR: 不把下载缓存留在镜像层里。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    HF_ENDPOINT=https://hf-mirror.com \
    EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 \
    HOST=0.0.0.0 \
    PORT=8080 \
    STATE_DIR=/data/state \
    UPLOAD_DIR=/data/uploads \
    VECTOR_DB_PATH=/data/chroma

# curl 给 HEALTHCHECK 用；tini 让 Ctrl-C / docker stop 能正确转发信号给 uvicorn
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷依赖清单：改业务代码时不必重装 1 GB 的依赖（利用层缓存）
COPY requirements.txt ./

# torch 先按 CPU 版装（默认 PyPI 的 Linux torch 会拖进 CUDA 依赖，镜像白白大 2 GB）；
# CPU 源不可达时退回默认源 —— 能跑优先，失败原因会出现在构建日志里。
RUN pip install --upgrade pip \
 && (pip install --index-url https://download.pytorch.org/whl/cpu torch \
     || pip install torch) \
 && pip install -r requirements.txt

# 把向量模型烤进镜像：运行时（尤其断网 / 内网）不需要再下载
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}'); \
print('embedding model baked into', __import__('os').environ['HF_HOME'])"

COPY src ./src
COPY server.py README.md LICENSE ./

# 非 root 运行：容器逃逸的成本高一个量级；/data 归 app 用户所有
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data/state /data/uploads /data/chroma \
 && chown -R app:app /data /app
USER app

VOLUME ["/data"]
EXPOSE 8080

# 健康检查打 /api/session：设了 AUTH_TOKEN 时会返回 401（也算活着），所以只要求「有 HTTP 响应」。
# 注意别用 curl -f：-f 会把 401 当失败，服务活着也会被判 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/api/session" \
        | grep -qE "^(200|401|403)$" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "server.py"]
