# 2026-11-25 Docker 部署：Dockerfile + compose，镜像实测跑通

> 用户问「能不能部署到 Docker」。答案是能，而且这轮把它**真跑通了再写文档**：
> 构建、启容器、走完「登录 → 建对话 → 上传 → 解析 → 向量化 → 重启后数据还在」全链路。

## 一、三个必须先想清楚的坑

| 坑 | 原因 | 处理 |
|---|---|---|
| **`AUTH_TOKEN` 变成硬要求** | 容器内必须 `HOST=0.0.0.0` 才能被端口映射接住；而 `src/security.py` 的启动闸门规定「绑非回环 + 无 token → 拒绝启动」 | **不给 Dockerfile 设任何 `AUTH_TOKEN` 默认值**（给了就等于用一把人尽皆知的钥匙绕过闸门），README 里写清「先生成令牌再起容器」。实测不设时容器直接退出并打出中文说明 |
| **模型下载** | `sentence-transformers` 首次会去 HF 拉模型；容器内没网就是启动失败 | **构建期把 `bge-small-zh-v1.5`（93 MB）烤进 `/opt/models`**，运行时零下载。重排模型（1.1 GB，默认关）不烤 |
| **数据持久化** | `state/ uploads/ chroma_db/` 都在工作目录下，容器一删全没 | 统一挪到 `/data`（`STATE_DIR/UPLOAD_DIR/VECTOR_DB_PATH` 三个环境变量覆盖），compose 用命名卷 `ragflow-data:/data` |

## 二、镜像（实测 2.82 GB）

- 基础 `python:3.11-slim`（对齐 `requires-python>=3.11`）；
- **torch 先按 CPU 源装**（`--index-url https://download.pytorch.org/whl/cpu`，装到 `2.15.0+cpu`，
  `torch.cuda.is_available() = False`）—— 默认 PyPI 的 Linux torch 会拖进 CUDA 依赖，镜像白胖约 2 GB；
  CPU 源不可达时退回默认源（能跑优先）；
- 依赖层与代码层分开：改业务代码不重装 1 GB 依赖；
- 非 root 运行（uid 10001）、`tini` 做 PID 1（`docker stop` 的信号能正确转给 uvicorn）、
  `HEALTHCHECK` 打 `/api/session`。

**健康检查为什么打 `/api/session` 且不用 `curl -f`**：项目还没有 `/healthz`（已记进待办），
而 `/api/session` 在设了 `AUTH_TOKEN` 时返回 401 —— `curl -f` 会把 401 当失败，服务活着也判 unhealthy。
所以只要求「有 HTTP 响应且状态码 ∈ {200,401,403}」。

## 三、冒烟验收（全部实测）

| # | 判据 | 结果 |
|---|---|---|
| 1 | 不设 `AUTH_TOKEN` 起容器 → 拒绝启动，并打出可读的中文说明 | ✅ 实测，退出码非 0 |
| 2 | 无凭证访问 `/api/session` → **401** | ✅ |
| 3 | 带 Basic 凭证 → 200 + `ragflow_owner` HttpOnly cookie | ✅ |
| 4 | 首页 `/` → 200（71 449 字节） | ✅ |
| 5 | 上传 txt → 解析 + 切片 + **容器内离线向量化**成功 | ✅ 返回 `file_names / has_original` |
| 6 | 数据落在 `/data/{state,uploads,chroma}` | ✅ 实测 8 K / 16 K / 416 K |
| 7 | `docker restart` 后 `/api/state` 仍返回那条对话（cookie 有效） | ✅ 200 + 原对话 |
| 8 | 健康检查 | ✅ 起容器 5 s 后 `healthy` |

> 顺带踩到一个**测试工具**的坑：PowerShell 的 `Invoke-WebRequest` 同时带 `Authorization` 和
> 手写 `Cookie` 头时会把 Cookie 丢掉，表现为「重启后 403」。换 `curl.exe` 立刻 200 ——
> **不是服务的问题**。以后冒烟一律用 curl，别用 PS 手搓头。

## 四、配套守卫（4 条新测试）

`tests/test_repo_hygiene.py` 里钉死四个**会出事**的性质，而不是「文件存在就算过」：

1. Dockerfile 基础镜像跟上 `python:3.11`、必须 `HOST=0.0.0.0`、**不许给 `AUTH_TOKEN` 默认值**；
2. 非 root 运行 + 有健康检查 + `/data` 是 volume；
3. `.dockerignore` 必须排除 `.env` / `.venv` / `state` / `uploads` / `chroma_db` / `.git`；
4. compose 必须挂命名卷、覆盖成容器内监听、默认只发布到宿主机回环。

`pytest` **835 → 839 passed**。

## 五、仍然没做

- **没有 `/healthz`**：健康检查只能借 `/api/session`（顺带说明「401 也算活着」这种别扭写法）；
  补一个专门的健康端点应进下一轮 P0；
- **没有 CI 构建镜像**：现在只在本地 build 过，没在 CI 里 build + 冒烟（要多花 ~5–10 min/次）；
- **没有镜像瘦身**：chromadb + transformers + torch 是硬成本，2.82 GB 里大头是 torch；
  真要压可以换 `python:3.11-slim` + `--no-cache` 之外的手段，或改用 ONNX embedding（工程量大，不值）。
