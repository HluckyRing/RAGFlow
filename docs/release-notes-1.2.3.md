# v1.2.3 — Docker 交付 + 数据备份 + 删文件从 3 分钟到 0.08 秒

> 自 v1.2.2 起 8 个提交：**官方 Docker 交付**（镜像 / 更新 / 备份 / 端口可配）+ 一条影响手感的
> 性能修复。**没有破坏性变更**；已有的 `state/` 与 `uploads/` 直接可用（本版没改数据格式）。
> 回退：`git checkout v1.2.2`，或退掉 Docker 用 `python server.py`。

## 亮点

### 1. 删一个文件不再重建整份索引（4.37 s → 0.08 s）

真实反馈：「上传错文件想删，删除很慢也没自动刷新，得手动刷新。」实测确认是两层问题叠加：

| 操作 | 改前 | 改后 |
|---|---|---|
| 删 1 个文件（剩 ~17 块） | **4.37 s** | **0.08 s** |
| 上传 250 块（26 万字） | 估 ~62 s | **13.6 s** |
| 811 块的真实语料删一个文件 | 约 **3 分钟** | ~0.1 s |

- 删除原来走 `_rebuild_collection` —— 把**剩下所有文件**重新切片 + 重新 embedding，
  而删除根本不需要 embedding。现在只删该文件的 chunk（`delete(where={"file_name": ...})`），
  **并且删完数一遍块数**：老索引没有 `file_name` 元数据时 where 会静默删 0 条（不报错），
  那种情况自动退回整份重建 —— 宁可慢，也不能让检索召回已删文件的内容。
- `_rebuild_collection` 改成**批量** embedding（一次 64 块），上传大文件同样受益。
- 前端：确认弹窗**等操作完成才关**（按钮禁用 +「处理中…」），删除**先本地移除再等服务端**、
  失败自动回滚 —— 慢操作没有反馈，在用户眼里就是 bug。

### 2. 官方 Docker 交付（`Dockerfile` + `docker-compose.yml`）

```bash
# .env 里加一行 AUTH_TOKEN（容器必然对外监听，没有它拒绝启动）
docker compose up -d --build      # 首次构建约 5–10 分钟
```

- 镜像 **2.82 GB**：CPU 版 torch（不带 CUDA，省约 2 GB）+ 依赖 + **构建期烤进镜像的向量模型**
  （运行时零下载）；
- 非 root 运行、`tini` 做 PID 1、健康检查、`/data` 命名卷（删容器不删数据）；
- **只发布到宿主机回环** `127.0.0.1:8080`，对外请自己套 HTTPS 反代。

### 3. 更新与备份都有脚本了

```bash
python scripts/update_deploy.py            # git pull → 重建镜像 → 等真健康（层缓存热时 ~20 s）
python scripts/update_deploy.py --backup   # 顺手备份数据卷
python scripts/backup_data.py              # 停容器 → 打包 → 启容器 → 保留最近 7 份
```

`docker compose up -d` **不会**因为源码变了就重建镜像 —— 这是最容易踩的坑，所以做成了脚本。
备份产物在 `backups/ragflow-data-<时间戳>.tar.gz`（实测 140 MB 数据 → 约 123 MB），
`backups/` 已被 `.gitignore` / `.dockerignore` 挡住。

### 4. 改端口只动一个地方

`.env` 里 `HOST_PORT=9090` 即可（容器内固定 8080）。本地直接跑仍然用 `PORT`，两套互不干扰。

## 验证

- `pytest`：**872 passed**（v1.2.2 时 835；新增删除/备份/端口/YAML 结构等守卫）
- 变异检查：删除提速 5/5 红、备份 3 项、compose 结构 1 项
- Docker：镜像构建 → 起容器 → 登录/上传/离线向量化/重启后数据仍在 → `healthy`，全链路实测

## 升级注意

1. **Docker 用户**：`git pull && python scripts/update_deploy.py`（会重建镜像，数据不受影响）；
2. **浏览器**：前端有改动，升级后请 **Ctrl+F5 硬刷新**一次；
3. 首次用 Docker 要生成 `AUTH_TOKEN`（`python -c "import secrets;print(secrets.token_hex(16))"`）；
4. 删文件现在毫秒级，但**对话里已生成的回答不会被追溯删除**（有意为之）。

## 已知限制

- 上传还没有进度条（250 块要 13 s，只能看「正在解析」提示）；
- 未做多用户账号体系：`AUTH_TOKEN` 是一把共享钥匙；
- 覆盖说明的缺口清单仍不判轻重；`md`/`docx` 的「页标题」其实是块首行。
