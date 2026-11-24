# 2026-11-24 AUTH_TOKEN 逐请求校验 + CSRF（待办 ⑦ 第一刀）

> 待办原文：「安全侧大头：真身份体系（账号 + 会话归属 + CSRF）、限流换共享存储、结构页阈值重校」。
> 这一刀只堵**最大的那个洞**：`AUTH_TOKEN` 以前**只在启动时**被检查，请求到达时从来没校验过。

## 一、读代码读出来的洞

- `src/security.startup_error()` 只保证「对外监听必须设 token」；
- 服务端**没有任何一处**读 `Authorization` / 校验 token —— 也就是说
  `HOST=0.0.0.0` + 设了 `AUTH_TOKEN` 的部署，**对外仍然完全开放**，还带着「我设过 token」的错觉。
  这比「没有身份体系」更糟：它给的是假的安全感。
- 第二层洞（本轮**没**修）：`session_id` 由客户端生成并存在 `localStorage`，服务端对任何 sid
  都无条件加载状态 —— sid 一旦泄漏（它天天出现在 URL 里），别人就能读到那份会话。

## 二、做法（不动数据库、不加登录页）

1. **HTTP Basic 逐请求校验**：设了 `AUTH_TOKEN` 就要求 `Authorization: Basic base64(任意用户名:token)`。
   为什么用 Basic：浏览器会自己弹一次框，之后所有同源请求（fetch / SSE / `<img>` 预览）自动带凭证，
   **前端一行都不用改**；401 + `WWW-Authenticate` 是标准行为。
2. **CSRF**：Basic 凭证由浏览器自动附带，所以写操作（POST/PUT/PATCH/DELETE）带 `Origin` 且与
   请求 Host 不同源 → 403；不带 `Origin`（curl/脚本）放行（匿名已被第 1 条挡住）。
3. **默认路径不动**：没设 token（默认回环单机）时**零摩擦**，行为与改造前逐位一致。
4. 密码比较用 `hmac.compare_digest`；错误响应体里**不含 token**。

判据与结果：无凭证 401（带 `WWW-Authenticate`）、错凭证/畸形凭证 401、正确凭证 200、
首页也受保护、跨站写 403、同源或缺 `Origin` 放行、GET 不做 Origin 检查、错误体不泄漏 token、
比较确实走 `compare_digest`。**变异检查 4/4 全红**（去掉校验 / 放行任意 Origin / 用 `==` 比较 /
没设 token 也要求凭证）。`pytest` 801 → **814**。

## 三、三个坑（都记下来）

1. **测试直接跑 `TestClient(server.app)` 会写真实工作区**：`/api/conversations` 按 sid 建状态文件、
   还建向量集合 —— 我第一版没隔离，跑一次测试留下 **17 个空会话文件 + 5 个空集合**，已逐个核对
   内容（全部 0 消息）后删除。教训：**测试里碰 `server` 之前先隔离 `state.STATE_DIR` 与
   `retrieval._chroma_client`**，这不只是「干净」问题，`state/*.json` 是用户数据目录。
2. **顺序依赖的守卫会假红**：`test_eval_retrieval` 里断言 `"torch" not in sys.modules` ——
   只要前面有任何一个用例建过 `TestClient`（import `src.server` → `sentence_transformers` → torch）
   它就挂。改成比较调用前后的**差集**，测的还是同一件事（main 的输入校验没触发模型加载），
   但不吃别的用例的副作用。
3. **「读源码找字符串」的守卫会被 docstring 骗**：第一版用
   `"hmac.compare_digest" in open(security.__file__).read()` 检查时序安全比较，
   而函数 docstring 里就有这串字 —— 变异检查 C 因此**假绿**。改成 monkeypatch 一个 spy 调
   `hmac.compare_digest`，用**行为**证明它真的被调用。

## 四、这一刀**不**解决（写清楚，别以为已经安全了）

- **会话归属**：sid 泄漏仍能读别人的会话。要做 HttpOnly cookie + 服务端令牌（状态里存一份），
  并改前端取会话的方式 —— 单独一轮。
- **多用户**：`AUTH_TOKEN` 是一把**共享钥匙**，没有账号、没有权限分离、没有审计。
- **Basic 是 base64 明文**：对外必须套 HTTPS，否则 token 等于明文传输。
- **限流仍是进程内内存**：多实例不共享（与 2026-11-12 那条待办同源）。
