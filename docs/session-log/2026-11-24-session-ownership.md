# 2026-11-24 会话归属：HttpOnly cookie + 服务端令牌（待办 ⑦ 第二刀）

> 上一刀把「匿名可访问」堵住了（AUTH_TOKEN 逐请求校验）。这一刀堵的是剩下的真洞：
> **`session_id` 是客户端生成的**，服务端对任何 sid 都无条件加载状态 —— sid 一泄漏就能读那份会话。

## 一、做法

1. `/api/session` 下发 **HttpOnly cookie**，值 = `sid.token`（`samesite=lax`、一年有效）；
   状态文件里**只存 `sha256(token)`** —— 状态文件被拷走也伪造不出 cookie。
2. 其余接口一律走 `_session_or_403(request, sid)`：cookie 里的 sid 与 token 都对得上才放行，否则 403。
3. `/api/session` 一次调用处理三种情况：**新建**（没 cookie 没 sid）/ **凭 cookie 恢复**（刷新、重开浏览器）/
   **旧会话首次绑定**（本刀之前建的会话，状态里没有令牌 → 发一把钥匙并落盘）。
4. 前端启动时把 localStorage 里的 sid 一起带上（`/api/session?session_id=...`）；遇 403 就丢掉本地 sid
   重开会话 —— 归属校验失败从此能自愈，而不是卡在错误里。

## 二、判据与验证

| # | 判据 | 结果 |
|---|---|---|
| 1 | cookie 是 HttpOnly + SameSite=Lax | 单测读 `Set-Cookie` |
| 2 | 状态文件里**没有明文令牌**，只有 64 位 sha256 | 单测扫文件内容 |
| 3 | 别人的 sid 直接请求 → **403** | 单测 + 真 HTTP 冒烟 |
| 4 | 伪造 token / A 的 cookie 开 B 的会话 → 403 | 单测 |
| 5 | 旧会话首次绑定后，无 cookie 进不来 | 单测 |
| 6 | 同一浏览器再问 `/api/session` → **恢复原会话**（`resumed: true`） | 单测 |
| 7 | 变异检查全红 | **5/5**（归属恒真 / 只比 sid / 存明文 / 非 HttpOnly / 不绑定旧会话） |
| 8 | 真 uvicorn + curl 冒烟 | **10/10**（无 token 模式 5 项、有 token 模式 4 项、公开监听拒绝启动 1 项） |

`pytest` 814 → **822**。

## 三、被迫改的既有测试（行为真的变了）

- 预览隔离两个用例原来用**一个** `TestClient` 模拟两个会话 —— 现在 `/api/session` 会**凭 cookie 恢复**
  同一个会话，所以「同一个 client 拿两个 sid」已经不成立了。改成两个 `TestClient`（= 两个 cookie jar），
  断言也从 404 放宽到 `403/404`（归属校验现在更早拦下来）。
- `test_auth` 的写操作用例、`test_security_runtime` 的限流用例：先 `/api/session` 拿会话再写。
- 这三处不是「改测试迁就代码」，而是**测试模型要与真实浏览器一致**：一个 client 就是一个浏览器。

## 四、仍然没解决

- **多用户**：`AUTH_TOKEN` 是共享钥匙，没有账号 / 权限分离 / 审计。
- **旧会话首次绑定的窗口**：本刀之前建的会话没有令牌，谁先带着那个 sid 来谁就绑上 —— 对已经泄漏的旧 sid
  是一次性风险。想干净就删旧状态文件重开。
- **限流仍是进程内内存**（多实例不共享）；**结构页阈值换语料要重校**。
- cookie 没设 `Secure`（本机 http 要能用）—— 对外部署套 HTTPS 后可以加上。
