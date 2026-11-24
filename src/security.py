# -*- coding: utf-8 -*-
"""启动安全闸 + 接口限流。

这个服务本来是**单机单人**用的：没有鉴权、任何人都能读写全部对话、每次提问都要花钱调
大模型。唯一的护栏是「默认只绑回环」（`HOST=127.0.0.1`）。但只要有人照着
`AGENTS.md` 的说明改成 `HOST=0.0.0.0` 对外提供，护栏就整个消失了 —— 而这件事
**不会报错、也不会提示**，属于「不安全配置可以悄悄发生」。

所以这里做两件事，都是 fail-closed：

1. `startup_error()`：监听了非回环地址却**没设 `AUTH_TOKEN`** → 拒绝启动（退出码 2）。
   本机回环 + 无 token 仍然零摩擦，跟改造前一模一样；
2. `SlidingWindowLimiter`：给 `/api/chat` 这种「每问一次都要花钱」的接口兜一层限流，
   防止把 API 额度刷光。按客户端 IP 计数（按 session_id 计数没有意义：那是客户端自己
   生成的，换一个就绕过了）。

两个都是纯函数 / 纯对象（时钟可注入），不依赖 FastAPI，单测不拉起服务。
"""
import base64
import hmac
import time
from collections import deque

# 回环地址的三个写法（Windows 上 localhost 也解析到 127.0.0.1）
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1")


def is_loopback(host):
    """这个监听地址是不是只对本机开放。"""
    return str(host or "").strip().lower() in LOOPBACK_HOSTS


def startup_error(host, auth_token):
    """启动前自检：返回 `None` 表示可以启动，否则返回中文错误说明。

    判据只有一条 —— **对外监听就必须有 token**。这是「不安全配置不许悄悄发生」的落地：
    与其在文档里提醒，不如让它起不来。
    """
    if is_loopback(host):
        return None
    if str(auth_token or "").strip():
        return None
    return (
        f"拒绝启动：HOST={host} 会把这个**没有鉴权**的服务暴露到本机之外，"
        f"而 AUTH_TOKEN 没有设置。\n"
        f"  二选一：① 改回 HOST=127.0.0.1（只本机用，默认）；"
        f"② 设 AUTH_TOKEN=<一串足够长的随机字符> 后再对外提供。\n"
        f"  提醒：这个服务能让任何访问者读取全部对话、上传文件、并消耗你的大模型额度。"
    )


def ensure_startup_safe(host, auth_token):
    """自检不过就 `sys.exit(2)`（两个启动入口共用一个出口，别各写一份）。"""
    message = startup_error(host, auth_token)
    if message:
        raise SystemExit(message)


class SlidingWindowLimiter:
    """滑动窗口限流：`window_seconds` 内同一个 key 最多 `limit` 次。

    为什么是滑动窗口而不是固定窗口：固定窗口在边界上会放过两倍突发
    （59 秒打满一窗、61 秒再打满下一窗）。这里只有几十条记录，实现成本一样。

    时钟可注入（`now`），所以单测不用真的等一分钟。
    """

    def __init__(self, limit, window_seconds=60.0):
        self.limit = max(0, int(limit))
        self.window = float(window_seconds)
        self._hits = {}

    def allow(self, key, now=None):
        """记一次并判断放不放行。`limit<=0` 表示不限流。"""
        if self.limit <= 0:
            return True
        stamp = time.monotonic() if now is None else float(now)
        queue = self._hits.setdefault(str(key), deque())
        cutoff = stamp - self.window
        while queue and queue[0] <= cutoff:
            queue.popleft()
        if len(queue) >= self.limit:
            return False
        queue.append(stamp)
        return True

    def retry_after(self, key, now=None):
        """还要等几秒才能再放行一次（给 429 的 `Retry-After` 用）。"""
        if self.limit <= 0:
            return 0
        stamp = time.monotonic() if now is None else float(now)
        queue = self._hits.get(str(key)) or deque()
        if len(queue) < self.limit:
            return 0
        return max(0.0, queue[0] + self.window - stamp)

    def reset(self):
        """测试与「配置热改」用：清空所有计数。"""
        self._hits.clear()


# ── AUTH_TOKEN 的逐请求校验（2026-11-24 第二十二轮，待办 ⑦ 第一刀）──
#
# 之前的洞：`AUTH_TOKEN` 只在**启动时**被检查（对外监听必须设），请求到达时一次都没校验过 ——
# 于是「设了 token 的对外部署」实际上仍然对任何人开放，还带着「我设过 token」的错觉。
# 这里补上校验：设了 token 就必须带 HTTP Basic 凭证（用户名随便填，密码 = token）。
#
# 为什么用 Basic 而不是自定义 header：浏览器会自己弹一次框，之后同源请求（fetch / SSE /
# `<img>` 预览）都自动带凭证，**前端一行都不用改**。代价是浏览器会自动附带凭证，所以必须配
# CSRF 检查（见 `origin_allowed`）。

AUTH_SCHEME = "basic"


def basic_auth_ok(authorization, token):
    """`Authorization` 头是否携带正确的 token。

    - `token` 为空 = 没开鉴权 → 永远放行（默认回环单机，零摩擦，行为与改造前一致）；
    - 缺失 / 不是 Basic / 解不开 / 密码不对 → `False`；
    - 比较用 `hmac.compare_digest`（定长比较，不按前缀提前返回）。
    """
    expected = str(token or "").strip()
    if not expected:
        return True
    header = str(authorization or "").strip()
    scheme, _, payload = header.partition(" ")
    if scheme.lower() != AUTH_SCHEME:
        return False
    try:
        decoded = base64.b64decode(payload.strip(), validate=True).decode("utf-8", "replace")
    except Exception:                      # noqa: BLE001 —— 任何畸形凭证都算失败
        return False
    _username, sep, password = decoded.partition(":")
    if not sep:
        return False
    return hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8"))


def origin_allowed(origin, host, method):
    """CSRF 检查：写操作的 `Origin` 必须与请求的 Host 同源（或缺失）。"""
    verb = str(method or "").upper()
    if verb not in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    value = str(origin or "").strip()
    if not value:
        return True                        # curl / 脚本：匿名请求已经被 Basic 挡在门外
    target = str(host or "").strip().lower()
    if not target:
        return False
    try:
        from urllib.parse import urlsplit
        netloc = urlsplit(value).netloc.lower()
    except Exception:                      # noqa: BLE001
        return False
    return bool(netloc) and netloc == target

