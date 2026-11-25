# 2026-11-25 安全两把小刀：cookie `Secure` + 旧会话迁移窗口（待办 ⑦ 遗留）

> 待办 ⑦ 的两大刀（2026-11-24：`AUTH_TOKEN` 逐请求校验、会话归属 HttpOnly cookie）做完后，
> 自己留了两条小尾巴：**cookie 没设 `Secure`**、**旧会话首次绑定的抢先绑定窗口**。
> 这轮把这两条收掉，然后发 v1.2.2（顺带验通真·Actions 发版链路，闭环待办 ⑧）。

## 一、cookie 的 `Secure`（`COOKIE_SECURE=auto|1|0`）

**病根**：`_attach_owner` 以前硬编码 `httponly=True, samesite="lax"`，**没有 `secure`**。
对外套 HTTPS 之后，这个带会话钥匙的 cookie 仍可能被明文信道带出去。

**但不能无脑加**：本机 http 调试时浏览器会**丢掉** `Secure` cookie，表现成「一刷新会话就没了」——
这正是当初没加它的原因。两头都要照顾，所以做 `auto`：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 看**这次请求的真实 scheme**：直连看 `request.url.scheme`；反代看 `X-Forwarded-Proto` 的**第一个值**（多级代理会追加，`https, http` 取 `https`） |
| `1` | 强制带 `Secure`（容器里 scheme 常常看不出来时用） |
| `0` | 强制不带 |

顺带写清楚一条边界：`X-Forwarded-Proto` **不参与任何安全判断**，只决定 cookie 要不要带标记 ——
判断反了最多是「该带的没带」，不会放宽归属校验。

## 二、旧会话的「首次绑定」窗口（`SESSION_BIND_LEGACY=1|0`）

**病根**：本刀（2026-11-24）之前建的会话状态里没有 `session_token_hash`。`/api/session` 为了
不让老会话当场失效，对「带 sid 来、状态里没令牌」的会话**发一把钥匙并绑定** ——
对**已经泄漏过的旧 sid**，这是一次抢先绑定窗口（谁先来谁绑走）。

**做法**：默认 1（继续迁移，升级平滑），可以设 0 关掉；关掉后旧 sid **换新会话**
（`resumed: False`，旧数据不删、只是拿不到钥匙）。前端本来就有「拿到新 sid 就切过去」的自愈路径，
所以关掉不报错。

真正的干净做法仍是运维动作：**把老 `state/*.json` 清掉再设 0**。这条不替用户做。

## 三、判据（6 条新单测 + 4 刀变异）

| # | 判据 | 结果 |
|---|---|---|
| 1 | 本机 http 上归属 cookie **不带** `Secure` | 单测 |
| 2 | 反代 `X-Forwarded-Proto: https` 时**带** `Secure` | 单测 |
| 3 | 多级代理只认第一个值（`https, http` → 带；`http, https` → 不带） | 单测 |
| 4 | `COOKIE_SECURE=1` / `=0` 压过 scheme 判断 | 单测 |
| 5 | `SESSION_BIND_LEGACY=0`：旧 sid 拿到**新会话**，且状态里不出现令牌哈希 | 单测 |
| 6 | 默认仍放行（升级后老会话不失效） | 单测 |

变异检查（`tests/test_auth.py`）：

| 变异 | 结果 |
|---|---|
| A cookie 永不带 `Secure` | 红 ✅ |
| B cookie 永远带 `Secure` | 红 ✅ |
| C 不认 `X-Forwarded-Proto` | 红 ✅ |
| D 迁移开关失效（总是绑定旧会话） | 红 ✅ |

`pytest` **829 → 835 passed**。

## 四、教训

- **「安全问题先记进待办」容易烂在待办里**：这两条都不难，但都挂在「等以后套 HTTPS」上。
  真正的成本不是写代码，是**想到要写**——把它们和发版绑在一起做，才有闭环。
- **默认值要照顾「升级不会打脸用户」**：`SESSION_BIND_LEGACY` 默认 1 而不是 0，是因为设 0 会让
  所有老会话当场失效；安全默认与可升级性冲突时，用开关把选择权交出去，并把风险写在 `.env.example` 里。
- **`Secure` 这种标记必须能按请求判断**：写死任一边都会毁掉一种部署形态（http 调试 / https 生产）。

## 五、发版：v1.2.2（顺带验通真·Actions 链路）

tag `v1.2.2` 推上去后 Release 工作流 **23 s 成功**（run `45604780571`），这是 2026-11-23 那套
「正文取仓库手写 notes」的第一次**真链路**验证（此前只用脚本层验过）：

| 判据 | 结果 |
|---|---|
| 工作流里 `check_release_notes.py --tag` 通过 | ✅（缺文件 / 太短会红） |
| Release 正文 = `docs/release-notes-1.2.2.md` 全文 | ✅ 2 474 字符，**没有**自动生成的 Full Changelog |
| 附件 | ✅ `ragflow-1.2.2-py3-none-any.whl`（805 011 字节；本地 build 抽查 37 entries / index.html / 12 vendor / Version 1.2.2） |
| `gh release list` | ✅ v1.2.2 显示 **Latest** |
| main 上的 CI | ✅ success |

待办 ⑧（发版小债）至此**闭环**。

## 六、仍然没做

- 多用户账号 / 权限分离 / 审计（`AUTH_TOKEN` 仍是共享钥匙）；
- 限流换共享存储（多实例）；
- 结构页阈值（4-gram 0.6 / 跨 3 份文件）换语料要重校。
