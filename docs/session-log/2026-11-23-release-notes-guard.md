# 2026-11-23 发版正文硬门：Release 必须用仓库里的手写 notes（待办 ⑧）

> 起因：`--generate-notes` 在「提交全是直推 main、没有 PR」的仓库里抓不到东西 ——
> v1.1.0、v1.2.0、v1.2.1 **三次**发出来的 Release 正文都只有一行 `**Full Changelog**`，
> 每次都得事后 `gh release edit --notes-file` 手工补。这轮把它变成硬门。

## 一、改了什么

1. **逻辑搬进可测的脚本** `scripts/check_release_notes.py`：
   标签 → `docs/release-notes-<版本>.md`；缺文件或正文短于 400 字符 → 打印 `::error::` + 退出 1；
   通过时把 `NOTES=docs/release-notes-<版本>.md` 写进 `$GITHUB_ENV` 供后续步骤用。
   为什么搬进 Python：工作流里的 bash **本地跑不了**（这台机器的 `bash` 指向 WSL，没有发行版），
   放进脚本才能被 pytest 直接测。
2. **工作流只负责调用**：`.github/workflows/publish.yml` 增加一步
   `python scripts/check_release_notes.py --tag "$GITHUB_REF_NAME"`，然后
   `gh release create "$GITHUB_REF_NAME" dist/*.whl --title ... --notes-file "$NOTES"`。
   **`--generate-notes` 从工作流里删掉了**。
3. **守卫**（`tests/test_repo_hygiene.py`，+6 条）：工作流必须调脚本 + 用 `--notes-file` + 不许有
   `--generate-notes`；当前版本号必须有对应 notes（≥400 字符、有标题行、写了「已知限制」）；
   缺文件/太短要红；通过时 `NOTES` 真的写进 env 文件。

## 二、判据与验证

| 判据 | 结果 |
|---|---|
| 工作流用仓库 notes、不再自动生成 | 测试钉住（注释行不算，免得解释历史的注释被误判） |
| 缺 notes → 工作流红 | 脚本退出码 1 + `::error::` 带文件路径；测试 + 变异检查 B 变红 |
| 正文太短 → 红 | `MIN_CHARS=400`；变异检查 C 变红 |
| `NOTES` 传给后续步骤 | 单测断言写进 env 文件的内容逐字相等；变异检查 D 变红 |
| YAML 语法 | `yaml.safe_load` 解析通过（steps 五步、`on.push.tags` 正确）**手工验的**，没进测试 |

**变异检查 4/4 全红**（工作流退回自动生成 / 缺文件也放行 / 不检查长度 / 不写 NOTES）。

## 三、量不到什么（别把它当已验）

- **真·Actions 运行没验过**：这台机器没有能用的 bash，跑不了工作流的 shell 步骤；
  脚本逻辑被单测覆盖，但「`$GITHUB_ENV` → 下一步 `$NOTES`」这条链路要等**下一次真实打标签**才算验过。
  下次发版的检查清单里加一条：确认 Release 正文是手写 notes、且工作流没有多余的人工伤补。
- 只防「缺正文」，不防「正文写得比实测更满」—— 那靠人写，测试只卡了「已知限制」这一节在不在。
- 顺带一个坑：`sync_readme.py` 对新增的 `scripts/*.py` 要求 `DESCRIPTIONS` 里有一句话，
  这轮靠它拦了一次（README 结构树 + 测试一条）。新脚本必须先补说明再跑 `--write`。
