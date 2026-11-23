# -*- coding: utf-8 -*-
"""发版守卫：标签对应的 `docs/release-notes-<版本>.md` 必须存在，缺了就让工作流红。

为什么要有它（2026-11-23）：`gh release create --generate-notes` 在「提交全是直推 main、
没有 PR」的仓库里抓不到东西，v1.1.0 / v1.2.0 / v1.2.1 三次发出来的 Release 正文都只有一行
`**Full Changelog**`，只能事后 `gh release edit --notes-file` 手工补。把「读哪份 notes」
搬进这个脚本，是为了让这段逻辑能被 pytest 直接测（工作流里的 bash 在本地跑不了）。

用法（工作流里被调用，也手工可用）：

    python scripts/check_release_notes.py --tag v1.2.1
    # → 通过时把 NOTES=docs/release-notes-1.2.1.md 写进 $GITHUB_ENV，供后续 gh release create 用
    # → 缺文件时打印 ::error:: 并以退出码 1 结束（发版停在这一步，不发空正文的 Release）
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_CHARS = 400


def notes_path(tag, root=ROOT):
    """标签 → notes 文件路径：`v1.2.1` → `docs/release-notes-1.2.1.md`。"""
    version = str(tag or "").strip()
    if version.startswith("v"):
        version = version[1:]
    return pathlib.Path(root) / "docs" / f"release-notes-{version}.md"


def check(tag, root=ROOT):
    """检查并返回 (是否通过, 说明, 路径)。缺文件与内容过短都算不通过。"""
    path = notes_path(tag, root)
    if not path.is_file():
        return False, (f"缺少 release notes：{path.name}（应放在 docs/ 下）—— "
                       f"发版前先写一份，模板见 docs/release-notes-1.2.1.md，"
                       f"提交推上来再打标签。"), path
    body = path.read_text(encoding="utf-8")
    if len(body.strip()) < MIN_CHARS:
        return False, (f"{path.name} 只有 {len(body.strip())} 字符（至少要 {MIN_CHARS}）—— "
                       f"太短的正文等于没有正文。"), path
    return True, f"使用 {path.name}（{len(body)} 字符）作为 Release 正文", path


def main(argv=None):
    parser = argparse.ArgumentParser(description="发版守卫：标签必须有对应的 release notes")
    parser.add_argument("--tag", required=True, help="触发本次构建的标签，如 v1.2.1")
    parser.add_argument("--github-env", default=os.getenv("GITHUB_ENV"),
                        help="把 NOTES=<路径> 追加到这个文件（GitHub Actions 的 $GITHUB_ENV）")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录（测试用）")
    args = parser.parse_args(argv)

    ok, message, path = check(args.tag, args.root)
    if not ok:
        print(f"::error::{message}", file=sys.stderr)
        return 1

    # 用 "/" 拼：这份值会写进 $GITHUB_ENV 给 ubuntu 上的 gh 用，别跟着跑测试的机器变成反斜杠
    relative = f"docs/{path.name}"
    if args.github_env:
        with open(args.github_env, "a", encoding="utf-8") as handle:
            handle.write(f"NOTES={relative}\n")
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
