# 前端 vendor 库

这些是「查看文件 → 原文件预览」用的第三方浏览器端库，**本地内置、不依赖 CDN**
（项目自 commit 4dabb6b 起刻意去掉了所有 CDN 依赖，避免网络加载失败）。
由 `src/server.py` 的 `GET /vendor/{name}` 只读路由提供，`index.html` 按
`jszip → docx-preview → xlsx` 的顺序加载。

| 文件 | 包 | 版本 | 许可证 | 用途 |
|---|---|---|---|---|
| `jszip.min.js` | [jszip](https://www.npmjs.com/package/jszip) | 3.10.2 | MIT 或 GPLv3（本项目按 MIT 使用） | docx-preview 的外部依赖，解压 OOXML |
| `docx-preview.min.js` | [docx-preview](https://www.npmjs.com/package/docx-preview) | 0.4.1 | Apache-2.0 | 浏览器内渲染 `.docx` |
| `xlsx.core.min.js` | [SheetJS xlsx](https://www.npmjs.com/package/xlsx) | 0.18.5 | Apache-2.0 | 解析 `.xlsx` 并转成 HTML 表格 |

许可证全文见同目录 `LICENSE-*`。

更新方式：从 npm registry 下载对应版本 tarball，取 `dist/` 下的 min 文件覆盖，
并同步更新上表；不要引入需要额外构建步骤的版本。注意 `docx-preview` 的 UMD
把 `JSZip` 当外部依赖，必须保证 `jszip.min.js` 先加载。
