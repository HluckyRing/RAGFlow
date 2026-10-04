# 前端 vendor 库

这些是「查看文件 → 原文件预览」用的第三方浏览器端库，**本地内置、不依赖 CDN**
（项目自 commit 4dabb6b 起刻意去掉了所有 CDN 依赖，避免网络加载失败）。
由 `src/server.py` 的 `GET /vendor/{name}` 白名单路由提供。

| 文件 | 包 | 版本 | 许可证 | 用途 |
|---|---|---|---|---|
| `jszip.min.js` | [jszip](https://www.npmjs.com/package/jszip) | 3.10.2 | MIT 或 GPLv3（本项目按 MIT 使用） | docx-preview 的外部依赖，解压 OOXML |
| `docx-preview.min.js` | [docx-preview](https://www.npmjs.com/package/docx-preview) | 0.4.1 | Apache-2.0 | 浏览器内渲染 `.docx` |
| `xlsx.core.min.js` | [SheetJS xlsx](https://www.npmjs.com/package/xlsx) | 0.18.5 | Apache-2.0 | 解析 `.xlsx` 并转成 HTML 表格 |
| `aiden0z-pptx-renderer.browser.es.js` | [@aiden0z/pptx-renderer](https://www.npmjs.com/package/@aiden0z/pptx-renderer) | 1.3.0 | Apache-2.0 | 浏览器内渲染 `.pptx`（含图表/图片，内嵌 PDF 回退未启用） |

许可证与第三方声明见同目录：

- `LICENSE-docx-preview.txt`、`LICENSE-SheetJS.txt`、`LICENSE-jszip.markdown`
- `LICENSE-aiden0z-pptx-renderer.txt`、`THIRD_PARTY_NOTICES-aiden0z-pptx-renderer.md`
- `aiden0z-pptx-renderer-licenses/`（其打包内含的 MPL-2.0 等第三方许可证）

加载方式：`jszip → docx-preview → xlsx` 三个 UMD 由 `index.html` 末尾的
`<script src>` 按序加载；PPTX 的浏览器 ESM bundle 不走 `<script>`，而是前端在
真正要渲染 PPTX 时动态 `import('/vendor/aiden0z-pptx-renderer.browser.es.js')`，
避免首屏为一个不常用功能多下 1.8MB。

更新方式：从 npm registry 下载对应版本 tarball，取 `dist/` 下的目标文件覆盖，
同步更新上表与许可证文件；不要引入需要额外构建步骤的版本。注意 `docx-preview`
的 UMD 把 `JSZip` 当外部依赖，必须保证 `jszip.min.js` 先加载。
