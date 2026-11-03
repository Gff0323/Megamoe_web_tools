# roofline_web 项目约定（AGENTS.md）

## 唯一维护源

- 本工具的正式维护目录是独立 worktree `C:\MegaMoe\roofline_web`（分支 `roofline-web-tool`），
  页面入口 `test/bigop_fused/roofline_web/index.html`。保持**单 HTML、无外部运行依赖**。
- 其他 worktree / 分支下的同名文件是历史副本，只编辑本 worktree，不做双向同步。
- 本地起服务：`python -m http.server 8791 --directory <worktree>/test/bigop_fused/roofline_web`。
- 布局公式的权威来源是本 worktree 内的
  `kernels/mega_moe_bf16/include/mega_moe_training_workspace.h` 与
  `test/megamoe_bf16_test/test_workspace_layout.py`（跟随 origin/train 更新，勿凭记忆改公式）。

## 交付纪律（每次修改必守）

- 交付前必跑两项校验，全绿才算完成：
  1. `node smoke.cjs` —— 静态守卫 + 运行时渲染守卫（自动桩 DOM 真跑渲染双分支）；
  2. `node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json` —— 数值对账。
- 数值修改必须先改权威来源 / 导出脚本（`export_layout_reference.py` → `ref/`），再同步页面模型，
  禁止手调页面常数去凑显示值。
- 每次交付区分三态并如实报告：**本地提交 / 远端推送 / 线上可访问**（本机 8791 或 GitHub Pages）。
  推送成功不能代替上线验收——从服务端重新拉取并核对内容。
- 渲染函数内不得引用模型函数的局部变量（作用域隔离：`memModel` 的 P/E/F/S 只在函数内）。

## 仓库与发布（GitHub Pages 启用后生效）

- 发布入口 `.github/workflows/pages.yml`：仅 main 且 `index.html` 变更时自动发布；
  产物只含 `index.html + .nojekyll`，工作流内 `cmp + sha256` 自校验。
- 发布后跟踪 Actions 运行，并从公网核对线上 HTML 与源文件 SHA-256 一致后再报告"已上线"。
- 独立发布仓与内部 monorepo 分工：monorepo（gitcode）承载公式源码与历史；
  GitHub 独立仓只承载页面与验证资产，用 `publish-github.ps1` 从本目录单向生成，不双向同步。

## 安全

- 内部布局细节（saved-state 偏移、SHMEM 头、backward workspace 公式）具有公司内部属性：
  发布仓库选公开/私有前必须经负责人确认；凭据、内网地址、主机名不入 Git、不入聊天记录。
- 内部 monorepo（gitcode）凭据不用于 GitHub 仓库，两套凭据严格分开。
