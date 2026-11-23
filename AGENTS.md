# megamoe_web_tools 项目约定（AGENTS.md）

## 唯一维护源（独立仓）

- 本仓库就是网页工具的**唯一维护源**：`C:\MegaMoe\megamoe_web_tools` = GitHub
  `Gff0323/Megamoe_web_tools`（私有），分支 `main`，站点入口 `index.html`。
- **不再**在 MegaMoe 主仓的 `test/bigop_fused/roofline_web/` 下编辑页面——那里是历史副本，
  会随分支切换/clean 丢失。所有改动只提交到本仓并 push。
- 保持**单 HTML、无外部运行依赖**（离线双击可开）。
- 本地起服务：`python -m http.server 8791 --directory C:\MegaMoe\megamoe_web_tools`。

## 公式权威来源在 MegaMoe 仓（本仓不含）

- 布局公式权威：MegaMoe 仓的
  `test/megamoe_bf16_test/test_workspace_layout.py` 与
  `kernels/mega_moe_bf16/include/mega_moe_training_workspace.h`。
- 本仓的 `ref/*.json|csv` 是从那份源码导出的**快照**，导出方式：
  ```powershell
  cmd /c "python export_layout_reference.py --repo-root <MegaMoe 仓路径> > ref\layout_ref.json"
  ```
  （PowerShell 直接 `>` 会写 UTF-16，必须走 `cmd /c`）
- MegaMoe 代码更新后（尤其 `origin/train` 合入），**先重导参考值再跑对账**；
  不一致项即为页面模型需要跟进的地方，禁止手调页面常数去凑数。

## 交付纪律（每次修改必守）

1. `node smoke.cjs` —— 静态守卫 + 运行时渲染守卫（桩 DOM 真跑渲染）。
2. `node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json` —— 数值对账（当前基线 666 项）。
3. 提交并 push 到 `main`；Pages 自动发布后，**从公网拉取 index.html 核对 SHA-256** 再报告上线。
4. 报告时区分三态：本地提交 / 已推送 / 公网可访问。

## 渲染与代码约束

- 渲染函数内不得引用模型函数的局部变量（`memModel` 的 M/K/F/T/EP/EPR/S 只在函数内）。
- 符号体系与面板一致：面板字段名 M / K / F / topK / EP / EPR / R（maxRecvTokens）即全站记号；
  kernel 头文件的 P/E 只作注脚。
- 计算量系数：前向 6RFK；反向保存 gate/up 为 12RFK、重算 GMM1 为 16RFK；整层对应 18 / 22RFK。

## 安全

- 页面含内部布局细节（saved-state 偏移、SHMEM 头、backward workspace 公式）。
  仓库转公开前必须经负责人确认；凭据、内网地址、主机名不入 Git、不入聊天记录。
