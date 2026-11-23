# megamoe_web_tools 项目约定（AGENTS.md）

## 唯一维护源（独立仓）

- 本仓库就是网页工具的**唯一维护源**：`C:\MegaMoe\megamoe_web_tools` = GitHub
  `Gff0323/Megamoe_web_tools`（私有），分支 `main`，站点入口 `index.html`。
- **不再**在 MegaMoe 主仓的 `test/bigop_fused/roofline_web/` 下编辑页面——那里是历史副本，
  会随分支切换/clean 丢失。所有改动只提交到本仓并 push。
- 保持**单 HTML、无外部运行依赖**（离线双击可开）。
- 本地起服务：`python -m http.server 8791 --directory C:\MegaMoe\megamoe_web_tools`。

## 公式权威来源：DeepEP（本仓只存镜像与快照）

- 算子代码的家在 **https://github.com/xixing-321/DeepEP**（`main`），
  布局公式在 `experiments/megamoe/csrc/kernels/mega_moe/common/`：
  `mega_moe_bf16_layout.hpp`、`mega_moe_workspace_backward.hpp`、`mega_moe_peermem.hpp`。
- 新仓**没有** python 布局规格，所以本仓维护 `layout_spec.py` —— 逐项复刻上面头文件的 python 镜像，
  头部记录对齐的 commit。**头文件一变，先改 layout_spec.py，再改页面。**
- 对账快照：`ref/layout_ref.json`（由 `python export_layout_reference.py > ref/layout_ref.json` 生成）。
- `probes/` 存 `roofline_probe.py` / `ep_probe.py`（零依赖），用于重导 `ref/r.csv`、`ref/e.csv`。
- SHMEM 走 **BF16 训练 MTE 路径**（`plan_bf16.cpp` 的 CompactTokenCountExchangeLayout +
  PeerPayloadLayout）；UDMA 后端只在 `codex/deepep-udma-backend` 分支，`main` 用 URMA peermem，未建模。
- 数值基线：smoke 78 项 + verify 666 项。

## 交付纪律（每次修改必守）

1. `node smoke.cjs` —— 静态守卫 + 运行时渲染守卫（桩 DOM 真跑渲染）。
2. `node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json` —— 数值对账（当前基线 667 项）。
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

