# MoE 理论 Roofline + 显存分析器（网页版）

两个纯理论工具的**实时交互版**，改配置即时出数，不再需要"跑一遍 Python → 生成 PNG/CSV → 打开看"：

- **⏱ 理论 Roofline 页**：`test/bigop_fused/roofline_probe.py` / `ep_probe.py` 的移植
- **🧠 显存分析页**：MegaMoe BF16 训练前向/反向的显存（HBM）账本，
  字节公式与 kernel 布局逐项一致，基于 docs《MegaMoe_BF16_前向持久Workspace第一版与反向适配》

纯理论模型（均衡路由 + 理想重叠假设），**不依赖 NPU**，全部计算在浏览器本地完成。

## 打开方式

任选其一：

```bash
# 方式 A：直接双击 index.html（file:// 即可用，零依赖）

# 方式 B：起个静态服务（推荐，方便多标签页/分享）
python -m http.server 8791 --directory test/bigop_fused/roofline_web
# 浏览器打开 http://127.0.0.1:8791/
```

## 功能

- **左侧配置面板**：模型 shape（M/K/topK/F 列表）、计算口径（整层 22RFK / 仅反向 16RFK）、
  cube TFLOPS、三级互连带宽（HCCS/柜内/跨柜，单向 GB/s）、拓扑（R_node/R_rack）、EP 扫描点
- **六张实时图**：
  1. 计算 vs 通信竞速（交叉点 = 受限状态切换）
  2. 融合加速上限 (C+N)/max(C,N)（2.0x 物理极限参考线）
  3. N:C 比值状态图（计算受限/通信受限分区）
  4. 绝对耗时：串行小算子 vs 理想重叠
  5. 各网络腿耗时 vs EP（对数轴，含瓶颈腿）
  6. 每向流量按网络层级拆分（堆叠柱状）

  图表布局可切 单列/双列/三列（双列默认，选择会记住）；**点击任意图表（或右上"放大 ⤢"）可全屏放大查看**
- **KPI 卡片**：每向毛流量、计算地板、单节点通信地板、最佳融合上限及其位置
- **全量数据表**：与 `roofline_probe.py` CSV 同名列，绿色高亮最佳融合点
- **导出/复现**：一键导出 CSV、复制等价 Python 命令行、复制分享链接（配置编码在 URL hash 里）

## 模型（与脚本逐行一致）

```
C = coef·R·F·K / cube_TFLOPS          # 全层 GEMM 时间，R = M·topK；整层 coef=22，仅反向=16
N = 4 × max(三条腿 流量÷带宽)          # 每层 4 趟（fwd/bwd × dispatch/combine）
四级命中概率：本卡 1/EP（不计网络）；节点内 (R_node−1)/EP → HCCS；
             柜内 → 柜内互联；其他机柜 → 跨柜；三腿同时发货取最慢
串行 T = C + N；理想融合 T = max(C, N)；上限 = (C+N)/max(C,N) < 2.0
```

详细口径见页面内"模型口径说明"，或两个脚本的 docstring。

## 预设

- **roofline_probe 默认**：cube=410.4，BW 247.8/141.6/141.6，EP 8→1024
- **ep_probe 默认**：cube=410.4，BW 784/80/15，EP 8→320

## 🧠 显存分析页

分析 MegaMoe BF16 训练（普通 BF16 / MGroup / UDMA 三变体）每 rank 的 HBM 占用：
前向**要存哪些**（training_saved_state）、**不存哪些**（transient，返回即释放）、
反向**重算哪些 / 直接读哪些**（取决于 gate/up 保存模式）。

- **gate/up 保存模式**（对应运行时 `MEGAMOE_SAVE_GMM1_OUTPUT`，默认 bf16）：
  - **重算（none）**：Z 不落盘，反向 GMM1 重算 gate/up（16RFK 口径，backward ws 含 recomputeGate/Up）
  - **保存 BF16**：前向 epilogue 写入 saved 追加区 2·A512(2·R·F)，反向直接读（≈12RFK）
  - **保存 MXFP8**：FP8 + 每 32 元素 scale；swiglu_limit 有限时另存三位裁剪平面（layout v2）
  - auto 规则：无裁剪→MXFP8、有裁剪→BF16
- **配置**：M / K / N（=2F）/ topK / EP / EPR / R（maxRecvTokens，含均衡·M·T、
  M·EP·min(EPR,T)、最坏 M·EP·T 预设）/ HBM 容量；可从 Roofline 页一键同步 M/K/topK
- **KPI**：前向显存峰值 / 反向显存峰值（HBM 利用率）、
  **SHMEM 额度（= local_mem_size，Align2MiB(需要量)）**、
  前向保存→反向使用（saved 激活，含 packed X / gate/up / 元数据分解）；
  "存 vs 算"代价在决策表
- **SHMEM 额度规则**：额度建议 = `Align2MiB(布局需要量)`——local_mem_size 仅服务本算子，
  固定 shape 按需给即可；使用时 `shmem.init(额度)` 与算子构造（如 `MegaMoeBF16Train(shmem, 额度)`）
  传同一个值，不满足时 C++ 侧直接报错并给出需要量。常驻/峰值按该额度计。
  测试脚本 `test_mega_moe.py local_mem_bytes()` 另有固定底座口径：
  `Align2MiB(max(底座, 需要量))`，底座普通/MGroup 2 GiB、UDMA 3 GiB，用于一次 init 兜多种 case
- **字节账本明细**（折叠；saved + dispatch 输出 + gate/up 追加区 /
  transient / backward（模式相关）/ SHMEM / 权重 / IO，逐字段形状、对齐公式、字节数与反向消费方式），可导出 CSV
- **存算决策表**（折叠）：每块数据的生命周期（F→B 驻留 / F→B·可释放 / 仅 F / 仅 B / 全程）、
  前向处理、反向获取方式与理由，随保存模式联动
- **三张图**（折叠，优先级低）：阶段驻留构成堆叠（前向峰值 / fwd→bwd 驻留 / 反向峰值 / 常驻）、
  in-flight micro-batch × 驻留显存（vs HBM 参考线）、R 容量扫描
- **实测 MFU**（Roofline 页卡片）：输入实测耗时（ms/次）+ EP/F，
  achieved TFLOPS / MFU%（FLOPs = coef·R·F·K，与数据表同源）、计算地板 C、
  通信 N（瓶颈腿）、融合上限 max(C,N) 与实测倍数（含串行 C+N 对照）
- **内置锚点自检**：默认 shape（M4096/N8192/K7168/topK16/EP8/EPR4/R65536）下
  十个数与权威来源对照（normal saved 四档：530,432 / 1,074,272,256 / 554,178,560 / 654,841,856 B，
  MGroup BF16 saved 2,013,927,424、dispatch 输出 939,655,168、
  普通 transient 536,936,960、UDMA transient 536,879,616、
  普通/MGroup SHMEM 1,879,257,088、UDMA SHMEM 2,818,769,920 B），
  来源 test_workspace_layout.py + docs《MXFP8_saved_state第一版》《前向持久Workspace》，
  页面加载即验，徽章常显

字节公式来源：`kernels/mega_moe_bf16/include/mega_moe_training_workspace.h`、
`mega_moe_workspace_info.h`、`mega_moe_base.h`（PeermemInfo）、
`csrc/ops/mega_moe_bf16_mgroup.cpp`（MGroup ready 区）与
`test/megamoe_bf16_test/test_workspace_layout.py`。注意口径：topk_ids 按 int32 估算；
x/y 与优化器状态由框架持有，不计入算子 workspace；多个未反传 micro-batch 各自独立持有一份 saved。

## 回归校验

参考资产已入库（`ref/`），temp 被清理也不影响复跑：

```bash
# 静态守卫（CSS/结构/文案/锚点常量，44 项）
node test/bigop_fused/roofline_web/smoke.cjs
# 数值动态校验（网页模型 vs python/头文件公式）
node test/bigop_fused/roofline_web/verify_model.mjs \
  test/bigop_fused/roofline_web/ref/r.csv \
  test/bigop_fused/roofline_web/ref/e.csv \
  test/bigop_fused/roofline_web/ref/layout_ref.json
# 布局公式变更后重新生成参考值（r.csv/e.csv 同理用 probe 脚本重导）
python test/bigop_fused/roofline_web/export_layout_reference.py > test/bigop_fused/roofline_web/ref/layout_ref.json
```

当前基线：**666 项数值全部一致**（roofline 560 + 显存 6 shape×三变体×gate/up 四档及 dispatch 输出 96 + docs 锚点 10）。

> dispatch 输出口径（train PR 后）：普通前向 `embedDispatchOutput=false`——saved 不含
> packed X / routed 权重，二者以独立 `dispatch_output` 张量返回框架，由
> `retain_dispatch_output`（python autograd 层，默认 True）控制是否驻留：
> False 即释放，但重派发反向未集成，backward 会抛 RuntimeError；
> MGroup 仍嵌 saved（默认 true）；UDMA 的 packed X / routed 权重本就在 SHMEM 窗。

> SHMEM 需要量口径（train PR 后）：普通/MGroup 数据起点为 **72 KiB 同步头**
> （60KiB 旧头 + 4KiB rank 同步 + 8KiB AIV 生成计数），EP≥512 且 EPR≤8 时
> 另加紧凑计数收件箱 `Align512(64·P)+Align512(288)`；UDMA 仍为 60KiB 控制。
> 测试脚本底座 2/3 GiB 口径不变。

> 本目录位于独立 worktree `C:\MegaMoe\roofline_web`（分支 `roofline-web-tool`），
> 与其他 worktree 隔离，避免被 `git clean` / 分支清理误删。日常起服务：
> `python -m http.server 8791 --directory C:\MegaMoe\roofline_web\test\bigop_fused\roofline_web`
