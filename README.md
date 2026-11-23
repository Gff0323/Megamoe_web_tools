# MegaMoe 计算器（网页版）

三个纯理论工具的**实时交互版**，改配置即时出数，不再需要"跑一遍 Python → 生成 PNG/CSV → 打开看"：

- **🧠 显存分析页**（默认页）：MegaMoe BF16 训练前向/反向的显存（HBM）账本，
  字节公式与 kernel 布局逐项一致，基于 docs《MegaMoe_BF16_前向持久Workspace第一版与反向适配》
- **📐 实测 MFU 页**：输入 bench 耗时，得到 achieved TFLOPS / MFU%，含计算地板与 baseline 加速比
- **⏱ 理论 Roofline 页**：`roofline_probe.py` / `ep_probe.py` 的移植

纯理论模型（均衡路由 + 理想重叠假设），**不依赖 NPU**，全部计算在浏览器本地完成。

线上地址：<https://gff0323.github.io/Megamoe_web_tools/>（私有仓，需 GitHub 权限）
本仓即维护源；MegaMoe 主仓的 `test/bigop_fused/roofline_web/` 是历史副本，不再编辑。

## 打开方式

任选其一：

```bash
# 方式 A：直接双击 index.html（file:// 即可用，零依赖）

# 方式 B：起个静态服务（推荐，方便多标签页/分享）
python -m http.server 8791 --directory .
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
- **🔀 流水页**（第三个 Tab）：
  - **⓪ 流水示意**：前向/反向切换、4/8 波展开的示意甘特（三车道 = AIC/AIV0/AIV1），
    块宽与阶段均值**取自真实 trace**（`SCHED_MEAS`，见下），只把 33 波压成 4/8 波展示。
    结论：前向 GMM1 领先约一波、GMM2 落后、Dispatch 提前数波连发；反向逐波只有
    dO→dHp→SwiGLUGrad，**大头是尾部一次性 dW1≈1.43ms / dW2≈0.73ms / dX≈1.40ms**；
    两端固定开销（INIT_ROUTING≈92µs、UNPERMUTE≈183/214µs）不可忽略
  ① 前向 Wave 五阶段（Dispatch → GMM1 → SwiGLU → GMM2 → Combine）+ 模式 0/1 对比；
  ② 反向 128 行块就绪链 + EARLY dW2 抢算 + `dP→Hp`/`dHp+dW2→dX` 复用规则 + bf16/mxfp8/none 三档；
  ③ 事件命名与 Chrome trace 对照（`wave_N_AIC_BW_DHP_E2`、`AIV_BW_DP_REDUCE_SEND` 等）；
  ④ **甘特图**：直接导入 `parse_backward_timer.py` 输出的 CSV（列
  `core_id,group_id,core_type,sub_id,<stage>_start_cycle,_end_cycle,_duration(us)…`），
  每 lane 一行、等待阶段灰色，附"载入示例"按钮
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

本仓自包含：`ref/` 存着从 `layout_spec.py` 导出的参考快照，校验不需要 MegaMoe 源码。

```bash
# 静态 + 运行时渲染守卫（78 项）
node smoke.cjs
# 数值动态校验（网页模型 vs layout_spec 镜像，667 项）
node verify_model.mjs ref/r.csv ref/e.csv ref/layout_ref.json
```

公式对齐：`layout_spec.py` 复刻 **DeepEP `main`@e18a161**（2026-11-23）的
`mega_moe_bf16_layout.hpp` / `mega_moe_workspace_backward.hpp`。
重导参考值：

```powershell
cmd /c "python export_layout_reference.py > ref\layout_ref.json"   # 必须走 cmd /c，PowerShell 的 > 写 UTF-16
cmd /c "cd probes && python roofline_probe.py --cube-tflops 410.4 --bw-hccs 247.8 --bw-rack 141.6 --bw-cage 141.6 --r-node 8 --r-rack 8 --eps 8,16,32,64,128,256,512,1024 --csv ..\ref\r.csv --out %TEMP%\r.png"
cmd /c "cd probes && python ep_probe.py --bw-hccs 784 --bw-rack 80 --bw-cage 15 --r-node 8 --r-rack 8 --eps 8,16,32,64,128,256,320 --csv ..\ref\e.csv --out %TEMP%\e.png"
```

**已对齐新仓**：saved 四档（−1,024 B：`bwProgress` 移入反向进度区）、dispatch 输出、
forward transient（−65,792 B：ready 区移出）、backward workspace（`dp` 走整行 topk grad、
新增 4 个 readiness + progress + stagedReady）。
> 流水实测基准：`all_rank_trace.json`（rank0，M=2048 H=5120 F=2304 topk=6 E/rank=4 EP8，
> 33 waves × 32 groups，前向 78,592 / 反向 41,984 事件）。

**SHMEM 已对齐（BF16 训练、MTE 路径）**：`plan_bf16.cpp:162-164` 的
`CompactTokenCountExchangeLayout` + `PeerPayloadLayout`（routeIndex / weight / return / topkGrad），
默认 shape 由旧口径 1.79 GiB 降到 **0.93 GiB**（1,879,257,088 → 998,846,464 B）。
UDMA 后端仅在 `codex/deepep-udma-backend` 分支，`main` 已改走 URMA peermem，未建模。

当前基线：**666 项数值全部一致**。

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




