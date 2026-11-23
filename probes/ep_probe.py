#!/usr/bin/env python3
"""MoE EP 通信时间/speedup 推算器（四级网络拓扑版）。

估算链条（均衡路由 + 均匀 expert placement 假设）：
  ① 每 rank 每 dispatch 方向流量 = M×topK×K×2B（BF16；远程占比按命中概率折减）
  ② 四级命中概率（专家归属决定，R_node=每节点卡数，R_rack=每机柜节点数）：
       本卡          = 1/EP                        → 走 HBM，不计网络
       节点内他卡    = (R_node−1)/EP                → 走 HCCS fullmesh
       柜内他节点    = R_node×(R_rack−1)/EP         → 走柜内互联
       其他机柜      = 1 − R_node×R_rack/EP         → 走跨柜网络
  ③ 三条网络腿独立硬件、同时发货 → T_每向 = max(三腿各自 流量÷带宽)
  ④ 每层 4 趟（fwd/bwd × dispatch/combine）→ T_通信/层 = 4 × T_每向
  ⑤ speedup 上限 = 小算子串行(C+N) / 融合max(C,N)   ← 与实现无关的物理上界

用途：给三个实测/规格带宽（HCCS/柜内/跨柜），定量看不同 EP/机器规模下
通信时间、通信:计算比、融合算子 speedup 上限的演变，找甜点区。
dual-node 实测后用 bigop 的 alltoall 分解行反推有效带宽回填校准。

用法：
  python ep_probe.py                                   # 默认 shape + 默认带宽
  python ep_probe.py --m 4096 --k 7168 --topk 8 --f 3072,8192
  python ep_probe.py --bw-hccs 784 --bw-rack 80 --bw-cage 15
  python ep_probe.py --out ep_probe.png               # 出图（无 matplotlib 降级 ASCII）
  python ep_probe.py --csv ep_probe.csv               # 全表落盘
输出列：
  ep, nodes, racks, T_each_way_ms, T_layer_ms(×4),
  comm_ratio_%(对 bwd floor), speedup_ceil（含 F=3072/8192 两档）
"""
from __future__ import annotations

import argparse
import csv
import sys

# ---- 常量 ----
ROOF_TFLOPS = 410.4   # 432 峰值 ×0.95（docs/backward_perf_analysis.md，单机 8 卡口径）
BW_HCCS = 784.0       # GB/s，节点内 HCCS fullmesh 每卡单向聚合（7×112）
BW_RACK = 80.0        # GB/s，柜内节点间互联每 rank 有效（待实测校准）
BW_CAGE = 15.0        # GB/s，跨柜网络每 rank 有效（200G NIC 扣协议/拥塞，待校准）
R_NODE = 8            # 每节点卡数
R_RACK = 8            # 每机柜节点数（单柜 64 卡）

# 扫描的 EP 序列：1 节点 → 1 柜 → 多柜（对数阶梯）
EP_POINTS = [8, 16, 32, 64, 128, 256, 320]
F_LIST = [3072, 8192]  # 两档 FFN（计算 floor 差 2.67x，甜点区不同）


def probe(m, k, topk, ep, r_node, r_rack, bw_hccs, bw_rack, bw_cage):
    """单 EP 点的推算。返回 dict（时间单位 ms，流量 MB）。"""
    total_mb = m * topk * k * 2 / 1e6                       # 每 rank 每向毛流量
    # 部署几何：本 EP 实际有多少节点/机柜（EP 小于单柜时截断）
    nodes = ep // r_node
    racks = max(1, nodes // r_rack)
    cards_in_my_rack = min(ep, r_node * r_rack)   # 我所在机柜实际卡数（EP<64 时 < 64）
    cards_in_my_node = r_node
    # 四级命中概率（按"容器内专家数/全网专家数"；容器尺寸被部署规模截断）
    frac_card = 1.0 / ep                         # 本卡（我拥有 EPR 个专家）
    frac_node = (cards_in_my_node - 1) / ep      # 本节点他卡（(R_node−1)×EPR/E）
    frac_rack = (cards_in_my_rack - cards_in_my_node) / ep  # 柜内他节点
    frac_cage = 1.0 - cards_in_my_rack / ep      # 其他机柜
    # 概率归一校验（浮点容差）
    assert abs(frac_card + frac_node + frac_rack + frac_cage - 1.0) < 1e-9
    mb_node = total_mb * frac_node
    mb_rack = total_mb * frac_rack
    mb_cage = total_mb * frac_cage
    # 三条腿同时发货，取慢者。单位换算：MB ÷ (GB/s) = MB ÷ (1000 MB/s) = 1e-3 s = ms
    t_node = mb_node / bw_hccs if mb_node > 0 else 0.0
    t_rack = mb_rack / bw_rack if mb_rack > 0 else 0.0
    t_cage = mb_cage / bw_cage if mb_cage > 0 else 0.0
    t_way = max(t_node, t_rack, t_cage)
    bottleneck = ("HCCS" if t_way == t_node else
                  "RACK" if t_way == t_rack else "CAGE")
    return {
        "ep": ep, "nodes": nodes, "racks": racks,
        "total_mb": total_mb,
        "mb_node": mb_node, "mb_rack": mb_rack, "mb_cage": mb_cage,
        "t_node": t_node, "t_rack": t_rack, "t_cage": t_cage,
        "t_way": t_way, "bottleneck": bottleneck,
        "t_layer": t_way * 4,
    }


def speedup_ceil(t_comm_layer, t_compute_ms):
    """融合 speedup 物理上限：串行(C+N) / max(C,N)。"""
    return (t_compute_ms + t_comm_layer) / max(t_compute_ms, t_comm_layer)


def probe_all(m, k, topk, f_list, ep_points, r_node, r_rack,
              bw_hccs, bw_rack, bw_cage):
    rows = []
    for ep in ep_points:
        if ep % r_node or ep < r_node:
            continue  # EP 必须是节点卡数的整数倍
        row = probe(m, k, topk, ep, r_node, r_rack, bw_hccs, bw_rack, bw_cage)
        for f in f_list:
            # bwd-only: 16 RFK（重计算口径）
            t_bwd = 16.0 * m * topk * f * k / (ROOF_TFLOPS * 1e12) * 1e3
            # full layer: fwd 6 RFK (GMM1 4 + GMM2 2) + bwd 16 RFK = 22 RFK
            t_layer_compute = 22.0 * m * topk * f * k / (ROOF_TFLOPS * 1e12) * 1e3
            t_comm = row["t_layer"]  # 4 趟通信（fwd/bwd 各 dispatch+combine）

            row_f = dict(row)
            row_f["f"] = f
            row_f["t_bwd_compute"] = t_bwd
            row_f["t_layer_compute"] = t_layer_compute

            # bwd-only 视角
            row_f["t_compute_ms"] = t_bwd  # 兼容旧列名
            row_f["comm_ratio_pct"] = 100.0 * t_comm / t_bwd
            row_f["speedup_ceil"] = (t_bwd + t_comm) / max(t_bwd, t_comm)
            row_f["bwd_ideal_ms"] = max(t_bwd, t_comm)
            row_f["bwd_smallop_ms"] = t_bwd + t_comm

            # full layer 视角（fwd+bwd 的 compute vs 同一份 comm）
            row_f["layer_ideal_ms"] = max(t_layer_compute, t_comm)
            row_f["layer_smallop_ms"] = t_layer_compute + t_comm
            row_f["layer_speedup_ceil"] = ((t_layer_compute + t_comm)
                                           / max(t_layer_compute, t_comm))
            row_f["layer_comm_ratio_pct"] = 100.0 * t_comm / t_layer_compute
            rows.append(row_f)
    return rows


def print_ascii(rows, m, k, topk, bw_hccs, bw_rack, bw_cage):
    print(f"[shape] M={m} K={k} topK={topk} | 每向流量 {m*topk*k*2/1e6:.0f}MB/rank"
          f" | BW: HCCS={bw_hccs:.0f} 柜内={bw_rack:.0f} 跨柜={bw_cage:.0f} GB/s")
    for f in ({r["f"] for r in rows}):
        sub = [r for r in rows if r["f"] == f]
        print(f"\n===== F={f} =====")
        print(f"  bwd-only: 计算地板 {sub[0]['t_bwd_compute']:.1f} ms | "
              f"full-layer: 计算地板 {sub[0]['t_layer_compute']:.1f} ms (22RFK)")
        print(f"  {'EP':>4}{'节点':>5}{'机柜':>5}{'瓶颈':>6}"
              f"{'T/向ms':>8}{'T/层ms':>8}"
              f"{'|bwd 串行':>9}{'bwd 理想':>9}{'bwd上限':>8}"
              f"{'|layer串行':>10}{'layer理想':>10}{'layer上限':>9}")
        for r in sub:
            print(f"{r['ep']:>4}{r['nodes']:>5}{r['racks']:>5}{r['bottleneck']:>6}"
                  f"{r['t_way']:>8.2f}{r['t_layer']:>8.1f}"
                  f"{r['bwd_smallop_ms']:>10.1f}{r['bwd_ideal_ms']:>10.1f}"
                  f"{r['speedup_ceil']:>7.2f}x"
                  f"{r['layer_smallop_ms']:>11.1f}{r['layer_ideal_ms']:>11.1f}"
                  f"{r['layer_speedup_ceil']:>8.2f}x")


def write_csv(rows, path):
    cols = ["ep", "nodes", "racks", "f", "bottleneck",
            "mb_node", "mb_rack", "mb_cage",
            "t_node", "t_rack", "t_cage", "t_way", "t_layer",
            "t_bwd_compute", "t_layer_compute",
            "bwd_smallop_ms", "bwd_ideal_ms", "speedup_ceil",
            "layer_smallop_ms", "layer_ideal_ms", "layer_speedup_ceil",
            "comm_ratio_pct", "layer_comm_ratio_pct"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[out] csv: {path} ({len(rows)} rows)")


def plot_png(rows, path, m, k, topk, bw_hccs, bw_rack, bw_cage):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fs = sorted({r["f"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"EP scaling: M={m} K={k} topK={topk} | "
                 f"BW HCCS={bw_hccs:.0f} rack={bw_rack:.0f} cage={bw_cage:.0f} GB/s",
                 fontsize=13)
    colors = {fs[0]: "tab:blue", fs[1]: "tab:orange"} if len(fs) > 1 else {fs[0]: "tab:blue"}

    def eps_of(f):
        return [r["ep"] for r in rows if r["f"] == f]

    def vals_of(f, key):
        return [r[key] for r in rows if r["f"] == f]

    # 1 各腿时间 vs EP（对数）
    ax = axes[0][0]
    f0 = fs[0]
    for key, lab, c in (("t_node", "node(HCCS)", "tab:green"),
                        ("t_rack", "in-rack", "tab:orange"),
                        ("t_cage", "cross-rack", "tab:red")):
        ax.plot(eps_of(f0), vals_of(f0, key), "o-", label=lab, color=c)
    ax.plot(eps_of(f0), vals_of(f0, "t_way"), "k^-", label="T/way (max)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("EP"); ax.set_ylabel("ms / way")
    ax.set_title("per-leg time vs EP"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 2 每层通信 vs 计算 floor
    ax = axes[0][1]
    for f in fs:
        ax.plot(eps_of(f), vals_of(f, "t_layer"), "o-", color=colors[f],
                label=f"comm (x4 ways), F={f}")
        ax.plot(eps_of(f), vals_of(f, "t_compute_ms"), "--", color=colors[f],
                label=f"compute floor, F={f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EP"); ax.set_ylabel("ms / layer")
    ax.set_title("comm (solid) vs compute floor (dashed)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 3 通信:计算比
    ax = axes[0][2]
    for f in fs:
        ax.plot(eps_of(f), vals_of(f, "comm_ratio_pct"), "o-", color=colors[f],
                label=f"F={f}")
    ax.axhline(100, color="red", ls="--", lw=1)
    ax.text(eps_of(f0)[-1], 105, "comm = compute (sweet spot)", fontsize=8, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EP"); ax.set_ylabel("%")
    ax.set_title("comm : compute"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 4 speedup 上限（两个视角）
    ax = axes[1][0]
    for f in fs:
        ax.plot(eps_of(f), vals_of(f, "speedup_ceil"), "o-", color=colors[f],
                label=f"bwd only, F={f}")
        ax.plot(eps_of(f), vals_of(f, "layer_speedup_ceil"), "s--", color=colors[f],
                label=f"full layer, F={f}", alpha=0.7)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axhline(2.0, color="red", ls="--", lw=1)
    ax.text(eps_of(f0)[0], 2.03, "theoretical max 2.0x", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EP"); ax.set_ylabel("speedup ceiling")
    ax.set_title("fused speedup ceiling (C+N)/max(C,N)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # 5 各级流量分布（堆叠）
    ax = axes[1][1]
    ax.stackplot(eps_of(f0),
                 vals_of(f0, "mb_node"), vals_of(f0, "mb_rack"), vals_of(f0, "mb_cage"),
                 labels=["node(HCCS)", "in-rack", "cross-rack"],
                 colors=["tab:green", "tab:orange", "tab:red"], alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EP"); ax.set_ylabel("MB / way / rank")
    ax.set_title("traffic split by network tier"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # 6 绝对时间：串行 vs 理想重叠（full layer）
    ax = axes[1][2]
    for f in fs:
        ax.plot(eps_of(f), vals_of(f, "layer_smallop_ms"), "o--", color=colors[f],
                label=f"smallop (serial), F={f}", alpha=0.6)
        ax.plot(eps_of(f), vals_of(f, "layer_ideal_ms"), "s-", color=colors[f],
                label=f"fused ideal, F={f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("EP"); ax.set_ylabel("ms / layer (fwd+bwd)")
    ax.set_title("absolute time: serial vs ideal overlap"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--k", type=int, default=7168)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--f", default="3072,8192", help="FFN 档位，逗号分隔")
    p.add_argument("--bw-hccs", type=float, default=BW_HCCS)
    p.add_argument("--bw-rack", type=float, default=BW_RACK)
    p.add_argument("--bw-cage", type=float, default=BW_CAGE)
    p.add_argument("--r-node", type=int, default=R_NODE)
    p.add_argument("--r-rack", type=int, default=R_RACK)
    p.add_argument("--eps", default=",".join(map(str, EP_POINTS)),
                   help="逗号分隔的 EP 扫描点")
    p.add_argument("--out", help="图输出路径（默认 ep_probe.png）")
    p.add_argument("--csv", help="CSV 输出路径（默认 ep_probe.csv）")
    args = p.parse_args()

    f_list = [int(s) for s in args.f.split(",") if s.strip()]
    ep_points = [int(s) for s in args.eps.split(",") if s.strip()]
    rows = probe_all(args.m, args.k, args.topk, f_list, ep_points,
                     args.r_node, args.r_rack,
                     args.bw_hccs, args.bw_rack, args.bw_cage)
    if not rows:
        sys.exit("[error] no valid EP points (must be multiples of --r-node)")
    print_ascii(rows, args.m, args.k, args.topk,
                args.bw_hccs, args.bw_rack, args.bw_cage)
    csv_path = args.csv or "ep_probe.csv"
    write_csv(rows, csv_path)
    png_path = args.out or "ep_probe.png"
    if plot_png(rows, png_path, args.m, args.k, args.topk,
                args.bw_hccs, args.bw_rack, args.bw_cage):
        print(f"[out] png: {png_path}")
    else:
        print("[warn] matplotlib 不可用：仅 ASCII + CSV（装 matplotlib 后重跑出图）")


if __name__ == "__main__":
    main()
