#!/usr/bin/env python3
"""MoE theoretical limit analysis: GEMM compute speed vs communication race.

Answers three questions (pure theory, independent of kernel implementation):
  1. How fast can we go if all communication is hidden under compute?
  2. At what scale does compute stop being able to hide communication?
  3. How much of the communication can be hidden at each scale?

Model (balanced routing + ideal overlap):
  C = 22*R*F*K / cube_TFLOPS            <- full-layer GEMM time (fwd 6 + bwd 16 RFK)
  N = 4 * max(leg_traffic/bandwidth)    <- full-layer comm time (4 ways)
  small-op (serial):  T = C + N
  fused (ideal):      T = max(C, N)     <- bottleneck resource dominates
  speedup = (C+N) / max(C,N)            <- physical ceiling < 2.0x
  overlap  = min(C, N) / max(C, N)      <- fraction of the slower one hidden

Usage:
  python roofline_probe.py                              # defaults
  python roofline_probe.py --cube-tflops 350            # if GEMM only reaches 350
  python roofline_probe.py --bw-rack 60 --bw-cage 12    # calibrated bandwidths
  python roofline_probe.py --f 3072,8192 --out roof.png
"""
from __future__ import annotations

import argparse
import csv
import sys

R_NODE = 8
R_RACK = 8
EP_POINTS = [8, 16, 32, 64, 128, 256, 512, 1024]
F_LIST = [3072, 4096, 8192]

# Default bandwidths (GB/s, to be calibrated). All are UNIDIRECTIONAL
# effective bandwidths per rank: MB / (GB/s) = ms directly.
BW_HCCS = 247.8   # in-node HCCS fullmesh per-rank unidirectional aggregate
BW_RACK = 141.6   # in-rack inter-node per rank (to calibrate)
BW_CAGE = 141.6   # cross-rack per rank (to calibrate)
CUBE_TFLOPS = 410.4  # 432 x 0.95


def comm_time_per_way(m, k, topk, ep, r_node, r_rack, bw_hccs, bw_rack, bw_cage):
    """Per-way communication time (ms) under the 4-tier topology."""
    total_mb = m * topk * k * 2 / 1e6
    cards_in_my_rack = min(ep, r_node * r_rack)
    frac_node = (r_node - 1) / ep
    frac_rack = (cards_in_my_rack - r_node) / ep if ep > r_node else 0.0
    frac_cage = 1.0 - cards_in_my_rack / ep if ep > r_node * r_rack else 0.0
    mb_node = total_mb * frac_node
    mb_rack = total_mb * frac_rack
    mb_cage = total_mb * frac_cage
    t_node = mb_node / bw_hccs if mb_node > 0 else 0.0
    t_rack = mb_rack / bw_rack if mb_rack > 0 else 0.0
    t_cage = mb_cage / bw_cage if mb_cage > 0 else 0.0
    t_way = max(t_node, t_rack, t_cage)
    bn = "HCCS" if t_way == t_node else "RACK" if t_way == t_rack else "CAGE"
    return t_way, bn, {"node": t_node, "rack": t_rack, "cage": t_cage}


def analyze(m, k, topk, f, ep, cube_tflops, r_node, r_rack,
            bw_hccs, bw_rack, bw_cage):
    """Full analysis at one (shape, EP) point."""
    r = m * topk
    flops_layer = 22.0 * r * f * k          # fwd 6 + bwd 16 RFK
    t_compute = flops_layer / (cube_tflops * 1e12) * 1e3  # ms
    t_way, bottleneck, legs = comm_time_per_way(
        m, k, topk, ep, r_node, r_rack, bw_hccs, bw_rack, bw_cage)
    t_comm = t_way * 4                       # 4 ways per layer

    t_smallop = t_compute + t_comm           # serial
    t_ideal = max(t_compute, t_comm)         # ideal overlap
    speedup = t_smallop / t_ideal
    overlap_pct = (min(t_compute, t_comm) / max(t_compute, t_comm)
                   if max(t_compute, t_comm) > 0 else 0.0)
    regime = "compute-bound" if t_compute > t_comm else "comm-bound"
    return {
        "ep": ep, "f": f, "r": r,
        "nodes": ep // r_node, "racks": max(1, ep // (r_node * r_rack)),
        "bottleneck": bottleneck,
        "t_compute": t_compute, "t_comm": t_comm,
        "t_way": t_way, "t_smallop": t_smallop, "t_ideal": t_ideal,
        "speedup_ceil": speedup,
        "overlap_pct": overlap_pct * 100.0,
        "comm_compute_ratio": t_comm / t_compute if t_compute > 0 else float("inf"),
        "regime": regime,
        "hidden_ms": min(t_compute, t_comm),
        "exposed_ms": max(t_compute, t_comm) - min(t_compute, t_comm),
    }


def analyze_all(m, k, topk, f_list, ep_points, cube_tflops,
                r_node, r_rack, bw_hccs, bw_rack, bw_cage):
    rows = []
    for ep in ep_points:
        if ep % r_node or ep < r_node:
            continue
        for f in f_list:
            rows.append(analyze(m, k, topk, f, ep, cube_tflops,
                                r_node, r_rack, bw_hccs, bw_rack, bw_cage))
    return rows


def print_ascii(rows, m, k, topk, cube_tflops, bw_hccs, bw_rack, bw_cage):
    print(f"[config] M={m} K={k} topK={topk} cube={cube_tflops:.0f} TFLOPS | "
          f"BW(unidirectional): HCCS={bw_hccs:.0f} rack={bw_rack:.0f} "
          f"cage={bw_cage:.0f} GB/s | scope=full layer (22RFK + 4 ways)")
    for f in sorted({r["f"] for r in rows}):
        sub = [r for r in rows if r["f"] == f]
        c = sub[0]["t_compute"]
        print(f"\n===== F={f} | compute floor={c:.1f}ms/layer "
              f"(22RFK @ {cube_tflops:.0f} TFLOPS) =====")
        print(f"  {'EP':>5}{'nodes':>6}{'racks':>6}{'bneck':>6}"
              f"{'compute':>9}{'comm':>8}{'N:C':>7}"
              f"{'|smallop':>9}{'ideal':>8}{'ceil':>7}"
              f"{'|overlap':>8}{'exposed':>8}  regime")
        for r in sub:
            print(f"{r['ep']:>5}{r['nodes']:>6}{r['racks']:>6}{r['bottleneck']:>6}"
                  f"{r['t_compute']:>9.1f}{r['t_comm']:>8.1f}"
                  f"{r['comm_compute_ratio']:>6.2f}x"
                  f"{r['t_smallop']:>10.1f}{r['t_ideal']:>8.1f}"
                  f"{r['speedup_ceil']:>6.2f}x"
                  f"{r['overlap_pct']:>7.0f}%{r['exposed_ms']:>8.1f}"
                  f"  {r['regime']}")


def write_csv(rows, path):
    cols = ["ep", "f", "nodes", "racks", "bottleneck", "regime",
            "t_compute", "t_comm", "t_way", "comm_compute_ratio",
            "t_smallop", "t_ideal", "speedup_ceil",
            "overlap_pct", "hidden_ms", "exposed_ms"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[out] csv: {path} ({len(rows)} rows)")


def plot_png(rows, path, m, k, topk, cube_tflops, bw_hccs, bw_rack, bw_cage):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fs = sorted({r["f"] for r in rows})
    f_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray"]
    colors = {f: f_colors[i] for i, f in enumerate(fs)}
    ep_values = sorted({r["ep"] for r in rows})
    # Equal-spaced x-axis: each EP point gets the same slot width.
    ep_x = {ep: i for i, ep in enumerate(ep_values)}
    f0 = fs[0]

    def xs(f):
        return [ep_x[r["ep"]] for r in rows if r["f"] == f]

    def vals(f, key):
        return [r[key] for r in rows if r["f"] == f]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Roofline analysis: M={m} K={k} topK={topk} | "
                 f"cube={cube_tflops:.0f} TFLOPS | "
                 f"BW(unidir): HCCS={bw_hccs:.0f} rack={bw_rack:.0f} "
                 f"cage={bw_cage:.0f} GB/s", fontsize=12)

    def style_x(ax):
        ax.set_xticks(list(ep_x.values()))
        ax.set_xticklabels([str(e) for e in ep_values])
        ax.set_xlabel("EP (card count)")
        ax.grid(alpha=0.3)

    def annotate_all(ax, f, key, fmt="{:.2f}x", dy=8):
        """Label every point on a line (for the speedup panel)."""
        for r in rows:
            if r["f"] != f:
                continue
            ax.annotate(fmt.format(r[key]),
                        (ep_x[r["ep"]], r[key]),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6, color=colors[f], alpha=0.8)

    def annotate_ends(ax, f, key, fmt="{:.1f}", dy=8):
        """Label only the first and last points (to avoid clutter)."""
        sub = [r for r in rows if r["f"] == f]
        if not sub:
            return
        for r in (sub[0], sub[-1]):
            ax.annotate(fmt.format(r[key]),
                        (ep_x[r["ep"]], r[key]),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=6.5, color=colors[f], alpha=0.8)

    # 1 compute vs comm race (the core chart)
    # Note: comm time is independent of F (traffic = M*topK*K*2B, no F),
    # so we plot ONE comm line instead of one per F (they would overlap).
    ax = axes[0][0]
    for f in fs:
        ax.plot(xs(f), vals(f, "t_compute"), "--o", color=colors[f],
                label=f"compute (22RFK), F={f}")
        annotate_ends(ax, f, "t_compute", fmt="{:.1f}", dy=8)
    ax.plot(list(ep_x.values()),
            [r["t_comm"] for r in rows if r["f"] == fs[0]],
            "-ks", color="black", markersize=5,
            label="comm (4 ways, same for all F)")
    for r in rows:
        if r["f"] != fs[0]:
            continue
        if r["ep"] in (ep_values[0], ep_values[-1]):
            ax.annotate(f"{r['t_comm']:.1f}",
                        (ep_x[r["ep"]], r["t_comm"]),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=6.5, color="black", alpha=0.8)
    ax.set_ylabel("ms / layer")
    ax.set_title("compute (dashed) vs comm (solid): crossover = regime switch")
    ax.legend(fontsize=8)
    style_x(ax)

    # 2 speedup ceiling (annotate all points - this is the key result)
    ax = axes[0][1]
    for f in fs:
        ax.plot(xs(f), vals(f, "speedup_ceil"), "o-", color=colors[f], label=f"F={f}")
        annotate_all(ax, f, "speedup_ceil", fmt="{:.2f}x", dy=7)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axhline(2.0, color="red", ls="--", lw=1.5)
    ax.text(0, 2.03, "theoretical max 2.0x", fontsize=9, color="red")
    ax.fill_between(list(ep_x.values()), 1.0, 2.0, alpha=0.05, color="green")
    ax.set_ylabel("speedup ceiling")
    ax.set_title("fused speedup ceiling = (C+N)/max(C,N)")
    ax.legend(fontsize=8)
    style_x(ax)

    # 3 N:C ratio (regime map, annotate endpoints)
    ax = axes[1][0]
    for f in fs:
        ax.plot(xs(f), vals(f, "comm_compute_ratio"), "o-", color=colors[f], label=f"F={f}")
        annotate_ends(ax, f, "comm_compute_ratio", fmt="{:.2f}x", dy=8)
    ax.axhline(1.0, color="red", ls="--", lw=1.5)
    ax.text(0, 1.05, "N = C (sweet spot, speedup = 2x)", fontsize=8, color="red")
    ax.axhspan(0, 1.0, alpha=0.05, color="blue", label="compute-bound")
    top = max(5, max(vals(f0, "comm_compute_ratio")))
    ax.axhspan(1.0, top, alpha=0.05, color="red", label="comm-bound")
    ax.set_ylabel("comm / compute")
    ax.set_title("regime map (below red line = compute-bound)")
    ax.legend(fontsize=8)
    style_x(ax)

    # 4 absolute time: serial vs ideal (annotate ideal line endpoints)
    ax = axes[1][1]
    for f in fs:
        ax.plot(xs(f), vals(f, "t_smallop"), "o--", color=colors[f],
                label=f"smallop (serial), F={f}", alpha=0.6)
        ax.plot(xs(f), vals(f, "t_ideal"), "s-", color=colors[f],
                label=f"fused ideal, F={f}")
        annotate_ends(ax, f, "t_ideal", fmt="{:.1f}", dy=8)
        annotate_ends(ax, f, "t_smallop", fmt="{:.1f}", dy=-12)
    ax.set_ylabel("ms / layer")
    ax.set_title("absolute time: serial (dashed) vs ideal overlap (solid)")
    ax.legend(fontsize=8)
    style_x(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--k", type=int, default=7168)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--f", default="3072,4096,8192",
                   help="FFN sizes to sweep (one line per F in the charts)")
    p.add_argument("--cube-tflops", type=float, default=CUBE_TFLOPS,
                   help="GEMM throughput (TFLOPS); calibrate from the bench roofline")
    p.add_argument("--bw-hccs", type=float, default=BW_HCCS,
                   help="unidirectional in-node HCCS per-rank aggregate, unit GB/s")
    p.add_argument("--bw-rack", type=float, default=BW_RACK,
                   help="unidirectional in-rack inter-node per rank, unit GB/s")
    p.add_argument("--bw-cage", type=float, default=BW_CAGE,
                   help="unidirectional cross-rack per rank, unit GB/s")
    p.add_argument("--r-node", type=int, default=R_NODE)
    p.add_argument("--r-rack", type=int, default=R_RACK)
    p.add_argument("--eps", default=",".join(map(str, EP_POINTS)))
    p.add_argument("--out", default="roofline_probe.png")
    p.add_argument("--csv", default="roofline_probe.csv")
    args = p.parse_args()

    f_list = [int(s) for s in args.f.split(",") if s.strip()]
    ep_points = [int(s) for s in args.eps.split(",") if s.strip()]
    rows = analyze_all(args.m, args.k, args.topk, f_list, ep_points,
                       args.cube_tflops, args.r_node, args.r_rack,
                       args.bw_hccs, args.bw_rack, args.bw_cage)
    if not rows:
        sys.exit("[error] no valid EP points")
    print_ascii(rows, args.m, args.k, args.topk,
                args.cube_tflops, args.bw_hccs, args.bw_rack, args.bw_cage)
    write_csv(rows, args.csv)
    if plot_png(rows, args.out, args.m, args.k, args.topk,
                args.cube_tflops, args.bw_hccs, args.bw_rack, args.bw_cage):
        print(f"[out] png: {args.out}")
    else:
        print("[warn] matplotlib unavailable: ASCII + CSV only")


if __name__ == "__main__":
    main()
