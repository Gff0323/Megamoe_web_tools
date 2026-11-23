#!/usr/bin/env python3
"""MegaMoe BF16 训练 workspace 布局的 python 规格（对齐 DeepEP xx/main）。

新仓（https://github.com/xixing-321/DeepEP，main）**没有** python 布局规格文件，
本文件是把下面这些 C++ 头文件逐项复刻成 python，作为网页模型的对账基准：

  experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_bf16_layout.hpp
  experiments/megamoe/csrc/kernels/mega_moe/common/mega_moe_workspace_backward.hpp

对齐的提交：xx/main @ e18a161（2026-11-23）。

已复刻：
  * DispatchOutputLayout
  * TrainingSavedStateLayout（含 weightBytes 可选 FP32 TopK 权重）
  * BackwardWorkspaceLayout + BackwardDispatchProgressLayout
  * ForwardTransientWorkspaceLayout

未复刻（下一轮）：
  * SHMEM / peermem（新仓为 mega_moe_peermem.hpp，MTE/URMA 两套布局，比训练旧版复杂）
  * UDMA 变体（新仓 main 上已无 UDMAForwardTransientWorkspaceLayout）

用法：
  python layout_spec.py                 # 打印默认 shape 的各项字节
  python layout_spec.py --json          # 输出 JSON（供 verify_model.mjs）
"""

import argparse
import json

# ---------- 常量（mega_moe_bf16_layout.hpp） ----------
BF16_BYTES = 2
INT32_BYTES = 4
INT_CACHELINE = 16
ALIGN_32, ALIGN_64 = 32, 64
ALIGN_128, ALIGN_256, ALIGN_512 = 128, 256, 512

# SavedGateUpMode（kSaveAuto=0, kSaveNone=1, kSaveBf16=2, kSaveMxFp8=3）
SAVE_AUTO, SAVE_NONE, SAVE_BF16, SAVE_MXFP8 = 0, 1, 2, 3

# A5（Ascend950）MIX 1:2 下最多 72 AIV / 36 AIC；plan_bf16.cpp: blockNumPerEP = max(1, aic/EP)
DEFAULT_AIC = 36


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


# ---------- 判定谓词（逐字复刻） ----------
def use_whole_row_topk_grad(mode, f):
    """UseWholeRowTopkGrad：BF16 保存模式且 F ≤ 16384、F%64==0 时走整行 topk grad。"""
    return mode == SAVE_BF16 and f > 0 and f <= 16384 and f % 64 == 0


def use_backward_staged_wave(mode, f):
    return use_whole_row_topk_grad(mode, f) or (mode == SAVE_MXFP8 and f > 0 and f % 32 == 0)


def backward_topk_partial_count(mode, f):
    """BackwardTopkPartialCount。"""
    return 1 if use_whole_row_topk_grad(mode, f) else (f + 255) // 256


def use_gm_for_backward_data_grad(mode):
    return mode != SAVE_NONE


def default_backward_dispatch_split_experts(mode, local_experts):
    return local_experts if (use_gm_for_backward_data_grad(mode) and local_experts <= 7) else 0


# ---------- 布局 ----------
def dispatch_output_size(k, max_recv_tokens, weight_bytes=BF16_BYTES):
    """DispatchOutputLayout：packed tokens + packed routed TopK 权重。"""
    tokens = align_up(max_recv_tokens * k * BF16_BYTES, ALIGN_512)
    weights = align_up(max_recv_tokens * weight_bytes, ALIGN_512)
    return tokens + weights


def training_saved_state_size(m, k, topk, ep, epr, max_recv_tokens, n,
                              saved_mode=SAVE_BF16, save_clip_masks=False,
                              embed_dispatch_output=True, weight_bytes=BF16_BYTES):
    """TrainingSavedStateLayout（新仓：无 bwProgress，bw 进度独立到反向 workspace）。"""
    save_gate_up = saved_mode != SAVE_NONE
    save_mxfp8 = saved_mode == SAVE_MXFP8

    ptr_a = align_up(max_recv_tokens * k * BF16_BYTES, ALIGN_512) if embed_dispatch_output else 0
    cumsum = ptr_a + align_up(ep * epr * INT32_BYTES, ALIGN_512)
    exp_row_idx = cumsum + align_up(m, ALIGN_256) * topk * INT32_BYTES
    exp_row_idx_gather = exp_row_idx + align_up(m, ALIGN_256) * topk * INT32_BYTES
    sum_before_rank = exp_row_idx_gather + align_up(ep * epr, ALIGN_128) * INT32_BYTES
    saved_token_counts = sum_before_rank + align_up(ep * align_up(ep * epr, ALIGN_128) * INT32_BYTES, ALIGN_512)
    dispatched_topk_weights = saved_token_counts
    dispatched_topk_weights_bytes = (align_up(max_recv_tokens * weight_bytes, ALIGN_512)
                                     if embed_dispatch_output else 0)
    legacy = dispatched_topk_weights + dispatched_topk_weights_bytes

    plane = align_up(max_recv_tokens * (n // 2) * (1 if save_mxfp8 else BF16_BYTES), ALIGN_512) if save_gate_up else 0
    gate = align_up(legacy, ALIGN_512)
    up = gate + plane
    gate_scale = up + plane
    scale = align_up(max_recv_tokens * ((n // 2 + 31) // 32), ALIGN_512) if (save_gate_up and save_mxfp8) else 0
    up_scale = gate_scale + scale
    gate_high = up_scale + scale
    clip_plane = (align_up(max_recv_tokens * ((n // 2 + 31) // 32) * INT32_BYTES, ALIGN_512)
                  if (save_gate_up and save_mxfp8 and save_clip_masks) else 0)
    up_low = gate_high + clip_plane
    up_high = up_low + clip_plane
    return (up_high + clip_plane) if save_gate_up else legacy


def forward_transient_size(n, epr, max_recv_tokens):
    """ForwardTransientWorkspaceLayout：A2 [R, F] + swiglu→gmm2 标志。"""
    a2 = align_up(max_recv_tokens * (n // 2) * BF16_BYTES, ALIGN_512)
    return a2 + align_up(epr * INT_CACHELINE * INT32_BYTES, ALIGN_128)


# ---------- SHMEM / peermem（BF16 训练、MTE 路径；plan_bf16.cpp:162-164） ----------
PEERMEM_DATA_OFFSET = 60 * 1024          # 旧 60KiB 头（CalcPeermemSize 口径）
K_PEER_SYNC_COUNT_OFFSET = 64 * 1024     # 64KiB 处放 rank 同步计数
K_PEER_DATA_OFFSET = 72 * 1024           # 数据区起点 = 72KiB（含 8KiB AIV 生成计数）
COMPACT_MIN_EP = 512                     # kCompactTokenCountAutoMinEp
COMPACT_MAX_LOCAL_EXPERTS = 8
COMPACT_MAX_AIV_CORES = 72


def compact_count_layout(world, epr, allow_compact=True):
    """CompactTokenCountExchangeLayout：EP≥512 且 1≤EPR≤8 时启用紧凑计数收件箱。"""
    enabled = allow_compact and (2 <= world <= 1024) and (1 <= epr <= COMPACT_MAX_LOCAL_EXPERTS)
    packet = align_up(world * 64, ALIGN_512) if enabled else 0
    block_sums = align_up(COMPACT_MAX_AIV_CORES * INT32_BYTES, ALIGN_512) if enabled else 0
    token_counts_offset = K_PEER_DATA_OFFSET + packet + block_sums
    stride = align_up(world * epr, ALIGN_128)
    token_counts_bytes = world * stride * INT32_BYTES
    return {
        "enabled": enabled,
        "packetBytes": packet,
        "blockSumsBytes": block_sums,
        "tokenCountsBytes": token_counts_bytes,
        "payloadOffset": token_counts_offset + token_counts_bytes,
    }


def peer_payload_size(base, m, h, topk, weight_bytes=BF16_BYTES):
    """PeerPayloadLayout：routeIndex + weight + return + topkGrad（plan_bf16.cpp 用它算 peerBytes）。"""
    route_index = base + m * h * BF16_BYTES
    weight = route_index + align_up(m * topk * INT32_BYTES, ALIGN_512)
    ret = weight + align_up(m * topk * weight_bytes, ALIGN_512)
    topk_grad = ret + m * topk * h * BF16_BYTES
    total = topk_grad + align_up(m * topk * weight_bytes, ALIGN_512)
    return {"routeIndex": route_index - base, "weight": weight - route_index,
            "return": ret - weight, "topkGrad": topk_grad - ret,
            "total": total - base, "upToTopkGrad": topk_grad - base}


def shmem_size(m, k, topk, ep, epr, route_weight_before_gmm2=True,
               weight_bytes=BF16_BYTES, compact=None):
    """BF16 训练 SHMEM 需要量：72KiB 头 + 紧凑计数 + 计数表 + PeerPayloadLayout。"""
    if compact is None:
        compact = ep >= COMPACT_MIN_EP
    layout = compact_count_layout(ep, epr, compact)
    payload = peer_payload_size(layout["payloadOffset"], m, k, topk, weight_bytes)
    return layout["payloadOffset"] + (payload["total"] if route_weight_before_gmm2 else payload["upToTopkGrad"])


def backward_dispatch_progress_size(experts, sources, producers, saved_mode):
    """BackwardDispatchProgressLayout（64B/槽；仅非 none 模式有）。"""
    if not (use_gm_for_backward_data_grad(saved_mode) and experts):
        return 0
    return align_up(experts * sources * producers * 64, 64)


def backward_workspace_size(epr, r, k, n, ep, saved_mode=SAVE_BF16,
                            aic=DEFAULT_AIC, split_experts=None):
    """BackwardWorkspaceLayout（新仓）。

    参数：epr=local experts，r=maxRecvTokens，k=hidden，n=2F，
          ep=world size，aic=AI core 数（决定 producers=blockNumPerEP）。
    """
    recompute = saved_mode == SAVE_NONE
    gm_data_grad = use_gm_for_backward_data_grad(saved_mode)
    f = n // 2

    offset = align_up(2 * r * k, ALIGN_512)                    # dO [r, k]
    if recompute:
        offset += align_up(r * n, ALIGN_512)                   # recomputeGate
    if recompute:
        offset += align_up(r * n, ALIGN_512)                   # recomputeUp
    offset += align_up(2 * r * n, ALIGN_512)                   # dZ [r, n]
    offset += align_up(r * n, ALIGN_512)                       # Hp [r, n]
    offset += align_up(4 * r * backward_topk_partial_count(saved_mode, f), ALIGN_512)  # dpProducts
    offset += align_up(4 * epr * 16, ALIGN_128)                # control / dispatchReady
    offset += align_up(4, ALIGN_32)                            # aivReady
    offset += align_up(4, ALIGN_32)                            # hDone
    offset += align_up(4 * epr * 16, ALIGN_128)                # dhpReady
    offset += align_up(4 * epr * 16, ALIGN_128)                # dzReady
    offset += align_up(4 * epr * 16, ALIGN_128)                # dxReady
    if gm_data_grad:
        offset += align_up(4 * epr * 16, ALIGN_128)            # doReleased

    if split_experts is None:
        split_experts = default_backward_dispatch_split_experts(saved_mode, epr)
    producers = max(1, aic // ep) if ep > 0 else 1
    total = align_up(offset, 64) + backward_dispatch_progress_size(split_experts, ep, producers, saved_mode)
    if split_experts and use_backward_staged_wave(saved_mode, f):
        total = align_up(total, 64) + ((r + 127) // 128 + epr) * 64
    return total


# ---------- CLI ----------
SHAPES = [
    dict(m=4096, k=7168, n=8192, topk=16, ep=8, epr=4, r=65536),
    dict(m=33, k=256, n=512, topk=4, ep=3, epr=3, r=17),
    dict(m=512, k=5120, n=3072, topk=6, ep=8, epr=16, r=512 * 6),
    dict(m=1024, k=1024, n=1024, topk=1, ep=1, epr=1, r=1024),
    dict(m=4096, k=3584, n=6144, topk=16, ep=8, epr=16, r=524288),
    dict(m=8192, k=4096, n=6144, topk=8, ep=16, epr=8, r=65536),
]


def dump(weight_bytes=BF16_BYTES):
    out = []
    for s in SHAPES:
        m, k, n, topk, ep, epr, r = s["m"], s["k"], s["n"], s["topk"], s["ep"], s["epr"], s["r"]
        row = dict(s, weight_bytes=weight_bytes)
        row["saved_none"] = training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_NONE, embed_dispatch_output=False)
        row["saved_bf16"] = training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_BF16, embed_dispatch_output=False)
        row["saved_mxfp8"] = training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_MXFP8, embed_dispatch_output=False)
        row["saved_mxfp8_clip"] = training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_MXFP8,
                                                            save_clip_masks=True, embed_dispatch_output=False)
        row["saved_mgroup_bf16"] = training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_BF16,
                                                             embed_dispatch_output=True, weight_bytes=weight_bytes)
        row["dispatch_output"] = dispatch_output_size(k, r, weight_bytes)
        row["transient_normal"] = forward_transient_size(n, epr, r)
        row["backward_none"] = backward_workspace_size(epr, r, k, n, ep, SAVE_NONE)
        row["backward_saved"] = backward_workspace_size(epr, r, k, n, ep, SAVE_BF16)
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--weight-bytes", type=int, default=BF16_BYTES, help="2=BF16, 4=FP32 TopK 权重")
    a = ap.parse_args()
    if a.json:
        print(json.dumps(dump(a.weight_bytes)))
    else:
        MiB = 1024 ** 2
        for row in dump(a.weight_bytes):
            print(f"\n=== M{row['m']} K{row['k']} N{row['n']} T{row['topk']} EP{row['ep']} EPR{row['epr']} R{row['r']} ===")
            for key in ("saved_none", "saved_bf16", "saved_mxfp8", "saved_mxfp8_clip",
                        "saved_mgroup_bf16", "dispatch_output", "transient_normal",
                        "backward_none", "backward_saved"):
                v = row[key]
                print(f"  {key:<20} {v:>14,} B   {v / MiB:10.2f} MiB")
