#!/usr/bin/env python3
"""导出布局参考值(JSON)，供 verify_model.mjs 对账网页显存模型。

数据来源（本仓自包含，无需 MegaMoe 源码）：
  layout_spec.py —— 复刻 DeepEP xx/main @ e18a161 的
                    mega_moe_bf16_layout.hpp / mega_moe_workspace_backward.hpp

未移植项（SHMEM/peermem、UDMA 变体）仍沿用旧口径的本地公式（shmem_normal），
verify_model.mjs 会对这些项单独标注"未对齐新仓"。

用法：
  python export_layout_reference.py > ref/layout_ref.json
"""
import json

from layout_spec import (
    BF16_BYTES, SAVE_BF16, SAVE_MXFP8, SAVE_NONE,
    align_up,
    backward_workspace_size,
    dispatch_output_size,
    forward_transient_size,
    training_saved_state_size,
    ALIGN_128, ALIGN_256, ALIGN_512,
)


def shmem_common(m, k, topk, ep, epr):
    """对称窗公共部分（新仓 72KiB 同步头 + 计数表 + 2×[M·T,K] 数据槽），未含 routed 槽与收件箱。"""
    s = 72 * 1024
    s += ep * align_up(ep * epr, 128) * 4
    s += 2 * m * topk * k * 2            # ptrA0 + ptrD
    return s


def shmem_normal(m, k, topk, ep, epr):
    """普通/MGroup SHMEM（旧仓口径，待对齐新仓 peermem）：+ routed 槽 + 紧凑收件箱。"""
    s = shmem_common(m, k, topk, ep, epr)
    s += align_up(2 * m * topk, ALIGN_512)
    if 512 <= ep <= 1024 and 0 < epr <= 8:
        s += align_up(ep * 64, ALIGN_512) + align_up(72 * 4, ALIGN_512)
    return s


def shmem_udma(m, k, topk, ep, epr, r):
    """UDMA SHMEM（旧仓口径，待对齐新仓 URMA peermem）：60KiB 控制 + 计数表 + 2×数据槽
    + 信号区 + [R,K] 激活接收区 + routed 槽。"""
    s = 60 * 1024
    s += ep * align_up(ep * epr, 128) * 4
    s += 2 * m * topk * k * 2
    s += align_up(8 * ep * epr, ALIGN_512) + 512
    s += align_up(2 * r * k, ALIGN_512)
    s += align_up(2 * m * topk, ALIGN_512)
    return s


SHAPES = [
    dict(m=4096, k=7168, n=8192, topk=16, ep=8, epr=4, r=65536),
    dict(m=33, k=256, n=512, topk=4, ep=3, epr=3, r=17),
    dict(m=512, k=5120, n=3072, topk=6, ep=8, epr=16, r=512 * 6),
    dict(m=1024, k=1024, n=1024, topk=1, ep=1, epr=1, r=1024),
    dict(m=4096, k=3584, n=6144, topk=16, ep=8, epr=16, r=4096 * 8 * 16),
    dict(m=8192, k=4096, n=6144, topk=8, ep=16, epr=8, r=8192 * 8),
]

out = []
for s in SHAPES:
    m, k, n, topk, ep, epr, r = s["m"], s["k"], s["n"], s["topk"], s["ep"], s["epr"], s["r"]
    transient = forward_transient_size(n, epr, r)
    out.append({
        **s,
        # saved 四档（普通前向 embedDispatchOutput=false）
        "saved_none": training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_NONE, embed_dispatch_output=False),
        "saved_bf16": training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_BF16, embed_dispatch_output=False),
        "saved_mxfp8": training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_MXFP8, embed_dispatch_output=False),
        "saved_mxfp8_clip": training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_MXFP8,
                                                       save_clip_masks=True, embed_dispatch_output=False),
        # MGroup 前向仍嵌 packed X
        "saved_mgroup_bf16": training_saved_state_size(m, k, topk, ep, epr, r, n, SAVE_BF16,
                                                       embed_dispatch_output=True),
        "dispatch_output": dispatch_output_size(k, r),
        # transient：新仓三变体共用同一 ForwardTransientWorkspaceLayout
        "transient_normal": transient,
        "transient_mgroup": transient,
        "transient_udma": transient,
        # backward（新仓 BackwardWorkspaceLayout）
        "backward_none": backward_workspace_size(epr, r, k, n, ep, SAVE_NONE),
        "backward_saved": backward_workspace_size(epr, r, k, n, ep, SAVE_BF16),
        # SHMEM：未对齐新仓，仍用旧口径
        "shmem_normal_cpp": shmem_normal(m, k, topk, ep, epr),
        "shmem_symmetric_py": shmem_common(m, k, topk, ep, epr),
        "shmem_udma": shmem_udma(m, k, topk, ep, epr, r),
    })

print(json.dumps(out))
