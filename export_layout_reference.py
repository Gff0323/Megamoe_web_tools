#!/usr/bin/env python3
"""导出布局参考值(JSON)，供 verify_model.mjs 对账网页显存模型。

数据来源（本仓自包含，无需 MegaMoe 源码）：
  layout_spec.py —— 复刻 DeepEP main @ e18a161 的
                    mega_moe_bf16_layout.hpp / mega_moe_workspace_backward.hpp
                    （SHMEM 走 BF16 训练 MTE 路径：plan_bf16.cpp 的
                      CompactTokenCountExchangeLayout + PeerPayloadLayout）

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
    shmem_size,
    training_saved_state_size,
    ALIGN_128, ALIGN_256, ALIGN_512,
)


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
        # SHMEM：BF16 训练 MTE 路径（三变体同布局）
        "shmem_normal_cpp": shmem_size(m, k, topk, ep, epr),
        "shmem_symmetric_py": shmem_size(m, k, topk, ep, epr),
        "shmem_udma": shmem_size(m, k, topk, ep, epr),
    })

print(json.dumps(out))
