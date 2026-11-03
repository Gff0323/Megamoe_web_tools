#!/usr/bin/env python3
"""导出 MegaMoe workspace 布局参考值(JSON)，供 verify_model.mjs 对账网页显存模型。

权威来源：
  - test/megamoe_bf16_test/test_workspace_layout.py   （saved 四档 / transient / UDMA / backward / python 对称窗）
  - kernels/mega_moe_bf16/include/mega_moe_training_workspace.h   （saved gate/up 追加区 + Dispatch tile ready）
  - csrc/ops/mega_moe_bf16_mgroup.cpp                （MGroup ready 追加区）
  - kernels/mega_moe_bf16/include/mega_moe_base.h    （PeermemInfo：普通/MGroup SHMEM 布局）

用法：python export_layout_reference.py > layout_ref.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "megamoe_bf16_test"))
from test_workspace_layout import (  # noqa: E402
    training_saved_state_size,
    forward_transient_workspace_size,
    workspace_sizes,
    udma_workspace_sizes,
    backward_workspace_size,
    dispatch_output_size,
    align_bytes,
)


def ready_normal(epr, r):
    """ForwardDispatchTileReadyLayout：E·⌈R/256⌉ 槽 × 16 int32，Align512。"""
    return align_bytes(epr * ((r + 255) // 256) * 16 * 4)


def ready_mgroup(epr, r):
    """csrc/ops/mega_moe_bf16_mgroup.cpp：E·⌈R/256⌉ 槽 × 64B，Align128。"""
    slots = epr * ((r + 255) // 256)
    return (slots * 64 + 127) // 128 * 128


def shmem_normal(m, k, topk, ep, epr):
    """普通/MGroup SHMEM 布局需要量（mega_moe_base.h + training_workspace.h）。

    72KiB 同步头（60KiB 旧头 + 4KiB rank 同步 + 8KiB AIV 生成计数）；
    EP≥512 且 1≤EPR≤8 时自动加紧凑计数收件箱（Align512(EP·64)+Align512(288)）；
    计数表 + ptrA0/ptrD 2×[M·T,K] + routed 权重槽。"""
    s = 72 * 1024
    if 512 <= ep <= 1024 and 0 < epr <= 8:
        s += align_bytes(ep * 64) + align_bytes(72 * 4)
    s += ep * ((ep * epr + 127) // 128 * 128) * 4
    s += 2 * m * topk * k * 2            # ptrA0 + ptrD 各 [M·T,K] BF16
    s += align_bytes(2 * m * topk)       # ptrRoutedTopkWeights
    return s


SHAPES = [
    dict(m=4096, k=7168, n=8192, topk=16, ep=8, epr=4, r=65536),      # docs 默认锚点 shape
    dict(m=33, k=256, n=512, topk=4, ep=3, epr=3, r=17),              # 奇数 M / 小 shape / 非整除
    dict(m=512, k=5120, n=3072, topk=6, ep=8, epr=16, r=512 * 6),     # 均衡容量
    dict(m=1024, k=1024, n=1024, topk=1, ep=1, epr=1, r=1024),        # 极小 EP=1
    dict(m=4096, k=3584, n=6144, topk=16, ep=8, epr=16, r=4096 * 8 * 16),  # UDMA 最坏容量 shape
    dict(m=8192, k=4096, n=6144, topk=8, ep=16, epr=8, r=8192 * 8),   # 大 EP
]

out = []
for s in SHAPES:
    m, k, n, topk, ep, epr, r = (
        s["m"], s["k"], s["n"], s["topk"], s["ep"], s["epr"], s["r"])
    base = forward_transient_workspace_size(n, epr, r)
    _, sym_py = workspace_sizes(m, k, n, topk, ep, epr, r)
    ud_local, ud_sym = udma_workspace_sizes(m, k, n, topk, ep, epr, r)
    out.append({**s,
        # saved gate/up 四档（train PR 后：普通前向 embedDispatchOutput=false，
        # saved 不含 packed X / dispatched routed 权重——它们在独立 dispatch_output）
        "saved_none": training_saved_state_size(
            m, k, topk, ep, epr, r, n, save_gate_up=False,
            embed_dispatch_output=False),
        "saved_bf16": training_saved_state_size(
            m, k, topk, ep, epr, r, n, save_gate_up=True,
            embed_dispatch_output=False),
        "saved_mxfp8": training_saved_state_size(
            m, k, topk, ep, epr, r, n, save_mxfp8=True, save_gate_up=True,
            embed_dispatch_output=False),
        "saved_mxfp8_clip": training_saved_state_size(
            m, k, topk, ep, epr, r, n, save_mxfp8=True, save_gate_up=True,
            save_clip_masks=True, embed_dispatch_output=False),
        # MGroup 前向仍用默认 embedDispatchOutput=true（mega_moe_bf16_mgroup.cpp）
        "saved_mgroup_bf16": training_saved_state_size(
            m, k, topk, ep, epr, r, n, save_gate_up=True,
            embed_dispatch_output=True),
        # dispatch 输出独立张量（packed tokens + packed routed 权重；与模式无关）
        "dispatch_output": dispatch_output_size(k, r),
        # transient 三变体
        "transient_normal": base + ready_normal(epr, r),
        "transient_mgroup": base + ready_mgroup(epr, r),
        "transient_udma": ud_local,
        # backward：none 模式含 recomputeGate/Up，保存模式不含
        "backward_none": backward_workspace_size(r, k, n, epr, "none"),
        "backward_saved": backward_workspace_size(r, k, n, epr, "bf16"),
        # SHMEM
        "shmem_normal_cpp": shmem_normal(m, k, topk, ep, epr),
        "shmem_symmetric_py": sym_py,
        "shmem_udma": ud_sym})

print(json.dumps(out))
