"""
kernel-7: RMS Norm
归一化算子 — 填补当前 lab 中缺失的 normalization 模式

RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
比 LayerNorm 少一步 mean 归约

Ascend NPU 对照:
  阶段 1: x² → sum → mean → sqrt (归约, Vector Unit)
  阶段 2: x / rstd * weight (逐元素, Vector Unit)
  Cube Unit 闲置 — RMS norm 不需要矩阵乘法

参考: FlagGems flag_gems/ops/rms_norm.py
"""

import torch
import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    D,               # 隐藏维度
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * D
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + D

    # Step 1: gm → ub
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Step 2: rms = sqrt(mean(x²) + eps)
    # ponytail: 直接用 sum(x²)/D + sqrt；与 FlagGems 的 prev_multiple_of 优化等效但更易读
    x_sq = x * x
    rms = tl.sqrt(tl.sum(x_sq, axis=0) / D + eps)

    # Step 3: normalize + scale
    x_norm = x / rms
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    output = x_norm * weight

    # Step 4: ub → gm
    tl.store(output_ptr + offsets, output, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    output = torch.empty_like(x)
    N, D = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(D), 4096)
    grid = (N,)
    rms_norm_kernel[grid](x, weight, output, D, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 128, 512
    x = torch.randn(N, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)

    output_triton = rms_norm(x, weight)

    # Torch 没有内置 RMSNorm，手动计算验证
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
    output_torch = x / rms * weight

    assert torch.allclose(output_triton, output_torch, atol=1e-4), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    print(f"✅ RMS Norm passed. Shape: ({N}, {D})")
