"""
kernel-4: Fused LayerNorm
融合算子 — 理解算子融合 + SIMT/SIMD 双模

Ascend NPU 对照:
  不融合: load→mean→store→load→var→store→load→norm→store (6 次 gm↔ub 搬运)
  融合后: load→mean→var→norm→store                       (2 次 gm↔ub 搬运)

  阶段 1 (归约 mean/var)  → Vector Unit
  阶段 2 (归一化计算)     → Vector Unit (逐元素)
  Cube Unit 闲置          → 正常，layernorm 不需要矩阵乘法
"""

import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr,          # 输入 [N, D]
    weight_ptr,     # gamma [D]
    bias_ptr,       # beta [D]
    output_ptr,     # 输出 [N, D]
    D,              # 隐藏维度
    eps: tl.constexpr,     # 防止除零
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * D
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + D

    # Step 1: gm→ub, 加载一行
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Step 2: 在 ub 上计算 mean (归约)
    mean = tl.sum(x, axis=0) / D

    # Step 3: 在 ub 上计算 var (归约)
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D

    # Step 4: normalize (逐元素)
    # rstd = reciprocal of stddev: 1/sqrt(var+eps), 用乘法代替除法更快
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Step 5: scale + shift (逐元素, ponytail: 融合在同一个 kernel 里)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    output = x_norm * weight + bias

    # Step 6: ub→gm
    tl.store(output_ptr + offsets, output, mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """CPU 上的 Triton fused layernorm"""
    output = torch.empty_like(x)
    N, D = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(D), 4096)
    grid = (N,)
    layernorm_kernel[grid](x, weight, bias, output, D, eps, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    N, D = 256, 768
    x = torch.randn(N, D, dtype=torch.float32)
    weight = torch.randn(D, dtype=torch.float32)
    bias = torch.randn(D, dtype=torch.float32)

    output_triton = layernorm(x, weight, bias)
    output_torch = torch.nn.functional.layer_norm(x, (D,), weight, bias, eps=1e-5)

    assert torch.allclose(output_triton, output_torch, atol=1e-4), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    print(f"✅ Fused LayerNorm passed. Shape: ({N}, {D})")
