"""
kernel-6: GELU
逐元素激活函数 — 比 sigmoid 复杂，涉及多项式近似

GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

Ascend NPU 对照:
  全部走 Vector Unit
  多项式近似是 Ascend 上激活函数的标准做法（详见文档 02 的 erf/asin 案例）

参考: FlagGems flag_gems/ops/gelu.py
"""

import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GELU tanh 近似（标准实现，比精确 Φ(x) 快 3-5x）
    # ponytail: 用 tanh 近似而非高斯 CDF；任务书里的 erf/asin 才需要精确分段
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    output = 0.5 * x * (1.0 + tl.tanh(inner))

    tl.store(output_ptr + offsets, output, mask=mask)


def gelu(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(50000, dtype=torch.float32)

    output_triton = gelu(x)
    # torch.nn.functional.gelu 默认使用 tanh 近似
    output_torch = torch.nn.functional.gelu(x, approximate='tanh')

    assert torch.allclose(output_triton, output_torch, atol=1e-4), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    # GELU 输出范围约 [-0.17, +∞)
    print(f"✅ GELU passed. Size={x.numel()}, range=[{output_triton.min():.4f}, {output_triton.max():.4f}]")
