"""
kernel-5: Sigmoid
逐元素激活函数 — 最简单的非线性和最常见的 ML 激活函数

Ascend NPU 对照:
  tl.exp(-x)    → Vector Unit 逐元素 exp
  1 / (1 + z)   → Vector Unit 逐元素除法
  全程纯 Vector Unit，不需要 Cube Unit

参考: FlagGems flag_gems/ops/sigmoid.py
"""

import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # gm → ub
    x = tl.load(x_ptr + offsets, mask=mask)

    # sigmoid(x) = 1 / (1 + exp(-x))
    # ponytail: 直接用公式，不做分段近似（sigmoid 天然数值稳定，不需要类似 softmax 的 max 技巧）
    output = 1.0 / (1.0 + tl.exp(-x))

    # ub → gm
    tl.store(output_ptr + offsets, output, mask=mask)


def sigmoid(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(100000, dtype=torch.float32)

    output_triton = sigmoid(x)
    output_torch = torch.sigmoid(x)

    assert torch.allclose(output_triton, output_torch, atol=1e-5), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    # sigmoid 输出范围 [0, 1]
    assert (output_triton >= 0).all() and (output_triton <= 1).all()
    print(f"✅ Sigmoid passed. Size={x.numel()}, range=[{output_triton.min():.4f}, {output_triton.max():.4f}]")
