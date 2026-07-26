"""
kernel-2: Softmax
归约类算子 — 理解多阶段算法 + warp reduce

Ascend NPU 对照:
  tl.max(x, axis=0) → 多阶段 Vector 归约 (reduce_max)
  tl.sum(x, axis=0)  → 多阶段 Vector 归约 (reduce_sum)
  全部操作走 Vector Unit (Cube Unit 不做归约)

关键: 归约在 Ascend 上有同步开销，需要分 tile 减少同步次数
"""

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,          # 输入矩阵 (rows, cols)
    output_ptr,     # 输出矩阵 (rows, cols)
    n_cols,         # 每行的列数
    BLOCK_SIZE: tl.constexpr,  # 一次处理的元素数
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_start + n_cols

    # 加载一行
    # other=-inf: mask 外的元素设为负无穷, 不影响后续 max() 操作
    # 对比: other=0.0 会使 max 错误地选 0 而非实际最大值
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # 阶段 1: 找最大值（数值稳定性: softmax(x) = softmax(x - max)）
    x_max = tl.max(x, axis=0)

    # 阶段 2: exp(x - max)
    # ponytail: 直接用 tl.exp，不做分段近似; 30个任务书里的erf/asin才需要分段
    x_exp = tl.exp(x - x_max)

    # 阶段 3: 求和 + 归一化
    x_sum = tl.sum(x_exp, axis=0)
    output = x_exp / x_sum

    tl.store(output_ptr + offsets, output, mask=mask)


def softmax(x: torch.Tensor):
    """CPU 上的 Triton softmax"""
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
    grid = (n_rows,)
    softmax_kernel[grid](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, dtype=torch.float32)

    output_triton = softmax(x)
    output_torch = torch.softmax(x, dim=-1)

    assert torch.allclose(output_triton, output_torch, atol=1e-4), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    # 验证每行和为 1
    row_sums = output_triton.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    print(f"✅ Softmax passed. Shape={x.shape}, row sums ≈ 1.0")
