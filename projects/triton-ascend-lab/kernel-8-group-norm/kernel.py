"""
kernel-8: Group Norm
归一化算子进阶 — 分组归一化，比 LayerNorm/InstanceNorm 更灵活

GroupNorm: 将 channels 分为 G 组，每组内独立归一化
用于小 batch 场景（batch_size=2/4 时比 BatchNorm 稳定）

Ascend NPU 对照:
  双维度归约 (C_per_group, HW) — 比 RMSNorm 的单一 D 维度复杂
  Vector Unit 多阶段归约 + 广播

参考: FlagGems flag_gems/ops/group_norm.py
"""

import torch
import triton
import triton.language as tl

@triton.jit
def group_norm_kernel(
    x_ptr,          # [N, C, H, W] 或 [N, C, HW]
    weight_ptr,     # [C]
    bias_ptr,       # [C]
    output_ptr,
    C,              # 通道数
    HW,             # 空间维度乘积 (H*W)
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,  # 沿 HW 维度的 tile 大小
):
    # 2D grid: (N, G) — 每个 program 处理一个 (batch, group) 对
    n_idx = tl.program_id(0)
    g_idx = tl.program_id(1)

    # 这个 group 的通道范围
    c_per_group = C // num_groups
    c_start = g_idx * c_per_group
    c_end = c_start + c_per_group

    # 沿 HW 维度遍历（用于 sum 归约）
    hw_offs = tl.arange(0, BLOCK_HW)

    # Step 1: 计算 mean (沿 C_per_group 和 HW 双维度)
    acc_sum = tl.zeros((1,), dtype=tl.float32)
    acc_sq_sum = tl.zeros((1,), dtype=tl.float32)

    numel = c_per_group * HW
    for c in range(c_start, c_end):
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + hw_offs
            mask = hw_idx < HW
            # ponytail: 两层循环 O(C × HW) 对小 tensor 适用；
            # 大 tensor 应分 tile 用 tl.atomic_add 并行归约
            flat_idx = n_idx * C * HW + c * HW + hw_idx
            x = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
            acc_sum += tl.sum(x)
            acc_sq_sum += tl.sum(x * x)

    mean = acc_sum / numel
    var = acc_sq_sum / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Step 2: normalize + scale + shift
    for c in range(c_start, c_end):
        w = tl.load(weight_ptr + c)
        b = tl.load(bias_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + hw_offs
            mask = hw_idx < HW
            flat_idx = n_idx * C * HW + c * HW + hw_idx
            x = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
            x_norm = (x - mean) * rstd * w + b
            tl.store(output_ptr + flat_idx, x_norm, mask=mask)


def group_norm(x: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    N, C = x.shape[:2]
    assert C % num_groups == 0, f"C={C} must be divisible by num_groups={num_groups}"
    HW = x.numel() // (N * C)
    x_flat = x.reshape(N * C, HW)
    output = torch.empty_like(x_flat)

    BLOCK_HW = min(triton.next_power_of_2(HW), 1024)
    grid = (N, num_groups)
    group_norm_kernel[grid](x_flat, weight, bias, output, C, HW, num_groups, eps, BLOCK_HW)
    return output.reshape_as(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 8, 32, 32
    num_groups = 4
    x = torch.randn(N, C, H, W, dtype=torch.float32)
    weight = torch.randn(C, dtype=torch.float32)
    bias = torch.randn(C, dtype=torch.float32)

    output_triton = group_norm(x, num_groups, weight, bias)

    # Torch reference: GroupNorm
    gn = torch.nn.GroupNorm(num_groups, C, eps=1e-5)
    gn.weight.data = weight
    gn.bias.data = bias
    output_torch = gn(x)

    assert torch.allclose(output_triton, output_torch, atol=1e-4), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    print(f"✅ Group Norm passed. Shape: ({N}, {C}, {H}, {W}), groups={num_groups}")
