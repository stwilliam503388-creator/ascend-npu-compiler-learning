"""
kernel-3: Matmul (Tiled)
矩阵乘法 — 理解 tile 切分 + Cube Unit 利用 + 沿 K 维度滑动累加

Ascend NPU 对照:
  tl.dot(a, b)      → Cube Unit 16×16 矩阵乘（这是唯一走 Cube 的操作）
  循环沿 K 维度      → tile 切分，每次加载一个 BLOCK_K 大小的子矩阵
  累加器 acc         → 在 L1 中保持，减少 gm↔ub 搬运

关键: BLOCK_M × BLOCK_N × 3 buffers × sizeof(dtype) < L1 1MB
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,       # 矩阵 A, B, C (行优先)
    M, N, K,                   # A: M×K, B: K×N, C: M×N
    stride_am, stride_ak,      # A 的步长
    stride_bk, stride_bn,      # B 的步长
    stride_cm, stride_cn,      # C 的步长
    BLOCK_M: tl.constexpr,     # M 维度的 tile 大小
    BLOCK_N: tl.constexpr,     # N 维度的 tile 大小
    BLOCK_K: tl.constexpr,     # K 维度的 tile 大小
):
    # 2D program grid: pid_m 沿 M 维度, pid_n 沿 N 维度
    # 每个 program 负责一个 BLOCK_M × BLOCK_N 的输出 tile
    pid_m = tl.program_id(0)   # 行方向索引 (第几行 tile)
    pid_n = tl.program_id(1)   # 列方向索引 (第几列 tile)

    # 这个 program 负责的 M×N tile 的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ponytail: float32 累加器 (即使输入是 float16)，避免精度损失
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 维度滑动
    for k in range(0, K, BLOCK_K):
        # 加载 A 的一个 tile [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        # 加载 B 的一个 tile [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # 边界保护
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # ← 走 Cube Unit
        acc += tl.dot(a, b)

    # 写回结果
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Triton tiled matmul"""
    assert a.shape[1] == b.shape[0], "Dimension mismatch"
    M, K = a.shape
    K2, N = b.shape

    c = torch.empty((M, N), dtype=torch.float32)

    # tile 大小: ponytail 默认值，17³×4B ≈ 20KB，L1 有余量
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 128, 256, 64
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)

    output_triton = matmul(a, b)
    output_torch = torch.mm(a, b)

    assert torch.allclose(output_triton, output_torch, atol=1e-2), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    print(f"✅ Matmul passed. Shape: {M}×{K} @ {K}×{N} = {M}×{N}")
