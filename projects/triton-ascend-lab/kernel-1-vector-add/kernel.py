"""
kernel-1: Vector Add
最简 Triton kernel — 理解 block/grid/program 三层抽象

Ascend NPU 对照:
  tl.load(ptr)    → hivm.hir.load  ins(gm) outs(ub)   (gm→ub 搬运)
  x + y           → hivm.hir.vadd  ins(ub,ub) outs(ub) (在 ub 上计算)
  tl.store(ptr)   → hivm.hir.store ins(ub) outs(gm)    (ub→gm 搬运)

关键: BLOCK_SIZE 不能超过 ub 容量 (L1 1MB / 3 buffers / sizeof(dtype))
"""

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,          # 输入向量 A (在 HBM/gm 中)
    y_ptr,          # 输入向量 B (在 HBM/gm 中)
    output_ptr,     # 输出向量 C (在 HBM/gm 中)
    n_elements,     # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数
):
    # Step 1: 定位这个 program 在 grid 中的位置
    pid = tl.program_id(axis=0)

    # Step 2: 计算这个 program 负责的数据范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Step 3: 边界保护 — 最后一个 block 可能超出数组
    mask = offsets < n_elements

    # Step 4: gm → ub (在 Ascend 上: hivm.hir.load)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Step 5: 在 ub 上计算 (在 Ascend 上: hivm.hir.vadd, 走 Vector Unit)
    output = x + y

    # Step 6: ub → gm (在 Ascend 上: hivm.hir.store)
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    """CPU 上的 Triton vector_add 调用入口"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


# 自检
if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, dtype=torch.float32)
    y = torch.rand(size, dtype=torch.float32)

    # Triton kernel 结果
    output_triton = add(x, y)
    # Torch 参考结果
    output_torch = x + y

    # 验证
    assert torch.allclose(output_triton, output_torch, atol=1e-5), \
        f"Max diff: {(output_triton - output_torch).abs().max()}"
    print(f"✅ Vector add passed. Size={size}, output shape={output_triton.shape}")
