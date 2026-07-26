# kernel-1: Vector Add

最简单的 Triton kernel。13 行有效代码。

## Ascend NPU 映射

```
Python Triton              Ascend NPU IR (hivm.hir)
─────────────────────      ──────────────────────────
tl.load(x_ptr + offs)  →   hivm.hir.load  ins(%gm_x) outs(%ub_x)
tl.load(y_ptr + offs)  →   hivm.hir.load  ins(%gm_y) outs(%ub_y)
output = x + y         →   hivm.hir.vadd  ins(%ub_x, %ub_y) outs(%ub_c)
tl.store(out_ptr + offs)→  hivm.hir.store ins(%ub_c) outs(%gm_out)
```

## 数据流

```
  HBM (gm)          L1 Buffer (ub)         Vector Unit
     │                    │                     │
  [x] ──load──→ [ub_x] ──┤                     │
  [y] ──load──→ [ub_y] ──┤                     │
                          ├─ vadd ──────────→ [ub_c]
                          │                     │
  [out] ←──store── [ub_c] ┘                     │
```

## 为什么 BLOCK_SIZE=1024？

- 3 个 buffer (x, y, out) × 1024 × 4B (float32) = 12KB
- 远小于 L1 (1MB)，安全
- 最大 BLOCK_SIZE ≈ 1MB / (3 × 4B) = 87,381 → 实际选 1024 为 2 的幂

## 训练你的直觉

- `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` — 这是 tile-based 编程的核心模式
- `mask = offsets < n_elements` — 必须写！最后一块可能不满
- `tl.load` / `tl.store` — 所有 Triton kernel 都这样开头和结尾
