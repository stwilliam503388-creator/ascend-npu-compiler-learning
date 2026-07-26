# kernel-3: Matmul (Tiled)

矩阵乘法的 tile 切分实现。唯一用到 Cube Unit 的 kernel。

## Ascend NPU 映射

| Python | TTIR | Ascend 执行 |
|--------|------|-----------|
| `tl.load(a_ptrs, mask)` | `tt.load` | hivm.hir.load (gm→ub) |
| `tl.dot(a, b)` | `tt.dot` | **Cube Unit 16×16 矩阵乘** |
| `acc += tl.dot(a,b)` | 循环累加 | 每次调 Cube，累加在 L1 中 |

## 为什么 tile？

```
A [M×K] × B [K×N] = C [M×N]

不分 tile: 一次性 load 整个 A 和 B (太大，L1 装不下)
分 tile:   每次 load A[M×BLOCK_K] + B[BLOCK_K×N]
           Cube Unit 算一块，累加到 C[M×N]
           沿 K 滑动，直到全部算完
```

## Tile 大小选择

```
BLOCK_M=16, BLOCK_N=16, BLOCK_K=32
ub 占用 = (16×32 + 32×16 + 16×16) × 4B = (512 + 512 + 256) × 4B = 5KB
L1 = 1MB → 5KB 非常安全
最大可设: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
          (128×32 + 32×128 + 128×128)×4B = 82KB < 1MB
```

## 沿 K 维度滑动

```python
for k in range(0, K, BLOCK_K):
    a_tile = A[pid_m*16 : (pid_m+1)*16,  k : k+32]   # gm→ub
    b_tile = B[k : k+32,  pid_n*16 : (pid_n+1)*16]   # gm→ub
    c_tile += a_tile @ b_tile                          # Cube Unit
```

## 训练你的直觉

- `tl.dot()` 是 Triton 中唯一映射到 Cube Unit 的操作
- float32 累加器：即使输入是 float16，用 float32 累加避免精度损失
- 循环沿 K 维度：这是所有 tile-based matmul 的核心模式
