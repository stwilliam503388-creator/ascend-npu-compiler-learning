# kernel-2: Softmax

归约类算子代表。3 阶段算法：max → exp → sum/div。

## Ascend NPU 映射

| Python | TTIR | Ascend 执行 |
|--------|------|-----------|
| `tl.load(x, mask, other=-inf)` | `tt.load` | hivm.hir.load (gm→ub) |
| `tl.max(x, axis=0)` | `tt.reduce` | Vector Unit 多阶段归约 |
| `tl.exp(x - x_max)` | `arith.subf` + `math.exp` | Vector Unit 逐元素 |
| `tl.sum(x_exp, axis=0)` | `tt.reduce` | Vector Unit 多阶段归约 |
| `x_exp / x_sum` | `arith.divf` | Vector Unit 逐元素 |
| `tl.store(out, mask=mask)` | `tt.store` | hivm.hir.store (ub→gm) |

## 为什么分 3 阶段？

```python
# 错误写法（数值不稳定）：
output = tl.exp(x) / tl.sum(tl.exp(x), axis=0)
# exp(1000) = inf → 除以 inf = NaN

# 正确写法（数值稳定）：
x_max = tl.max(x)         # 找到最大值
x_stable = x - x_max      # 所有值 ≤ 0
x_exp = tl.exp(x_stable)  # 最大 = 1.0，其余 [0, 1]
output = x_exp / tl.sum(x_exp)
```

## Ascend 归约的代价

归约（max/sum）需要跨元素同步。在 Ascend Vector Unit 上：
- 小块 (≤1024) 归约：一次 warp 级操作，快
- 大块 (>4096) 归约：分 tile 做 partial reduction，有同步开销
- 所以 BLOCK_SIZE 用 `min(next_power_of_2(n_cols), 4096)` 避免过大

## 训练你的直觉

- `other=-float('inf')` — mask 外元素设为 -inf，不影响 max
- `tl.max(x, axis=0)` 对 1D tensor 归约为标量
- softmax 的三个阶段是所有归约类算子的模板（rms_norm, log_softmax 同理）
