# kernel-4: Fused LayerNorm

融合算子的典范。5 个操作一个 kernel 搞定。

## Ascend NPU 映射

| Python | TTIR | Ascend 执行 |
|--------|------|-----------|
| `tl.sum(x)/D` | `tt.reduce` | Vector Unit 归约 |
| `x - mean` | `arith.subf` | Vector Unit 逐元素 |
| `tl.sum((x-mean)²)/D` | `tt.reduce` | Vector Unit 归约 |
| `1/sqrt(var+eps)` | `math.sqrt` + `arith.divf` | Vector Unit |
| `x_norm * weight + bias` | `arith.mulf` + `arith.addf` | Vector Unit |

全部走 Vector Unit — layernorm 不需要 Cube Unit。

## 融合 vs 不融合

```
不融合 (3 次 kernel 启动):
  kernel-1: load(x) → compute(mean,var) → store(norm)
  kernel-2: load(norm) → mul(weight) → store(scaled)
  kernel-3: load(scaled) → add(bias) → store(output)

  6 次 gm↔ub 搬运

融合 (1 次 kernel 启动):
  kernel: load(x,weight,bias) → compute(all) → store(output)

  2 次 gm↔ub 搬运 (load x/weight/bias, store output)
```

## 为什么 gm↔ub 搬运是瓶颈？

```
HBM 带宽 ≈ 1.2 TB/s (Ascend 910B)
L1 带宽  ≈ 数十 TB/s

一次搬运 = 受限于 HBM 带宽
减少搬运 = 数据尽量留在 L1 中
```

## 训练你的直觉

- 融合的本质：把能用 L1 算完的操作串在一起，减少回写 HBM
- `eps=1e-5` 防止除零（`var ≈ 0` 时会出问题）
- `tl.sqrt()` 是唯一有开销的非线性操作
