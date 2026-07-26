# kernel-5: Sigmoid

逐元素激活函数。一行公式的 kernel，验证最基本的 Triton→Ascend 管线。

## Ascend NPU 映射

| Python | TTIR | Ascend |
|--------|------|--------|
| `tl.load(x, mask)` | `tt.load` | hivm.hir.load (gm→ub) |
| `tl.exp(-x)` | `math.exp` + `arith.negf` | Vector Unit 逐元素 |
| `1/(1+z)` | `arith.addf` + `arith.divf` | Vector Unit 逐元素 |
| `tl.store(out, mask)` | `tt.store` | hivm.hir.store (ub→gm) |

## 参考

FlagGems `flag_gems/ops/sigmoid.py` — 完整的前向+反向+inplace 实现。
