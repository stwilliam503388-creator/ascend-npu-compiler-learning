# kernel-6: GELU

比 sigmoid 复杂一档的激活函数，用 tanh 多项式近似高斯 CDF。

## Ascend NPU 映射

全部 Vector Unit 逐元素。

## 多项式近似的意义

`GELU(x) = x · Φ(x)` 需要算高斯累积分布函数（查表或精确计算都昂贵）。

实际用 `tanh` 近似：
```
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Ascend 上大量数学函数（erf, asin, acos 等）都走类似的"多项式近似 + 逐元素求值"路径。详见 [Phase 4 文档 02](../../../docs/triton-ascend/02-Triton到Ascend全链路实战.md) 的 Decomposition Pass 案例。

## 参考

FlagGems `flag_gems/ops/gelu.py` — 含前向+反向。
