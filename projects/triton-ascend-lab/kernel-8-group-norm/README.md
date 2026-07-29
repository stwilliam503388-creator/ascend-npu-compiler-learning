# kernel-8: Group Norm

归一化进阶。双维度归约（C_per_group + HW），2D grid（N × G）。

## Ascend NPU 映射

| Python | Ascend |
|--------|--------|
| 沿 C_per_group + HW 双重循环 | Vector Unit 分段归约 |
| `acc_sum / numel` | Vector Unit 标量除法 |
| `(x-mean)*rstd*w+b` | Vector Unit 逐元素（广播） |

## 归一化家族对比

| 算子 | 归一化维度 | Grid | 适用 |
|------|----------|------|------|
| LayerNorm | (C) | 1D (N) | Transformer 默认 |
| RMSNorm | (C) | 1D (N) | LLaMA/Mistral |
| InstanceNorm | (HW) | 1D (N×C) | 风格迁移 |
| **GroupNorm** | **(C/G, HW)** | **2D (N, G)** | 小 batch + 检测 |

## 参考

FlagGems `flag_gems/ops/group_norm.py` — 含 forward + backward。
