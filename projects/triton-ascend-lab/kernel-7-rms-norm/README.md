# kernel-7: RMS Norm

归一化模式代表。当前 lab 首次覆盖 normalization。

## Ascend NPU 映射

全部 Vector Unit，无 Cube 参与。

## RMS Norm vs LayerNorm

LayerNorm: mean(x) → var(x) → (x-mean)/sqrt(var) → scale+shift
RMS Norm:  rms(x²)  →            x/rms        → scale

少一个 mean 归约，快 ~10-15%。LLaMA/Mistral 等主流模型均使用 RMSNorm。

## 参考

FlagGems `flag_gems/ops/rms_norm.py` — 含 loop kernel 分 tile 实现大 D 场景。
